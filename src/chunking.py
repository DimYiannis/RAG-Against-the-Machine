"""
    structure-aware chunkers with exact character offsets.

    each file is read once as a single string and all offsets index into
    that exact string.
    No chunk ever spans more than max_chunk_size
    characters — every produced span passes through :_split_span,
    which enforces the cap, so a chunk can always be turned into a valid
    MinimalSource.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

MIN_SECTION_SIZE = 600

HEADER_RE = re.compile(r"^#{1,6} ", re.MULTILINE)

TEXT_EXTE = {
    ".py", ".md", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".jinja", ".cmake", ".in",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh",
    ".js", ".html", ".css",
}

# abstract syntax tree definitions to handle code smoothly
PY_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Chunk:
    """
        one retrievable span of a corpus file

        attrs:
            file_path
            first
            last
            text
    """

    file_path: str
    first: int
    last: int
    text: str


def read_text(path: Path) -> str | None:
    """
        read a file as utf-8

        args:
            path

        return:
            decode content or none
    """

    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (UnicodeDecodeError, OSError):
        return None


def iter_corpus_files(root: Path) -> list[Path]:
    """
        list indexable files under a corpus root, deterministically

        args:
            root
        return:
            sorted paths whose extension is in TEXT_EXTE
    """

    return sorted(
        path
        # matches pattern against every file/dir at any depth under root

        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_EXTE
    )


def chunk_text(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """
        chunk a file's content with the matching strategy to its type

        args:
            text
            file_path
            max_chunk_size

        return:
            chunks in file order, drop whitespace-only spans
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return chunk_python(text, file_path, max_chunk_size)
    if suffix in {".md", ".rst", ".txt"}:
        return chunk_markdown(text, file_path, max_chunk_size)
    return chunk_lines(text, file_path, max_chunk_size)


def chunk_markdown(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000
) -> list[Chunk]:
    """
        chunk markdown like text,
        treat each ATX header (``#`` .. ``######``) as the start
        of a new section, so a header always stays attached to its own
        body rather than being separated from it.
        Sections shorter than MIN_SECTION_SIZE are merged into the previous
        section (as long as the merge stays within max_chunk_size),
        since a short section on its own is too small to usefully overlap.
        Sections still over max_chunk_size after merging are split at
        paragraph, then line, boundaries.
        A file with no headers is treated as one section spanning the
        whole file, which then falls through the same oversize-split
        path as a fixed, non-overlapping window.

        args:
            text
            file_path
            max_chunk_size

        return:
            chunks in file order
    """
    # handle headers
    bounds = [match.start() for match in HEADER_RE.finditer(text)]
    if not bounds or bounds[0] != 0:
        bounds.insert(0, 0)
    bounds.append(len(text))

    # build sections then merge undersized ones
    merged: list[list[int]] = []
    for start, end in zip(bounds, bounds[1:]):
        if merged:
            prev_start, prev_end = merged[-1]
            small = (
                end - start < MIN_SECTION_SIZE
                or prev_end - prev_start < MIN_SECTION_SIZE
            )
            if small and end - prev_start <= max_chunk_size:
                merged[-1][1] = end
                continue
        merged.append([start, end])

    # enforce size cap and return
    spans: list[tuple[int, int]] = []
    for start, end in merged:
        spans.extend(_split_span(text, start, end, max_chunk_size))
    return _to_chunks(text, file_path, spans)


def chunk_python(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000
) -> list[Chunk]:
    """
        chunk python script, each top level func/class is a chunk
        a class larger than the cap is re_chunked per method, its
        header is a seperate chunk. Files that fail to parse fall
        back to line-window chunking

        args:
            text
            file_path
            max_chunk_size

        return:
            chunks
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return chunk_lines(text, file_path, max_chunk_size)
    line_starts = _line_starts(text)
    spans = _split_body(
        tree.body,
        0,
        len(text),
        text,
        line_starts,
        max_chunk_size
    )
    return _to_chunks(text, file_path, spans)


def chunk_lines(
    text: str, file_path: str, max_chunk_size: int = 2000
) -> list[Chunk]:
    """
        chunk text into windows at line boundaries
        fallback for unstructured files and unparsable python

        args:
            text
            file_path
            max_chunk_size

        return:
            chunks
    """

    spans = _split_span(text, 0, len(text), max_chunk_size)
    return _to_chunks(text, file_path, spans)


def _split_body(
    body: list[ast.stmt],
    region_start: int,
    region_end: int,
    text: str,
    line_starts: list[int],
    max_chunk_size: int,
) -> list[tuple[int, int]]:
    """
        split code into spans

        args:
            body
            region_start
            region_end
            text
            line_starts
            max_chunk_size

         return:
            spans -> [region_start, region_end)
    """
    spans: list[tuple[int, int]] = []
    cursor = region_start
    for node in body:
        if not isinstance(node, PY_DEFS):
            continue
        start, end = _node_span(node, line_starts)
        if start > cursor:
            spans.extend(_split_span(text, cursor, start, max_chunk_size))
        oversized_class = (
            isinstance(node, ast.ClassDef) and end - start > max_chunk_size
        )
        if oversized_class:
            spans.extend(
                _split_body(
                    node.body, start, end, text, line_starts, max_chunk_size
                )
            )
        else:
            spans.extend(_split_span(text, start, end, max_chunk_size))
        cursor = end
    if cursor < region_end:
        spans.extend(_split_span(text, cursor, region_end, max_chunk_size))
    return spans


def _node_span(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    line_starts: list[int],
) -> tuple[int, int]:
    """
        char span of a definition, decorators included.

        the span starts at column 0 of the first decorator's line (or the
        def/class line), so indentation stays inside the chunk.

        args:
            node: the definition node.
            line_starts

        return:
            (first, last)
    """
    first_node: ast.expr | ast.stmt = node
    if node.decorator_list:
        first_node = node.decorator_list[0]
    end_lineno = node.end_lineno if node.end_lineno else node.lineno
    end_col = node.end_col_offset if node.end_col_offset else 0
    start = line_starts[first_node.lineno - 1]
    end = line_starts[end_lineno - 1] + end_col
    return start, end


def _split_span(
    text: str, first: int, last: int, max_chunk_size: int
) -> list[tuple[int, int]]:
    """
        cut a span into pieces no longer than the cap.

        prefers cutting at a blank line,
        then any newline,
        then hard mid-line (single lines longer than the cap).

        args:
            text
            first
            last
            max_chunk_size

        return:
            (first, last)
    """

    spans: list[tuple[int, int]] = []
    start = first
    while last - start > max_chunk_size:
        window_end = start + max_chunk_size
        cut = text.rfind("\n\n", start + 1, window_end)
        if cut <= start:
            cut = text.rfind("\n", start + 1, window_end)
        if cut <= start:
            cut = window_end
        spans.append((start, cut))
        start = cut
    if start < last:
        spans.append((start, last))
    return spans


def _line_starts(text: str) -> list[int]:
    """
        char offset of every line start, index i = 0-based line i.

        args:
            text

        return:
            offsets list offset = starts[lineno - 1] converts an ast
            1-based lineno to a character position.
    """
    starts = [0]
    pos = text.find("\n", 0)
    while pos != -1:
        starts.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return starts


def _to_chunks(
    text: str, file_path: str, spans: list[tuple[int, int]]
) -> list[Chunk]:
    """
        materialize spans as chunks, dropping whitespace-only ones.

        args:
            text
            file_path
            spans: (first, last) pairs.

        returns:
            chunks whose text contains at least one non-space character.
    """
    chunks = []
    for first, last in spans:
        piece = text[first:last]
        if piece.strip():
            chunks.append((Chunk(file_path, first, last, piece)))
    return chunks

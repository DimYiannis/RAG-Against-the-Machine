"""Structure-aware chunkers with exact character offsets.

Every chunk satisfies the round-trip contract
``open(file_path, encoding="utf-8").read()[first:last] == chunk.text``:
each file is read once as a single string and all offsets index into
that exact string. No chunk ever spans more than ``max_chunk_size``
characters — every produced span passes through :func:`_split_span`,
which enforces the cap, so a chunk can always be turned into a valid
``MinimalSource``.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

#: Markdown sections smaller than this are merged into a neighbour.
MIN_SECTION_SIZE = 200

#: ATX headers (``# `` … ``###### ``) at the start of a line.
_HEADER_RE = re.compile(r"^#{1,6} ", re.MULTILINE)

#: Extensions treated as textual corpus content; everything else
#: (images, binaries) is skipped at file-selection time.
TEXT_EXTENSIONS = {
    ".py", ".md", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".jinja", ".cmake", ".in",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh",
    ".js", ".html", ".css",
}

_PY_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Chunk:
    """One retrievable span of a corpus file.

    Attributes:
        file_path: Corpus-relative path, stamped verbatim into outputs.
        first: Start offset into the file's decoded text.
        last: End offset (exclusive).
        text: The exact slice ``text[first:last]`` of the file.
    """

    file_path: str
    first: int
    last: int
    text: str


def read_text(path: Path) -> str | None:
    """Read a file as UTF-8, returning None for undecodable/unreadable.

    Args:
        path: File to read.

    Returns:
        The decoded content, or None if the file is not valid UTF-8
        or cannot be opened.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (UnicodeDecodeError, OSError):
        return None


def iter_corpus_files(root: Path) -> list[Path]:
    """List indexable files under a corpus root, deterministically.

    Args:
        root: Directory to walk (e.g. ``data/raw``).

    Returns:
        Sorted paths whose extension is in TEXT_EXTENSIONS.
    """
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS
    )


def chunk_text(
    text: str, file_path: str, max_chunk_size: int = 2000
) -> list[Chunk]:
    """Chunk one file's content with the strategy matching its type.

    Args:
        text: Full decoded file content.
        file_path: Corpus-relative path stamped into each chunk.
        max_chunk_size: Hard cap on ``last - first`` per chunk.

    Returns:
        Chunks in file order; whitespace-only spans are dropped.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return chunk_python(text, file_path, max_chunk_size)
    if suffix in {".md", ".rst", ".txt"}:
        return chunk_markdown(text, file_path, max_chunk_size)
    return chunk_lines(text, file_path, max_chunk_size)


def chunk_markdown(
    text: str, file_path: str, max_chunk_size: int = 2000
) -> list[Chunk]:
    """Chunk markdown-like text along its header structure.

    Sections start at ATX headers so a header always stays with its
    body. Undersized sections merge into their predecessor (while the
    merge stays under the cap); oversized ones are split at paragraph
    then line boundaries. A headerless file is a single section, which
    makes this degrade gracefully to fixed windows.

    Args:
        text: Full decoded file content.
        file_path: Corpus-relative path stamped into each chunk.
        max_chunk_size: Hard cap on ``last - first`` per chunk.

    Returns:
        Chunks in file order.
    """
    bounds = [match.start() for match in _HEADER_RE.finditer(text)]
    if not bounds or bounds[0] != 0:
        bounds.insert(0, 0)
    bounds.append(len(text))

    merged: list[list[int]] = []
    for start, end in zip(bounds, bounds[1:]):
        if merged:
            prev_start, prev_end = merged[-1]
            small = (end - start < MIN_SECTION_SIZE
                     or prev_end - prev_start < MIN_SECTION_SIZE)
            if small and end - prev_start <= max_chunk_size:
                merged[-1][1] = end
                continue
        merged.append([start, end])

    spans: list[tuple[int, int]] = []
    for start, end in merged:
        spans.extend(_split_span(text, start, end, max_chunk_size))
    return _to_chunks(text, file_path, spans)


def chunk_python(
    text: str, file_path: str, max_chunk_size: int = 2000
) -> list[Chunk]:
    """Chunk Python source along its top-level definitions.

    Each top-level function/class (decorators included) is one chunk;
    module-level code between definitions forms its own chunks, so the
    module docstring plus imports become a natural first chunk. A class
    larger than the cap is re-chunked per method, its header (class
    line, docstring, class-level assignments) forming a separate chunk.
    Files that fail to parse fall back to line-window chunking.

    Args:
        text: Full decoded file content.
        file_path: Corpus-relative path stamped into each chunk.
        max_chunk_size: Hard cap on ``last - first`` per chunk.

    Returns:
        Chunks in file order.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return chunk_lines(text, file_path, max_chunk_size)
    line_starts = _line_starts(text)
    spans = _split_body(
        tree.body, 0, len(text), text, line_starts, max_chunk_size
    )
    return _to_chunks(text, file_path, spans)


def chunk_lines(
    text: str, file_path: str, max_chunk_size: int = 2000
) -> list[Chunk]:
    """Chunk arbitrary text into windows at line boundaries.

    Fallback for unstructured files and unparsable Python: greedy
    windows up to the cap, cut at the last newline inside the window
    (mid-line only when a single line exceeds the cap).

    Args:
        text: Full decoded file content.
        file_path: Corpus-relative path stamped into each chunk.
        max_chunk_size: Hard cap on ``last - first`` per chunk.

    Returns:
        Chunks in file order.
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
    """Split one code region into definition spans and gap spans.

    Walks the statements of a module (or class) body: each def/class
    becomes its own span, and whatever lies between them ("gaps":
    docstrings, imports, assignments) becomes separate spans. Used
    recursively for oversized classes, where the gap before the first
    method is exactly the class-header chunk.

    Args:
        body: Statement list of the module or class.
        region_start: Char offset where this region begins.
        region_end: Char offset where this region ends (exclusive).
        text: Full file content the offsets index into.
        line_starts: Char offset of each line start (0-based lines).
        max_chunk_size: Hard cap forwarded to the span splitter.

    Returns:
        Spans tiling ``[region_start, region_end)`` in order.
    """
    spans: list[tuple[int, int]] = []
    cursor = region_start
    for node in body:
        if not isinstance(node, _PY_DEFS):
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
    """Char span of a definition, decorators included.

    The span starts at column 0 of the first decorator's line (or the
    ``def``/``class`` line), so indentation stays inside the chunk; it
    ends at the node's exact ``end_lineno``/``end_col_offset``.

    Args:
        node: The definition node.
        line_starts: Char offset of each line start (0-based lines).

    Returns:
        ``(first, last)`` character offsets.
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
    """Cut a span into pieces no longer than the cap.

    Prefers cutting at a blank line, then any newline, then hard mid-
    line (single lines longer than the cap, e.g. minified JSON). The
    pieces tile ``[first, last)`` exactly — no character is lost.

    Args:
        text: Full file content the offsets index into.
        first: Span start offset.
        last: Span end offset (exclusive).
        max_chunk_size: Maximum piece length in characters.

    Returns:
        ``(first, last)`` pairs in order.
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
    """Char offset of every line start; index i = 0-based line i.

    Args:
        text: Full file content.

    Returns:
        Offsets list; ``offset = starts[lineno - 1]`` converts an ast
        1-based ``lineno`` to a character position.
    """
    starts = [0]
    pos = text.find("\n")
    while pos != -1:
        starts.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return starts


def _to_chunks(
    text: str, file_path: str, spans: list[tuple[int, int]]
) -> list[Chunk]:
    """Materialize spans as Chunks, dropping whitespace-only ones.

    Args:
        text: Full file content the offsets index into.
        file_path: Corpus-relative path stamped into each chunk.
        spans: ``(first, last)`` pairs.

    Returns:
        Chunks whose text contains at least one non-space character.
    """
    chunks = []
    for first, last in spans:
        piece = text[first:last]
        if piece.strip():
            chunks.append(Chunk(file_path, first, last, piece))
    return chunks

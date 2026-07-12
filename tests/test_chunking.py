"""Chunking tests. The round-trip test is the project's most critical:
offsets are ground truth, and any chunking change must keep it green.
"""

import random
from pathlib import Path

import pytest

from src.chunking import (
    Chunk,
    chunk_lines,
    chunk_markdown,
    chunk_python,
    chunk_text,
    iter_corpus_files,
    read_text,
)

CORPUS_ROOT = Path("data/raw/vllm-0.10.1")
MAX = 2000

needs_corpus = pytest.mark.skipif(
    not CORPUS_ROOT.is_dir(), reason="corpus not downloaded"
)


def _assert_round_trip(text: str, chunks: list[Chunk]) -> None:
    """Every chunk slices back out of the original text, within cap."""
    for chunk in chunks:
        assert 0 <= chunk.first < chunk.last
        assert chunk.last - chunk.first <= MAX
        assert text[chunk.first:chunk.last] == chunk.text


@needs_corpus
def test_round_trip_on_real_corpus_sample() -> None:
    """Offsets round-trip on a deterministic 40-file corpus sample."""
    files = iter_corpus_files(CORPUS_ROOT)
    assert files, "corpus present but no indexable files found"
    sample = random.Random(42).sample(files, min(40, len(files)))
    checked = 0
    for path in sample:
        text = read_text(path)
        if text is None:
            continue
        rel = path.as_posix()
        chunks = chunk_text(text, rel, MAX)
        _assert_round_trip(text, chunks)
        checked += len(chunks)
    assert checked > 0


@needs_corpus
def test_round_trip_on_known_reference_files() -> None:
    """Files the subject cites verbatim chunk cleanly."""
    for rel in (
        "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
        "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
    ):
        text = read_text(Path(rel))
        assert text is not None
        chunks = chunk_text(text, rel, MAX)
        assert chunks
        _assert_round_trip(text, chunks)


def test_markdown_header_stays_with_section() -> None:
    """Each header starts a chunk containing its own body."""
    body_a = "alpha " * 60
    body_b = "beta " * 60
    text = f"# Title\n\n{body_a}\n\n## Sub\n\n{body_b}\n"
    chunks = chunk_markdown(text, "doc.md", MAX)
    _assert_round_trip(text, chunks)
    assert chunks[0].text.startswith("# Title")
    assert "alpha" in chunks[0].text
    sub = [c for c in chunks if c.text.lstrip().startswith("## Sub")]
    assert sub and "beta" in sub[0].text


def test_markdown_small_sections_merge() -> None:
    """Adjacent tiny sections collapse into one chunk."""
    text = "# A\ntiny\n# B\nalso tiny\n# C\nstill tiny\n"
    chunks = chunk_markdown(text, "doc.md", MAX)
    _assert_round_trip(text, chunks)
    assert len(chunks) == 1


def test_python_defs_and_module_gap() -> None:
    """Imports form a gap chunk; decorated def spans its decorator."""
    text = (
        '"""Mod doc."""\n'
        "import os\n"
        "\n"
        "\n"
        "@decorator\n"
        "def foo() -> int:\n"
        "    return 1\n"
        "\n"
        "\n"
        "def bar() -> int:\n"
        "    return 2\n"
    )
    chunks = chunk_python(text, "mod.py", MAX)
    _assert_round_trip(text, chunks)
    assert chunks[0].text.startswith('"""Mod doc."""')
    foo = next(c for c in chunks if "def foo" in c.text)
    assert foo.text.startswith("@decorator")
    assert any(c.text.startswith("def bar") for c in chunks)


def test_python_oversized_class_splits_per_method() -> None:
    """Class over the cap yields a header chunk plus method chunks."""
    filler = '        x = "%s"\n' % ("y" * 60)
    methods = "".join(
        f"    def m{i}(self) -> None:\n{filler * 12}" for i in range(5)
    )
    text = f'class Big:\n    """Doc."""\n\n    attr = 1\n\n{methods}'
    assert len(text) > MAX
    chunks = chunk_python(text, "big.py", MAX)
    _assert_round_trip(text, chunks)
    assert len(chunks) > 1
    assert chunks[0].text.startswith("class Big:")
    assert '"""Doc."""' in chunks[0].text
    assert any(c.text.lstrip().startswith("def m1") for c in chunks)


def test_python_syntax_error_falls_back() -> None:
    """Unparsable source still chunks via line windows."""
    text = "def broken(:\n" + "x = 1\n" * 400
    chunks = chunk_python(text, "broken.py", MAX)
    _assert_round_trip(text, chunks)
    assert chunks


def test_single_long_line_hard_split() -> None:
    """A line longer than the cap is cut mid-line, losing nothing."""
    text = "z" * (3 * MAX + 7)
    chunks = chunk_lines(text, "one_line.json", MAX)
    _assert_round_trip(text, chunks)
    assert "".join(c.text for c in chunks) == text


def test_full_coverage_of_nonblank_text() -> None:
    """Union of chunk spans covers every non-whitespace character."""
    text = (
        "# Header\n\npara one\n\n" + "word " * 500 + "\n\n## Two\nend\n"
    )
    chunks = chunk_markdown(text, "doc.md", MAX)
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.first, chunk.last))
    missing = [
        i for i, ch in enumerate(text)
        if not ch.isspace() and i not in covered
    ]
    assert missing == []

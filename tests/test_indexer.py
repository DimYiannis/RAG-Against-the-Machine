"""Indexer build/persist tests on a synthetic mini-corpus."""

from pathlib import Path

import pytest

from src.indexer import Index, build_index, load_index, save_index


@pytest.fixture()
def mini_corpus(tmp_path: Path) -> Path:
    """Two-file corpus with a shared term and a unique term."""
    root = tmp_path / "raw"
    root.mkdir()
    (root / "doc.md").write_text(
        "# Server guide\n\nConfigure the server here.\n", encoding="utf-8"
    )
    (root / "code.py").write_text(
        "def enable_lora(server: str) -> None:\n    pass\n",
        encoding="utf-8",
    )
    return root


def test_build_indexes_our_tokens(mini_corpus: Path) -> None:
    """Our tokenizer's output reaches the bm25s vocabulary.

    Guards the load-bearing property of feeding bm25s pre-tokenized
    lists: subtokens (lora) and the whole identifier (enable_lora) are
    both matchable, which raw-text indexing would lose.
    """
    index = build_index(mini_corpus, 2000, show_progress=False)
    assert index.doc_count == 2
    assert index.avgdl > 0
    vocab = index.scorer.vocab_dict
    for term in ("server", "enable_lora", "lora", "enable"):
        assert term in vocab


def test_chunk_meta_has_offsets_and_length(mini_corpus: Path) -> None:
    """Chunk metadata carries path, valid span, and token length."""
    index = build_index(mini_corpus, 2000, show_progress=False)
    for file_path, first, last, length in index.chunks:
        assert file_path.endswith((".md", ".py"))
        assert 0 <= first < last
        assert last - first <= 2000
        assert length > 0


def test_save_load_round_trip(mini_corpus: Path, tmp_path: Path) -> None:
    """A saved index loads back identical."""
    index = build_index(mini_corpus, 2000, show_progress=False)
    save_index(index, tmp_path / "processed")
    loaded = load_index(tmp_path / "processed")
    assert isinstance(loaded, Index)
    assert loaded.chunks == index.chunks
    assert loaded.avgdl == index.avgdl
    assert loaded.max_chunk_size == index.max_chunk_size
    assert loaded.scorer.vocab_dict == index.scorer.vocab_dict


def test_load_missing_index_is_clean_error(tmp_path: Path) -> None:
    """Loading before indexing raises a helpful FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="run"):
        load_index(tmp_path)


def test_build_errors_are_clean(tmp_path: Path, mini_corpus: Path) -> None:
    """Missing dir, empty dir, bad chunk size raise typed errors."""
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "nope", 2000, show_progress=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no indexable"):
        build_index(empty, 2000, show_progress=False)
    with pytest.raises(ValueError, match="positive"):
        build_index(mini_corpus, 0, show_progress=False)

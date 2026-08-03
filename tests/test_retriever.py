"""BM25 retriever tests over the bm25s-backed index.

The hand-computed score check lives on the `bm25-handrolled` branch:
with bm25s the arithmetic belongs to the library, so these assert
ranking behaviour and the contract top_k owes its callers instead.
"""

import json
from pathlib import Path

import bm25s
import pytest

from src.indexer import Index
from src.models import StudentSearchResults
from src.retriever import (
    load_dataset,
    save_results,
    search_dataset,
    to_source,
    top_k,
)


@pytest.fixture()
def tiny_index() -> Index:
    """Two chunks: doc0 about lora (3 tokens), doc1 about servers."""
    corpus_tokens = [
        ["enable_lora", "enable", "lora", "lora"],
        ["server", "config"],
    ]
    scorer = bm25s.BM25(k1=1.3, b=0.85, method="lucene")
    scorer.index(corpus_tokens, show_progress=False)
    return Index(
        max_chunk_size=2000,
        chunks=[
            ("data/raw/a.py", 0, 30, 4),
            ("data/raw/b.md", 0, 20, 2),
        ],
        scorer=scorer,
        avgdl=3.0,
    )


def test_scores_are_positive_and_descending(tiny_index: Index) -> None:
    """Results come back best-first with usable positive scores."""
    ranked = top_k(tiny_index, "lora enable server config", k=2)
    assert len(ranked) == 2
    scores = [score for _, score in ranked]
    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_matching_doc_ranks_first(tiny_index: Index) -> None:
    """Query about servers ranks the server chunk on top."""
    ranked = top_k(tiny_index, "how to config the server", k=2)
    assert ranked
    assert ranked[0][0] == 1


def test_degenerate_queries_yield_empty(tiny_index: Index) -> None:
    """Empty, whitespace, out-of-vocabulary, k=0, negative k."""
    assert top_k(tiny_index, "", 5) == []
    assert top_k(tiny_index, "   ", 5) == []
    assert top_k(tiny_index, "zzzqqq unknownterm", 5) == []
    assert top_k(tiny_index, "lora", 0) == []
    assert top_k(tiny_index, "lora", -3) == []


def test_k_caps_results(tiny_index: Index) -> None:
    """k=1 returns exactly the single best chunk."""
    ranked = top_k(tiny_index, "lora enable server", k=1)
    assert len(ranked) == 1


def test_to_source_round_trips_metadata(tiny_index: Index) -> None:
    """MinimalSource carries the chunk's exact path and span."""
    source = to_source(tiny_index, 0)
    assert source.file_path == "data/raw/a.py"
    assert source.first_character_index == 0
    assert source.last_character_index == 30


def test_search_dataset_preserves_ids_and_order(
    tiny_index: Index, tmp_path: Path
) -> None:
    """Dataset run keeps question ids/order and records k."""
    dataset_file = tmp_path / "ds.json"
    dataset_file.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {"question_id": "q1", "question": "enable lora?"},
                    {"question_id": "q2", "question": ""},
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = load_dataset(dataset_file)
    results = search_dataset(
        tiny_index, dataset, k=3, show_progress=False
    )
    assert results.k == 3
    assert [r.question_id for r in results.search_results] == ["q1", "q2"]
    assert results.search_results[0].retrieved_sources
    assert results.search_results[1].retrieved_sources == []

    target = save_results(results, tmp_path / "out", "ds.json")
    reloaded = StudentSearchResults.model_validate_json(
        target.read_text(encoding="utf-8")
    )
    assert reloaded == results


def test_load_dataset_errors_are_clean(tmp_path: Path) -> None:
    """Missing file and malformed JSON raise typed, readable errors."""
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        load_dataset(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed dataset JSON"):
        load_dataset(bad)

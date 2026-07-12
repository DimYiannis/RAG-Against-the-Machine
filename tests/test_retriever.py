"""BM25 retriever tests, including a hand-computed score check."""

import json
import math
from pathlib import Path

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
    return Index(
        max_chunk_size=2000,
        chunks=[
            ("data/raw/a.py", 0, 30, 3),
            ("data/raw/b.md", 0, 20, 2),
        ],
        postings={
            "lora": [(0, 2)],
            "enable": [(0, 1)],
            "server": [(1, 1)],
            "config": [(1, 1)],
        },
        avgdl=2.5,
    )


def test_bm25_score_matches_hand_calculation(tiny_index: Index) -> None:
    """Score for 'lora' on doc0, computed by hand with k1=1.5 b=0.75.

    idf = ln((2 - 1 + 0.5)/(1 + 0.5) + 1) = ln 2
    denom = 2 + 1.5 * (1 - 0.75 + 0.75 * 3/2.5) = 3.725
    score = ln 2 * 2 * 2.5 / 3.725
    """
    ranked = top_k(tiny_index, "lora", k=5)
    assert len(ranked) == 1
    chunk_id, score = ranked[0]
    assert chunk_id == 0
    expected = math.log(2) * 2 * 2.5 / 3.725
    assert score == pytest.approx(expected)


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

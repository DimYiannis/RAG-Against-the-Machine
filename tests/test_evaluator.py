"""Evaluator tests: IoU math, matching rules, recall aggregation."""

import json
from pathlib import Path

import pytest

from src.evaluator import (
    evaluate,
    interval_iou,
    load_results,
    source_found,
)
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def _src(path: str, first: int, last: int) -> MinimalSource:
    return MinimalSource(
        file_path=path,
        first_character_index=first,
        last_character_index=last,
    )


def test_interval_iou_values() -> None:
    """Hand-checked IoU cases."""
    assert interval_iou(0, 100, 50, 150) == pytest.approx(50 / 150)
    assert interval_iou(0, 100, 100, 200) == 0.0  # touching, disjoint
    assert interval_iou(0, 100, 0, 100) == 1.0
    assert interval_iou(0, 2000, 900, 1100) == pytest.approx(200 / 2000)


def test_iou_threshold_is_strict() -> None:
    """IoU exactly 0.05 does NOT count; just above does."""
    ref = _src("f.py", 0, 100)
    exactly = _src("f.py", 90, 200)  # inter 10, union 200 -> 0.05
    assert not source_found(ref, [exactly])
    just_above = _src("f.py", 89, 200)  # inter 11, union 200
    assert source_found(ref, [just_above])


def test_file_path_compared_verbatim() -> None:
    """Same span, differently written path -> no match."""
    ref = _src("data/raw/vllm-0.10.1/docs/a.md", 0, 100)
    other = _src("./data/raw/vllm-0.10.1/docs/a.md", 0, 100)
    assert not source_found(ref, [other])


def test_recall_aggregation() -> None:
    """Two questions: one fully found, one half found -> 0.75."""
    reference = RagDataset.model_validate(
        {
            "rag_questions": [
                {
                    "question_id": "q1",
                    "question": "a?",
                    "answer": "x",
                    "sources": [
                        {"file_path": "f1.py",
                         "first_character_index": 0,
                         "last_character_index": 100},
                    ],
                },
                {
                    "question_id": "q2",
                    "question": "b?",
                    "answer": "y",
                    "sources": [
                        {"file_path": "f2.py",
                         "first_character_index": 0,
                         "last_character_index": 100},
                        {"file_path": "f3.py",
                         "first_character_index": 0,
                         "last_character_index": 100},
                    ],
                },
                {"question_id": "q3", "question": "unanswered, skipped"},
            ]
        }
    )
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="a?",
                retrieved_sources=[_src("f1.py", 10, 90)],
            ),
            MinimalSearchResults(
                question_id="q2",
                question="b?",
                retrieved_sources=[_src("f2.py", 0, 100)],
            ),
        ],
        k=5,
    )
    report = evaluate(results, reference)
    assert report.recall == pytest.approx((1.0 + 0.5) / 2)
    assert report.questions_evaluated == 2
    assert report.questions_missing == 0
    assert report.sources_found == 2
    assert report.sources_total == 3


def test_missing_question_scores_zero() -> None:
    """Reference question absent from results counts as recall 0."""
    reference = RagDataset.model_validate(
        {
            "rag_questions": [
                {
                    "question_id": "q1",
                    "question": "a?",
                    "answer": "x",
                    "sources": [
                        {"file_path": "f1.py",
                         "first_character_index": 0,
                         "last_character_index": 100},
                    ],
                },
            ]
        }
    )
    results = StudentSearchResults(search_results=[], k=5)
    report = evaluate(results, reference)
    assert report.recall == 0.0
    assert report.questions_missing == 1


def test_unanswered_only_reference_is_error() -> None:
    """Nothing to grade against -> clean ValueError."""
    reference = RagDataset.model_validate(
        {"rag_questions": [{"question_id": "q", "question": "?"}]}
    )
    results = StudentSearchResults(search_results=[], k=5)
    with pytest.raises(ValueError, match="nothing to evaluate"):
        evaluate(results, reference)


def test_load_results_errors_are_clean(tmp_path: Path) -> None:
    """Missing and malformed results files raise readable errors."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_results(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"wrong": "shape"}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_results(bad)

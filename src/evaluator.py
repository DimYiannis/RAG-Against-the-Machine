"""Independent recall@k evaluation (no moulinette involvement).

A reference source counts as found when some retrieved source has the
identical file_path (verbatim string comparison, like the grader) and
the character spans overlap with IoU > IOU_THRESHOLD. Recall@k is the
fraction of a question's reference sources found, averaged over all
questions that have reference sources.
"""

from dataclasses import dataclass
from pathlib import Path

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

IOU_THRESHOLD = 0.05


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate recall figures for one results/reference pair.

    Attributes:
        recall: Mean per-question recall in [0, 1].
        questions_evaluated: Questions with reference sources that
            were matched against results.
        questions_missing: Reference questions absent from results.
        sources_found: Reference sources matched by some retrieval.
        sources_total: All reference sources over evaluated questions.
    """

    recall: float
    questions_evaluated: int
    questions_missing: int
    sources_found: int
    sources_total: int


def interval_iou(
    first_a: int, last_a: int, first_b: int, last_b: int
) -> float:
    """Intersection-over-union of two character ranges.

    Args:
        first_a: Start of range A.
        last_a: End (exclusive) of range A.
        first_b: Start of range B.
        last_b: End (exclusive) of range B.

    Returns:
        |A ∩ B| / |A ∪ B| in [0, 1]; 0 when disjoint or degenerate.
    """
    intersection = min(last_a, last_b) - max(first_a, first_b)
    if intersection <= 0:
        return 0.0
    union = max(last_a, last_b) - min(first_a, first_b)
    return intersection / union


def source_found(
    reference: MinimalSource, retrieved: list[MinimalSource]
) -> bool:
    """Whether any retrieved source matches a reference source.

    Args:
        reference: Ground-truth source.
        retrieved: Sources returned by the retriever.

    Returns:
        True if some retrieved source has the identical file_path and
        span IoU strictly above IOU_THRESHOLD.
    """
    return any(
        candidate.file_path == reference.file_path
        and interval_iou(
            reference.first_character_index,
            reference.last_character_index,
            candidate.first_character_index,
            candidate.last_character_index,
        )
        > IOU_THRESHOLD
        for candidate in retrieved
    )


def evaluate(
    results: StudentSearchResults, reference: RagDataset
) -> EvaluationReport:
    """Score search results against a reference dataset.

    Args:
        results: The student's saved search results.
        reference: Dataset whose AnsweredQuestions carry ground truth.

    Returns:
        An EvaluationReport with mean per-question recall.

    Raises:
        ValueError: If the reference has no answered questions with
            sources (nothing to evaluate against).
    """
    retrieved_by_id = {
        entry.question_id: entry.retrieved_sources
        for entry in results.search_results
    }
    per_question: list[float] = []
    missing = 0
    found_total = 0
    sources_total = 0
    for question in reference.rag_questions:
        if not isinstance(question, AnsweredQuestion) or not question.sources:
            continue
        retrieved = retrieved_by_id.get(question.question_id)
        if retrieved is None:
            missing += 1
            retrieved = []
        found = sum(
            1 for ref in question.sources if source_found(ref, retrieved)
        )
        per_question.append(found / len(question.sources))
        found_total += found
        sources_total += len(question.sources)
    if not per_question:
        raise ValueError(
            "reference dataset contains no answered questions with "
            "sources — nothing to evaluate"
        )
    return EvaluationReport(
        recall=sum(per_question) / len(per_question),
        questions_evaluated=len(per_question),
        questions_missing=missing,
        sources_found=found_total,
        sources_total=sources_total,
    )


def load_results(path: Path) -> StudentSearchResults:
    """Load and validate a StudentSearchResults JSON file.

    Args:
        path: Results file location.

    Returns:
        The parsed results.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it is not valid StudentSearchResults JSON.
    """
    if not path.is_file():
        raise FileNotFoundError(f"search results not found: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            return StudentSearchResults.model_validate_json(handle.read())
    except ValueError as exc:
        raise ValueError(
            f"malformed search results JSON in {path}: {exc}"
        ) from exc

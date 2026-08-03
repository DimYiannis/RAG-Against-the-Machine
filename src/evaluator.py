"""
    recall@k evaluation

    reference source counts as found when some retrieved source has the
    identical file_path (verbatim string comparison, like the grader) and
    the character spans overlap with IoU > IOU_THRESHOLD. 
    
    recall@k -> the fraction of a question's reference sources found, 
    averaged over all questions that have reference sources.
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
    """
        aggregate recall figures for one results/reference pair

        attrs:
            recall: mean per-question recall in 0-1
            questions_evaluated: questions with reference sources that
                were matched against results.
            questions_missing: reference questions absent from resutls
            sources_found: sources matched by some retrieval
            sources_total: all reference source over evaluated questions
    """
    recall: float
    questions_evaluated: int
    questions_missing: int
    sources_found: int
    sources_total: int

def interval_iou(
    first_a: int,
    last_a: int,
    first_b: int,
    last_b: int,
) -> float:
    """
        IoU of 2 character ranges

        args:
            first_a: start of range a
            last_a: end of range a
            first_b
            last_b

        return:
            |A ∩ B| / |A ∪ B| in [0, 1]; 0 when disjoint or degenerate
    """
    intersection = min(last_a, last_b) - max(first_a, first_b)
    if intersection <= 0:
        return 0.0
    union = max(last_a, last_b) - min(first_a, first_b)
    return intersection / union

def source_found(
    reference: MinimalSource, retrieved: list[MinimalSource]
) -> bool:
    """
        if a retrieved source matches a reference source

        args:
            reference: ground-truth source
            retrieved: sources returned by the retriever
        
        return:
            true if some retrieved source has the identical file_path
            and span IoU strictly above the threshold
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
)-> EvaluationReport:
    """
        score search results against a reference dataset

        args:
            results: saved search results
            reference: dataset whose answeredquestions carry truth
        
        return:
            evaluationreport with mean per-question recall
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
        # skkip if question is unasnwered or there is no context
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
            "reference dataset has no answered questions"
            "with sources - nothing to evaluate"
        )
    return EvaluationReport(
        recall=sum(per_question) / len(per_question),
        questions_evaluated= len(per_question),
        questions_missing=missing,
        sources_found=found_total,
        sources_total=sources_total,
    )
    
def load_results(path: Path) -> StudentSearchResults:
    """
        load and validate a StudentSearchResults json file

        args:
            path
        return:
            parsed results
    """
    if not path.is_file():
        raise FileNotFoundError(f"search results not found: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            return StudentSearchResults.model_validate_json(handle.read())
    except ValueError as exc:
        raise ValueError(
            f"malformed search results json in {path}: {exc}"
        ) from exc

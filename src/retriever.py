"""BM25 retrieval over the inverted index.

Hand-rolled Okapi BM25:

    score(q, d) = sum_t idf(t) * tf * (k1 + 1)
                  / (tf + k1 * (1 - b + b * dl / avgdl))
    idf(t)      = ln((N - df + 0.5) / (df + 0.5) + 1)

k1 controls term-frequency saturation (repeats of a term add less and
less), b controls document-length normalization (0 = none, 1 = full).
Only chunks appearing in at least one query term's postings list are
scored, so a query touches a few thousand postings, never all chunks.
"""

import heapq
import math
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.indexer import Index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.tokenizer import tokenize

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def top_k(
    index: Index,
    query: str,
    k: int,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> list[tuple[int, float]]:
    """Return the k best chunks for a query, best first.

    Args:
        index: The loaded inverted index.
        query: Free-text question; tokenized identically to chunks.
        k: Number of results wanted; k <= 0 yields no results.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalization parameter.

    Returns:
        (chunk_id, score) pairs, score descending; ties break on the
        lower chunk id so results are deterministic. Empty for empty
        or fully out-of-vocabulary queries.
    """
    if k <= 0:
        return []
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []
    scores: defaultdict[int, float] = defaultdict(float)
    n_docs = index.doc_count
    for term in terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        df = len(postings)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        for chunk_id, tf in postings:
            dl = index.chunks[chunk_id][3]
            denom = tf + k1 * (1 - b + b * dl / index.avgdl)
            scores[chunk_id] += idf * tf * (k1 + 1) / denom
    return heapq.nlargest(
        k, scores.items(), key=lambda item: (item[1], -item[0])
    )


def to_source(index: Index, chunk_id: int) -> MinimalSource:
    """Convert an index chunk id into a MinimalSource.

    Args:
        index: The loaded index.
        chunk_id: Position of the chunk in ``index.chunks``.

    Returns:
        A validated MinimalSource with the chunk's verbatim path
        and character span.
    """
    file_path, first, last, _ = index.chunks[chunk_id]
    return MinimalSource(
        file_path=file_path,
        first_character_index=first,
        last_character_index=last,
    )


def load_dataset(path: Path) -> RagDataset:
    """Load and validate a RagDataset JSON file.

    Args:
        path: Dataset file location.

    Returns:
        The parsed dataset (answered and/or unanswered questions).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it is not valid RagDataset JSON.
    """
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            return RagDataset.model_validate_json(handle.read())
    except ValueError as exc:
        raise ValueError(f"malformed dataset JSON in {path}: {exc}") from exc


def search_dataset(
    index: Index,
    dataset: RagDataset,
    k: int,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    show_progress: bool = True,
) -> StudentSearchResults:
    """Retrieve top-k sources for every question in a dataset.

    Args:
        index: The loaded inverted index.
        dataset: Questions to search for.
        k: Number of sources to keep per question.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalization parameter.
        show_progress: Display a tqdm bar over questions.

    Returns:
        StudentSearchResults preserving question ids and order.
    """
    results = []
    questions = tqdm(
        dataset.rag_questions,
        desc="Searching",
        unit="question",
        disable=not show_progress,
    )
    for question in questions:
        ranked = top_k(index, question.question, k, k1, b)
        results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    to_source(index, chunk_id) for chunk_id, _ in ranked
                ],
            )
        )
    return StudentSearchResults(search_results=results, k=k)


def save_results(
    results: StudentSearchResults, save_directory: Path, filename: str
) -> Path:
    """Write search results as JSON into a directory.

    Args:
        results: The results to serialize.
        save_directory: Target directory (created if missing).
        filename: Output file name, conventionally the dataset's.

    Returns:
        Path of the written file.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    target = save_directory / filename
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(results.model_dump_json(indent=2))
    return target

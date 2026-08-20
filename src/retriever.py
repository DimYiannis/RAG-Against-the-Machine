"""
    BM25 retrieval over the corpus index.

    bm25s is fed our pre-tokenized lists, so
    subtokens and path tokens are part of the index
"""

from pathlib import Path

from tqdm import tqdm

from src.indexer import Index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.tokenizer import strip_stopwords, tokenize


def top_k(index: Index, query: str, k: int) -> list[tuple[int, float]]:
    """
        return the k best chunks for a query, best first.

        args:
            index: The loaded index (chunk metadata + bm25s scorer).
            query: Free-text question; tokenized identically to chunks.
            k: Number of results wanted; k <= 0 yields no results.

        return:
            (chunk_id, score) pairs, score descending; ties break on the
            lower chunk id so results are deterministic. Empty for empty
            or fully out-of-vocabulary queries.
    """

    if k <= 0:
        return []
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []
    # Stopwords are dropped from the query but kept in the index: removing
    # them from the index would change every chunk length and every idf,
    # Filtering the query changes only which terms we ask about.
    terms = strip_stopwords(terms)
    # bm25s errors if asked for more documents than it holds.
    wanted = min(k, index.doc_count)
    ids, scores = index.scorer.retrieve(
        [terms], k=wanted, show_progress=False
    )
    ranked = [
        (int(chunk_id), float(score))
        for chunk_id, score in zip(ids[0], scores[0])
        if score > 0
    ]
    # deterministic tie-break: sort by score descending (-item[1]),
    # and when scores are equal, sort by chunk_id ascending
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def to_source(index: Index, chunk_id: int) -> MinimalSource:
    """
        convert an index chunk id to a MinimalSource

        args:
            index
            chunk_id

        return:
            MinimalSource with the chunk's verbatim path
            and char span
    """
    file_path, first, last, _ = index.chunks[chunk_id]
    return MinimalSource(
        file_path=file_path,
        first_character_index=first,
        last_character_index=last,
    )


def load_dataset(path: Path) -> RagDataset:
    """
        load and validate a RagDataset JSON file

        args:
            path

        return:
            parsed dataset - (answered or unanswered questions)
    """
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            return RagDataset.model_validate_json(handle.read())
    except ValueError as exc:
        raise ValueError(f"malformed dataset JSON in {path}: {exc}") from exc


def search_dataset(
    index: Index,
    dataset: RagDataset,
    k: int,
    show_progress: bool = True,
) -> StudentSearchResults:
    """
        retrieve top-k sources for every question in a dataset

        args:
            index: chunk metadata + bm25 scorer
            dataset: questions
            k: number of sources to keep per question
            show_progress: tqdm bar

        return:
            StudentSearchResults
    """
    results = []
    questions = tqdm(
        dataset.rag_questions,
        desc="Searching",
        unit="question",
        disable=not show_progress,
    )
    for question in questions:
        ranked = top_k(index, question.question, k)
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
    """
        output search results as json in a dir

        args:
            results: the results to serialize
            save_directory: target dir
            filename: output file

        return:
            path of the written file
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    target = save_directory / filename
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(results.model_dump_json(indent=2))
    return target

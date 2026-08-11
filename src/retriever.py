"""
    BM25 retrieval over the corpus index.

    bm25s is fed our pre-tokenized lists, so
    subtokens and path tokens are part of the index
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from src.indexer import Index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.tokenizer import tokenize

if TYPE_CHECKING:
    # type-only: keeps sentence-transformers/torch out of the import
    # path for lexical-only (mandatory) runs.
    from sentence_transformers import SentenceTransformer

VALID_MODES = ("lexical", "semantic", "hybrid")
# Cormack et al. standard RRF constant.
RRF_C = 60
# candidates pulled from each retriever before fusion (subject bonus #2:
# "top-N (~100) from each retriever").
FUSION_CANDIDATES = 100


def top_k(
    index: Index,
    query: str,
    k: int,
    mode: str = "lexical",
    embeddings: np.ndarray | None = None,
    model: "SentenceTransformer | None" = None,
) -> list[tuple[int, float]]:
    """
        return the k best chunks for a query, best first.

        args:
            index: The loaded index (chunk metadata + bm25s scorer).
            query: Free-text question; tokenized identically to chunks.
            k: Number of results wanted; k <= 0 yields no results.
            mode:
                "lexical" (bm25, default - mandatory path, untouched),
                "semantic" (cosine similarity over embeddings),
                "hybrid" (RRF fusion of both - needs embeddings+model).
            embeddings: matrix, required for "semantic"
                and "hybrid".
            model: loaded SentenceTransformer, required for "semantic"
                and "hybrid".

        return:
            (chunk_id, score) pairs, score descending; ties break on the
            lower chunk id so results are deterministic. Empty for empty
            or fully out-of-vocabulary queries.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"unknown mode {mode!r} (expected one of {VALID_MODES})"
        )

    if mode == "hybrid":
        if embeddings is None or model is None:
            raise ValueError("hybrid mode requires embeddings + model")
        lexical_ranked = top_k(index, query, FUSION_CANDIDATES)
        semantic_ranked = top_k(
            index, query, FUSION_CANDIDATES,
            mode="semantic", embeddings=embeddings, model=model,
        )
        return fuse(lexical_ranked, semantic_ranked, k)

    if mode == "semantic":
        if embeddings is None or model is None:
            raise ValueError("semantic mode requires embeddings + model")
        from src.embeddings import semantic_top_k
        return semantic_top_k(embeddings, model, query, k)

    if k <= 0:
        return []
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []
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


def fuse(
    lexical_ranked: list[tuple[int, float]],
    semantic_ranked: list[tuple[int, float]],
    k: int,
    c: int = RRF_C,
) -> list[tuple[int, float]]:
    """
        Reciprocal Rank Fusion of a lexical and a semantic ranking.

        RRF(d) = sum over retrievers of 1 / (c + rank_i(d)), with
        1-based ranks. Rank-based, so bm25 and cosine scores never
        need normalizing onto a shared scale.

        args:
            lexical_ranked: (chunk_id, score) pairs, best first.
            semantic_ranked: (chunk_id, score) pairs, best first.
            k: number of fused results to return; k <= 0 yields none.
            c: RRF constant (Cormack et al. standard = 60).

        return:
            (chunk_id, rrf_score) pairs, score descending, top-k; ties
            break on the lower chunk id.
    """
    if k <= 0:
        return []
    rrf_scores: dict[int, float] = {}
    for ranked in (lexical_ranked, semantic_ranked):
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0) + 1.0 / (c + rank)
            )
    fused = sorted(
        rrf_scores.items(), key=lambda item: (-item[1], item[0])
    )
    return fused[:k]


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
    mode: str = "lexical",
    embeddings: np.ndarray | None = None,
    model: "SentenceTransformer | None" = None,
    cache_dir: Path | None = None,
    show_progress: bool = True,
) -> StudentSearchResults:
    """
        retrieve top-k sources for every question in a dataset

        args:
            index: chunk metadata + bm25 scorer
            dataset: questions
            k: number of sources to keep per question
            mode: "lexical" (default), "semantic", or "hybrid"
            embeddings: matrix, required for "semantic"/"hybrid" unless
                cache_dir makes every question a cache hit
            model: loaded SentenceTransformer, same requirement
            cache_dir: when given, checks/stores (query,k,mode) results
                in the on-disk query cache under this dir (bonus #4).
                For semantic/hybrid, embeddings/model are only loaded
                lazily on the first actual cache miss, so an
                all-cache-hit rerun never pays the model-load cost.
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
        ranked = None
        if cache_dir is not None:
            from src import cache
            ranked = cache.get(cache_dir, question.question, k, mode)
        if ranked is None:
            if (
                cache_dir is not None and mode in ("semantic", "hybrid")
                and model is None
            ):
                from src import embeddings as embeddings_module
                embeddings = embeddings_module.load_embeddings(cache_dir)
                model = embeddings_module.load_model()
            ranked = top_k(
                index, question.question, k,
                mode=mode, embeddings=embeddings, model=model,
            )
            if cache_dir is not None:
                from src import cache
                cache.put(cache_dir, question.question, k, mode, ranked)
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

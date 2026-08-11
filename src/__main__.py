"""Command-line entry point for the RAG system.

Exposes the six required commands through Python Fire. This module stays
thin: each method validates nothing itself and delegates to the dedicated
module once the corresponding phase is implemented. A single try/except
at the boundary guarantees the CLI never exits with a raw traceback.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import numpy as np

if TYPE_CHECKING:
    # type-only: keeps sentence-transformers/torch out of the import
    # path for lexical-only (mandatory) runs.
    from sentence_transformers import SentenceTransformer


def _load_semantic(
    mode: str, processed_directory: str
) -> "tuple[np.ndarray | None, SentenceTransformer | None]":
    """Load the embedding matrix + model needed for semantic mode.

    Args:
        mode: Retrieval mode requested by the caller.
        processed_directory: Directory holding index.pkl + embeddings.npy.

    Returns:
        (embeddings, model) pair; both None for "lexical" mode.
    """
    if mode not in ("semantic", "hybrid"):
        return None, None
    from src import embeddings

    matrix = embeddings.load_embeddings(Path(processed_directory))
    model = embeddings.load_model()
    return matrix, model


class RagCLI:
    """RAG over the vLLM 0.10.1 codebase: index, search, answer, evaluate."""

    def index(
        self,
        max_chunk_size: int = 2000,
        data_directory: str = "data/raw",
        save_directory: str = "data/processed",
        mode: str = "lexical",
    ) -> None:
        """Chunk the corpus and build the persisted inverted index.

        Args:
            max_chunk_size: Maximum chunk span in characters (default 2000).
            data_directory: Corpus root to ingest.
            save_directory: Directory the index file is written into.
            mode: "lexical" (default, mandatory path) or "semantic"/
                "hybrid" to also embed every chunk and persist the
                vector matrix alongside the bm25 index (hybrid needs
                the same embeddings semantic search does).
        """
        import time

        from src import indexer

        if mode not in ("lexical", "semantic", "hybrid"):
            raise ValueError(
                f"unknown mode {mode!r} "
                "(expected 'lexical', 'semantic' or 'hybrid')"
            )

        index = indexer.build_index(Path(data_directory), max_chunk_size)
        target = indexer.save_index(index, Path(save_directory))
        print(
            f"Indexed {index.doc_count} chunks "
            f"({len(index.scorer.vocab_dict)} terms, "
            f"avgdl {index.avgdl:.0f}) "
            f"-> {target}"
        )

        if mode in ("semantic", "hybrid"):
            from src import embeddings

            start = time.perf_counter()
            model = embeddings.load_model()
            matrix = embeddings.build_embeddings(index, model=model)
            emb_target = embeddings.save_embeddings(
                matrix, Path(save_directory)
            )
            elapsed = time.perf_counter() - start
            print(
                f"Embedded {matrix.shape[0]} chunks -> {emb_target} "
                f"({elapsed:.1f}s)"
            )

    def search(
        self,
        query: str,
        k: int = 5,
        processed_directory: str = "data/processed",
        mode: str = "lexical",
    ) -> None:
        """Print the top-k results for a single query.

        Args:
            query: Free-text question to search the corpus with.
            k: Number of results to return.
            processed_directory: Directory holding the built index.
            mode: "lexical" (bm25, default), "semantic" (cosine
                similarity over a previously built embedding matrix),
                or "hybrid" (RRF fusion of both).
        """
        from src import cache, indexer, retriever

        index = indexer.load_index(Path(processed_directory))
        proc_dir = Path(processed_directory)
        # cache check first: on a hit, the (possibly heavy) embedding
        # model never gets loaded at all - the real cold-start win.
        ranked = cache.get(proc_dir, str(query), int(k), mode)
        if ranked is None:
            embeddings_matrix, model = _load_semantic(
                mode, processed_directory
            )
            ranked = retriever.top_k(
                index, str(query), int(k),
                mode=mode, embeddings=embeddings_matrix, model=model,
            )
            cache.put(proc_dir, str(query), int(k), mode, ranked)
        if not ranked:
            print("No results.")
            return
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            file_path, first, last, _ = index.chunks[chunk_id]
            print(
                f"{rank}. {file_path} "
                f"[{first}:{last}] score={score:.2f}"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        processed_directory: str = "data/processed",
        mode: str = "lexical",
    ) -> None:
        """Run retrieval for every question in a dataset and save results.

        Args:
            dataset_path: Path to a RagDataset JSON file.
            k: Number of results to keep per question.
            save_directory: Directory the results JSON is written into.
            processed_directory: Directory holding the built index.
            mode: "lexical" (bm25, default), "semantic", or "hybrid".
        """
        from src import indexer, retriever

        index = indexer.load_index(Path(processed_directory))
        dataset = retriever.load_dataset(Path(dataset_path))
        # cache_dir lets search_dataset check the query cache per
        # question and only load the embedding model lazily on the
        # first actual miss - an all-cache-hit rerun skips it entirely.
        results = retriever.search_dataset(
            index, dataset, int(k),
            mode=mode, cache_dir=Path(processed_directory),
        )
        target = retriever.save_results(
            results, Path(save_directory), Path(dataset_path).name
        )
        print(
            f"Searched {len(results.search_results)} questions "
            f"(k={results.k}) -> {target}"
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        processed_directory: str = "data/processed",
        mode: str = "lexical",
    ) -> None:
        """Retrieve top-k sources for a query and generate an answer.

        Args:
            query: Free-text question to answer.
            k: Number of retrieved sources to ground the answer on.
            processed_directory: Directory holding the built index.
            mode: "lexical" (bm25, default), "semantic", or "hybrid".
        """
        from src import cache, generator, indexer, retriever

        index = indexer.load_index(Path(processed_directory))
        proc_dir = Path(processed_directory)
        ranked = cache.get(proc_dir, str(query), int(k), mode)
        if ranked is None:
            embeddings_matrix, embedding_model = _load_semantic(
                mode, processed_directory
            )
            ranked = retriever.top_k(
                index, str(query), int(k), mode=mode,
                embeddings=embeddings_matrix, model=embedding_model,
            )
            cache.put(proc_dir, str(query), int(k), mode, ranked)
        sources = [
            retriever.to_source(index, chunk_id) for chunk_id, _ in ranked
        ]
        tokenizer, model = generator.load_model()
        answer_text = generator.generate_answer(
            tokenizer, model, str(query), sources
        )
        print(answer_text)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for previously saved search results.

        Args:
            student_search_results_path: Path to a StudentSearchResults JSON.
            save_directory: Directory the answered JSON is written into.
        """
        from src import evaluator, generator

        results = evaluator.load_results(Path(student_search_results_path))
        tokenizer, model = generator.load_model()
        answered = generator.answer_results(tokenizer, model, results)
        results_name = Path(student_search_results_path).name
        target = generator.save_answers(
            answered, Path(save_directory), results_name
        )
        print(
            f"Answered {len(answered.search_results)} questions -> {target}"
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Compute recall@k of saved search results against a reference.

        Args:
            student_search_results_path: Path to a StudentSearchResults JSON.
            dataset_path: Path to the reference RagDataset JSON.
        """
        from src import evaluator, retriever

        results = evaluator.load_results(Path(student_search_results_path))
        reference = retriever.load_dataset(Path(dataset_path))
        report = evaluator.evaluate(results, reference)
        print(
            f"recall@{results.k}: {report.recall:.4f} "
            f"({report.sources_found}/{report.sources_total} sources, "
            f"{report.questions_evaluated} questions"
            + (
                f", {report.questions_missing} missing from results"
                if report.questions_missing
                else ""
            )
            + ")"
        )


def main() -> None:
    """Run the Fire CLI, converting any unhandled error into a message."""
    try:
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 — CLI boundary, never traceback
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

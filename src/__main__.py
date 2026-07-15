"""Command-line entry point for the RAG system.

Exposes the six required commands through Python Fire. This module stays
thin: each method validates nothing itself and delegates to the dedicated
module once the corresponding phase is implemented. A single try/except
at the boundary guarantees the CLI never exits with a raw traceback.
"""

import sys
from pathlib import Path

import fire


class RagCLI:
    """RAG over the vLLM 0.10.1 codebase: index, search, answer, evaluate."""

    def index(
        self,
        max_chunk_size: int = 2000,
        data_directory: str = "data/raw",
        save_directory: str = "data/processed",
    ) -> None:
        """Chunk the corpus and build the persisted inverted index.

        Args:
            max_chunk_size: Maximum chunk span in characters (default 2000).
            data_directory: Corpus root to ingest.
            save_directory: Directory the index file is written into.
        """
        from src import indexer

        index = indexer.build_index(Path(data_directory), max_chunk_size)
        target = indexer.save_index(index, Path(save_directory))
        print(
            f"Indexed {index.doc_count} chunks "
            f"({len(index.postings)} terms, avgdl {index.avgdl:.0f}) "
            f"-> {target}"
        )

    def search(
        self,
        query: str,
        k: int = 5,
        processed_directory: str = "data/processed",
    ) -> None:
        """Print the top-k BM25 results for a single query.

        Args:
            query: Free-text question to search the corpus with.
            k: Number of results to return.
            processed_directory: Directory holding the built index.
        """
        from src import indexer, retriever

        index = indexer.load_index(Path(processed_directory))
        ranked = retriever.top_k(index, str(query), int(k))
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
    ) -> None:
        """Run retrieval for every question in a dataset and save results.

        Args:
            dataset_path: Path to a RagDataset JSON file.
            k: Number of results to keep per question.
            save_directory: Directory the results JSON is written into.
            processed_directory: Directory holding the built index.
        """
        from src import indexer, retriever

        index = indexer.load_index(Path(processed_directory))
        dataset = retriever.load_dataset(Path(dataset_path))
        results = retriever.search_dataset(index, dataset, int(k))
        target = retriever.save_results(
            results, Path(save_directory), Path(dataset_path).name
        )
        print(
            f"Searched {len(results.search_results)} questions "
            f"(k={results.k}) -> {target}"
        )

    def answer(self, query: str, k: int = 5) -> None:
        """Retrieve top-k sources for a query and generate an answer.

        Args:
            query: Free-text question to answer.
            k: Number of retrieved sources to ground the answer on.
        """
        print(f"[stub] answer: query={query!r} k={k}")

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
        print(
            "[stub] answer_dataset: student_search_results_path="
            f"{student_search_results_path!r} "
            f"save_directory={save_directory!r}"
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

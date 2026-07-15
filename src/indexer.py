"""Inverted-index construction and persistence.

The index maps each term to a postings list ``[(chunk_id, tf), ...]``
so query time is O(query terms x matching postings) instead of
O(all chunks). Chunk text is deliberately NOT stored: offsets are the
ground truth, and any consumer (display, generation) re-slices the
corpus file, guaranteeing byte-identical spans and a small index file.
"""

import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.chunking import chunk_text, iter_corpus_files, read_text
from src.tokenizer import tokenize

#: (file_path, first_char, last_char, token_count) per chunk.
ChunkMeta = tuple[str, int, int, int]

_INDEX_FILENAME = "index.pkl"
_FORMAT_VERSION = 1


@dataclass
class Index:
    """A searchable BM25 index over the chunked corpus.

    Attributes:
        max_chunk_size: Chunk size cap the corpus was chunked with.
        chunks: Per-chunk metadata; position = chunk id.
        postings: term -> list of (chunk_id, term frequency).
        avgdl: Average chunk length in tokens (BM25 length norm).
    """

    max_chunk_size: int
    chunks: list[ChunkMeta]
    postings: dict[str, list[tuple[int, int]]]
    avgdl: float

    @property
    def doc_count(self) -> int:
        """Number of chunks in the index."""
        return len(self.chunks)


def build_index(
    data_directory: Path,
    max_chunk_size: int = 2000,
    show_progress: bool = True,
) -> Index:
    """Chunk and tokenize the corpus into an inverted index.

    Args:
        data_directory: Corpus root to ingest (e.g. ``data/raw``).
        max_chunk_size: Hard cap on chunk span in characters.
        show_progress: Display a tqdm bar over files.

    Returns:
        The in-memory index.

    Raises:
        FileNotFoundError: If the corpus root does not exist.
        ValueError: If it contains no indexable files or
            max_chunk_size is not positive.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"corpus directory not found: {data_directory}"
        )
    files = iter_corpus_files(data_directory)
    if not files:
        raise ValueError(f"no indexable files under {data_directory}")

    chunks: list[ChunkMeta] = []
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    total_tokens = 0

    iterator = tqdm(files, desc="Indexing", unit="file",
                    disable=not show_progress)
    for path in iterator:
        text = read_text(path)
        if text is None:
            continue
        rel_path = Path(os.path.relpath(path)).as_posix()
        # Path tokens ("vllm", "model_executor", "gpu_model_runner",
        # "py", ...) ride along with every chunk of the file: many
        # questions name a file or module (verbatim or paraphrased)
        # rather than quoting its content, and the path would
        # otherwise never appear in any chunk's text. Measured in
        # Phase 5: code recall@5 0.667 -> 0.758 with this alone.
        path_tokens = tokenize(rel_path.removeprefix(f"{data_directory}/"))
        for chunk in chunk_text(text, rel_path, max_chunk_size):
            tokens = tokenize(chunk.text) + path_tokens
            if not tokens:
                continue
            chunk_id = len(chunks)
            chunks.append(
                (chunk.file_path, chunk.first, chunk.last, len(tokens))
            )
            total_tokens += len(tokens)
            for term, freq in Counter(tokens).items():
                postings[term].append((chunk_id, freq))

    if not chunks:
        raise ValueError("corpus produced no indexable chunks")
    return Index(
        max_chunk_size=max_chunk_size,
        chunks=chunks,
        postings=dict(postings),
        avgdl=total_tokens / len(chunks),
    )


def save_index(index: Index, save_directory: Path) -> Path:
    """Persist an index under a directory as a pickle.

    Args:
        index: The index to save.
        save_directory: Target directory (created if missing).

    Returns:
        Path of the written index file.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    target = save_directory / _INDEX_FILENAME
    payload = {
        "version": _FORMAT_VERSION,
        "max_chunk_size": index.max_chunk_size,
        "chunks": index.chunks,
        "postings": index.postings,
        "avgdl": index.avgdl,
    }
    with open(target, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_index(processed_directory: Path) -> Index:
    """Load a previously saved index.

    Args:
        processed_directory: Directory the index was saved into.

    Returns:
        The loaded index.

    Raises:
        FileNotFoundError: If no index file exists there yet.
        ValueError: If the file is corrupt or from another version.
    """
    target = processed_directory / _INDEX_FILENAME
    if not target.is_file():
        raise FileNotFoundError(
            f"no index at {target} — run "
            "`uv run python -m src index` first"
        )
    try:
        with open(target, "rb") as handle:
            payload = pickle.load(handle)
        if payload["version"] != _FORMAT_VERSION:
            raise ValueError(
                f"index format {payload['version']} unsupported"
            )
        return Index(
            max_chunk_size=payload["max_chunk_size"],
            chunks=payload["chunks"],
            postings=payload["postings"],
            avgdl=payload["avgdl"],
        )
    except (pickle.UnpicklingError, KeyError, EOFError) as exc:
        raise ValueError(f"corrupt index file {target}: {exc}") from exc

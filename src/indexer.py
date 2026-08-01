"""
    inverted-index construction and persistence.

    - the index maps each term to a postings list [(chunk_id, tf),..]
        so query time is O(query terms x matching postings) instead of
        O(all chunks).
    - chunk text is deliberately NOT stored:
    - consumer (display, generation): offsets + re-slice the
        corpus file -> identical spans and a small index file.
"""
import os
# object serialization
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# progress-bar library
from tqdm import tqdm

from src.chunking import chunk_text, iter_corpus_files, read_text
from src.tokenizer import tokenize

ChunkMeta = tuple[str, int, int, int]

INDEX_FILENAME = "index.pkl"
FORMAT_VERSION = 1

@dataclass
class Index:
    """
        BM25 index over the chunked corpus

        attrs:
            max_chunk_size
            chunks
            postings: terms -> list of chunk_id, tf
            avgdl: average chunk length in tokens (bm25 length norm)
    """

    max_chunk_size: int
    chunks: list[ChunkMeta]
    postings: dict[str, list[tuple[int, int]]]
    avgdl: float

    @property
    def doc_count(self) -> int:
        """
            number of chunks in the index
        """
        return len(self.chunks)


def build_index(
    data_dir: Path,
    max_chunk_size: int = 2000,
    show_progress: bool = True,
) -> Index:
    """
        chunk and tokenize the corpus to inverted index

        args:
            data_dir: corpus root
            max_chunk_size
            show_progress: display tqdm bar

        return:
            the in_memory index
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {data_dir}")
    files = iter_corpus_files(data_dir)
    if not files:
        raise ValueError(f"no indexable files under {data_dir}")

    chunks: list[ChunkMeta] = []
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    total_tokens = 0

    iterator = tqdm(files, desc="indexing", unit="files", disable=not show_progress)
    for path in iterator:
        text = read_text(path)
        if text is None:
            continue
        rel_path = Path(os.path.relpath(path)).as_posix()
        # path tokens let a chunk match questions that name its file
        # (e.g. gpu_model_runner.py) without quoting any code content.
        path_tokens = tokenize(rel_path.removeprefix(f"{data_dir}/"))
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
        raise ValueError("corpus producced no chunks")
    return Index(
        max_chunk_size=max_chunk_size,
        chunks=chunks,
        postings=dict(postings),
        avgdl=total_tokens / len(chunks),
    )


def save_index(index: Index, save_dir: Path) -> Path:
    """
        save chunk metadata as a dict using pickle to avoid save chunks
        based on these offsets consumers reslice the corpus

        args:
            index
            save_dir: target dir
        return:
            path of the written index file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / INDEX_FILENAME
    payload = {
        "version": FORMAT_VERSION,
        "max_chunk_size": index.max_chunk_size,
        "chunks": index.chunks,
        "postings": index.postings,
        "avgdl": index.avgdl,
    }
    with open(target, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_index(processed_dir: Path) -> Index:
    """
        load a previously saved index

        args:
            processed_dir

        return:
            the loaded index
    """
    target = processed_dir / INDEX_FILENAME
    if not target.is_file():
        raise FileNotFoundError(
            f"no index at {target} "
            "- run 'uv run python -m src index' first"
        )
    try:
        with open(target, "rb") as handle:
            payload = pickle.load(handle)
        if payload["version"] != FORMAT_VERSION:
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

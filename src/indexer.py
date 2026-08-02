"""
    inverted-index construction and persistence.

    - after chunking and tokenization we feed bm25s our pre-tokenized
        lists so path tokens and subtokens survive into the index.
    - chunk text is deliberately NOT stored:
    - consumer (display, generation): offsets + re-slice the
        corpus file -> identical spans and a small index file.
"""
import os
# object serialization
import pickle
from dataclasses import dataclass
from pathlib import Path

import bm25s
# progress-bar library
from tqdm import tqdm

from src.chunking import chunk_text, iter_corpus_files, read_text
from src.tokenizer import tokenize

ChunkMeta = tuple[str, int, int, int]

INDEX_FILENAME = "index.pkl"
SCORER_DIRNAME = "bm25s"
# bumped when the on-disk layout changes; v1 was the hand-rolled
# postings dict, v2 is metadata + a bm25s scorer directory.
FORMAT_VERSION = 2

# BM25 params, tuned in Phase 5. Unlike the hand-rolled scorer these are
# baked in at *index* time by bm25s, so changing them needs a rebuild.
DEFAULT_K1 = 1.3
DEFAULT_B = 0.85


@dataclass
class Index:
    """
        BM25 index over the chunked corpus

        attrs:
            max_chunk_size
            chunks: per-chunk metadata; position = chunk id, and the
                same position bm25s returns from a query
            scorer: the bm25s retriever, built from our token lists
            avgdl: average chunk length in tokens (reported stat;
                bm25s does its own length normalization internally)
    """

    max_chunk_size: int
    chunks: list[ChunkMeta]
    scorer: bm25s.BM25
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
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> Index:
    """
        chunk and tokenize the corpus, then hand the token lists to bm25s

        args:
            data_dir: corpus root
            max_chunk_size
            show_progress: display tqdm bar
            k1: bm25 term-frequency saturation (index-time under bm25s)
            b: bm25 length normalization (index-time under bm25s)

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
    corpus_tokens: list[list[str]] = []
    total_tokens = 0

    iterator = tqdm(files, desc="indexing", unit="files",
                    disable=not show_progress)
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
            chunks.append(
                (chunk.file_path, chunk.first, chunk.last, len(tokens))
            )
            corpus_tokens.append(tokens)
            total_tokens += len(tokens)

    if not chunks:
        raise ValueError("corpus producced no chunks")

    # bm25s takes the token lists directly, so our tokenizer (subtokens
    # + path tokens) stays the thing that decides what is matchable.
    scorer = bm25s.BM25(k1=k1, b=b, method="lucene")
    scorer.index(corpus_tokens, show_progress=show_progress)

    return Index(
        max_chunk_size=max_chunk_size,
        chunks=chunks,
        scorer=scorer,
        avgdl=total_tokens / len(chunks),
    )


def save_index(index: Index, save_dir: Path) -> Path:
    """
        save chunk metadata as a dict using pickle to avoid save chunks
        based on these offsets consumers reslice the corpus.
        the bm25s scorer is written next to it in its own format.

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
        "avgdl": index.avgdl,
    }
    with open(target, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    index.scorer.save(str(save_dir / SCORER_DIRNAME), show_progress=False)
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
                " - rebuild with 'uv run python -m src index'"
            )
        scorer = bm25s.BM25.load(
            str(processed_dir / SCORER_DIRNAME), show_progress=False
        )
        return Index(
            max_chunk_size=payload["max_chunk_size"],
            chunks=payload["chunks"],
            scorer=scorer,
            avgdl=payload["avgdl"],
        )
    except (pickle.UnpicklingError, KeyError, EOFError, OSError) as exc:
        raise ValueError(f"corrupt index file {target}: {exc}") from exc

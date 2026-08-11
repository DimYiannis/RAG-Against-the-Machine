"""
    semantic search with dense retrieval
    relevance is based on cosine similarity amongst the vectors generated
    from the embedding models
"""


from pathlib import Path
from typing import cast

from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

from src.indexer import Index

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILENAME = "embeddings.npy"


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
        load the sentence-transformers bi-encoder

        args:
            model_name

        return:
            loaded SentenceTransformer
    """
    model = SentenceTransformer(model_name, device="cpu")
    return cast(SentenceTransformer, model)


def _get_chunks(index: Index, show_progress: bool = True) -> list[str]:
    """
        reslice and build chunk_id and spans

        args:
            index: chunk metadata
            show_progress: tqdm bar

        return:
            chunk texts, position= chunk_id
    """
    cache: dict[str, str] = {}
    texts: list[str] = []
    iterator = tqdm(
        index.chunks, desc="reslicing", unit="chunk",
        disable=not show_progress
    )
    for file_path, first, last, _ in iterator:
        if file_path not in cache:
            with open(file_path, encoding="utf-8") as handle:
                cache[file_path] = handle.read()
        texts.append(cache[file_path][first:last])
    return texts


def build_embeddings(
    index: Index,
    model: SentenceTransformer | None = None,
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
        embed every chunk in the index

        args:
            index: chunk metadata
            model
            batch_size
            show_progress

        return:
            l2-normalized matrix
    """
    if model is None:
        model = load_model()
    texts = _get_chunks(index, show_progress=show_progress)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def save_embeddings(matrix: np.ndarray, save_dir: Path) -> Path:
    """
        save embedding matrix next to the lexical index

        args:
            matrix
            save_dir: same dir the bm25 index is

        return:
            path of written .npy file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / EMBEDDINGS_FILENAME
    np.save(target, matrix)
    return target


def load_embeddings(save_dir: Path) -> np.ndarray:
    """
        load previously saved embedding matrix

        args:
            save_dir
        return:
            matrix
    """
    target = save_dir / EMBEDDINGS_FILENAME
    if not target.is_file():
        raise FileNotFoundError(
            f"no embeddings at {target} "
            "- run 'uv run python -m src index --mode semantic' first"
        )
    try:
        return cast(np.ndarray, np.load(target))
    except (OSError, ValueError) as exc:
        raise ValueError(f"corrupt embeddings file {target}: {exc}") from exc


def semantic_top_k(
    embeddings: np.ndarray,
    model: SentenceTransformer,
    query: str,
    k: int,
) -> list[tuple[int, float]]:
    """
        return the k best chunks for a query by cosine similarity

        args:
            embeddings
            model
            query: text question
            k: number of results wanted

            return:
                (chunk_id, score) pairs, tie break on lower chunk id
    """
    if k <= 0 or embeddings.shape[0] == 0:
        return []
    query_vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False
    )[0].astype(np.float32)
    sims = embeddings @ query_vec
    wanted = min(k, sims.shape[0])
    top_idx = np.argpartition(-sims, wanted - 1)[:wanted]
    ranked = [(int(i), float(sims[i])) for i in top_idx]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked

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




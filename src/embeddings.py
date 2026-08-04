"""
    semantic search with dense retrieval
    relevance is based on cosine similarity amongst the vectors generated
    from the embedding models
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
from src.tokenizer import tokenize

def get_chunks()
    """
        reslice and build chunk_id and spans
    """

def create_vectors()
    """
        through chunk_id i get the text feed it to the 
        embedding model to produce the vectors
    """

"""
    not sure if i will do cosine similarity 
    by myself or i can call it from the model too
"""
"""Local retrieval-augmented research support layer."""

from src.support.rag import (
    LocalSupportRAG,
    RetrievalHit,
    SupportAnswer,
    SupportChunk,
)

__all__ = [
    "LocalSupportRAG",
    "RetrievalHit",
    "SupportAnswer",
    "SupportChunk",
]

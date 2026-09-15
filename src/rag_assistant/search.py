from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rag_assistant.chunking import Chunk
from rag_assistant.embeddings import Embedder


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class SemanticSearch:
    """In-memory semantic search suitable for learning and small experiments."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("at least one chunk is required")

        self._chunks = list(chunks)
        self._embedder = embedder
        self._vectors = embedder.encode([chunk.text for chunk in chunks])
        if self._vectors.shape[0] != len(chunks):
            raise ValueError("embedder returned the wrong number of vectors")

    def search(self, query: str, *, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self._embedder.encode([query])[0]
        # Embeddings are normalized, so their dot product equals cosine similarity.
        scores = self._vectors @ query_vector
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(chunk=self._chunks[index], score=float(scores[index]))
            for index in ranked_indices
        ]


from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    """Anything capable of mapping text to one vector per input string."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]: ...


class SentenceTransformerEmbedder:
    """Local semantic embeddings using a small pretrained transformer."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the project dependencies before running the real demo."
            ) from exc

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float64)


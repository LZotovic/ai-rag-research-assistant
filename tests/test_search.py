import unittest
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rag_assistant.chunking import Chunk
from rag_assistant.search import SemanticSearch


class ControlledEmbedder:
    """Predictable vectors make this a unit test of ranking, not an ML model."""

    vectors = {
        "passage about citations": [1.0, 0.0],
        "passage about cooking": [0.0, 1.0],
        "how do sources support an answer?": [0.9, 0.1],
    }

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float64)


class SemanticSearchTests(unittest.TestCase):
    def test_returns_most_similar_chunk_first(self) -> None:
        chunks = [
            Chunk("passage about citations", "research.pdf", 0),
            Chunk("passage about cooking", "recipes.pdf", 0),
        ]
        engine = SemanticSearch(chunks, ControlledEmbedder())

        results = engine.search("how do sources support an answer?", top_k=2)

        self.assertEqual(results[0].chunk.source, "research.pdf")
        self.assertGreater(results[0].score, results[1].score)


if __name__ == "__main__":
    unittest.main()

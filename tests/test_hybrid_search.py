import numpy as np

from rag_assistant.index import DocumentIndex


class EqualEmbedder:
    """Return equal semantic vectors so exact terms decide the test ranking."""

    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts])


def test_exact_technical_term_breaks_semantic_tie():
    index = DocumentIndex(EqualEmbedder())
    index.add_document(
        "metrics.pdf",
        [
            "The model was evaluated with several general measurements.",
            "The classifier achieved macro-F1 of 0.8055 on the test split.",
        ],
    )

    result = index.search("What was the macro-F1?", top_k=1)[0]

    assert "0.8055" in result.chunk.text
    assert result.chunk.page == 2

import numpy as np

from rag_assistant.index import DocumentIndex


class KeywordEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray([
                float("citation" in lowered or "source" in lowered),
                float("pasta" in lowered),
            ])
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else np.asarray([0.1, 0.1]))
        return np.asarray(vectors)


def test_add_search_and_delete_document():
    index = DocumentIndex(KeywordEmbedder())
    document = index.add_document(
        "paper.pdf",
        ["A citation identifies the source of a claim.", "Pasta uses salted water."],
    )

    results = index.search("Which source supports the citation?", top_k=1)

    assert document.pages == 2
    assert results[0].chunk.page == 1
    assert results[0].chunk.source == "paper.pdf"
    assert index.delete_document(document.id)
    assert index.list_documents() == []


def test_sqlite_restores_index_after_restart(tmp_path):
    database = tmp_path / "citewise.db"
    first_index = DocumentIndex(KeywordEmbedder(), database)
    document = first_index.add_document(
        "persistent.pdf",
        ["A citation identifies the source of a claim."],
    )

    # A new object simulates stopping and restarting the web server.
    restored_index = DocumentIndex(KeywordEmbedder(), database)
    results = restored_index.search("Which citation gives the source?", top_k=1)

    assert restored_index.list_documents() == [document]
    assert results[0].chunk.source == "persistent.pdf"
    assert results[0].chunk.page == 1

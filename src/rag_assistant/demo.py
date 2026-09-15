from rag_assistant.chunking import chunk_words
from rag_assistant.embeddings import SentenceTransformerEmbedder
from rag_assistant.search import SemanticSearch


SAMPLE_DOCUMENT = """
Retrieval-augmented generation combines information retrieval with text
generation. Documents are divided into passages and converted into embeddings.
At question time, the most relevant passages are supplied to a language model.

Model fine-tuning changes a model's parameters by training it on additional
examples. It is useful for changing behavior or teaching a consistent task, but
it is usually not the simplest way to provide frequently changing documents.

Grounded assistants should cite the evidence used for important claims. They
should also abstain when retrieved passages do not contain enough information
to answer a question reliably.
"""


def main() -> None:
    chunks = chunk_words(
        SAMPLE_DOCUMENT,
        source="rag_notes.txt",
        chunk_size=35,
        overlap=8,
    )
    search_engine = SemanticSearch(chunks, SentenceTransformerEmbedder())
    query = "How can an assistant show that its answer is supported?"

    print(f"Question: {query}\n")
    for rank, result in enumerate(search_engine.search(query, top_k=2), start=1):
        print(f"{rank}. score={result.score:.3f} source={result.chunk.source}")
        print(f"   {result.chunk.text}\n")


if __name__ == "__main__":
    main()


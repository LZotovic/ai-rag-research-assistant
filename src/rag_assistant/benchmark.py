from collections.abc import Callable

from rag_assistant.chunking import Chunk, chunk_paragraphs, chunk_words
from rag_assistant.demo import SAMPLE_DOCUMENT
from rag_assistant.embeddings import SentenceTransformerEmbedder
from rag_assistant.evaluation import EvaluationCase, evaluate_retrieval
from rag_assistant.search import SemanticSearch


CASES = [
    EvaluationCase(
        "What two components does RAG combine?",
        "information retrieval with text generation",
    ),
    EvaluationCase(
        "What is sent to the language model at question time?",
        "most relevant passages",
    ),
    EvaluationCase(
        "How does fine-tuning modify a model?",
        "changes a model's parameters",
    ),
    EvaluationCase(
        "Is fine-tuning ideal for frequently changing documents?",
        "not the simplest way",
    ),
    EvaluationCase(
        "How should claims be supported?",
        "cite the evidence",
    ),
    EvaluationCase(
        "What should the assistant do without enough information?",
        "should also abstain",
    ),
]


def fixed_chunks() -> list[Chunk]:
    return chunk_words(
        SAMPLE_DOCUMENT,
        source="rag_notes.txt",
        chunk_size=35,
        overlap=8,
    )


def paragraph_chunks() -> list[Chunk]:
    return chunk_paragraphs(SAMPLE_DOCUMENT, source="rag_notes.txt")


def main() -> None:
    embedder = SentenceTransformerEmbedder()
    strategies: dict[str, Callable[[], list[Chunk]]] = {
        "fixed word windows": fixed_chunks,
        "paragraph-aware": paragraph_chunks,
    }

    print(f"Evaluation questions: {len(CASES)}\n")
    for name, create_chunks in strategies.items():
        chunks = create_chunks()
        metrics = evaluate_retrieval(SemanticSearch(chunks, embedder), CASES)
        print(
            f"{name:20} chunks={len(chunks):2} "
            f"Recall@1={metrics.recall_at_1:.0%} "
            f"Recall@3={metrics.recall_at_3:.0%}"
        )


if __name__ == "__main__":
    main()

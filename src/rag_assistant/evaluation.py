from dataclasses import dataclass
from typing import Sequence

from rag_assistant.search import SemanticSearch


@dataclass(frozen=True)
class EvaluationCase:
    """A question and text that must occur in its relevant source chunk."""

    question: str
    expected_text: str


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float


def evaluate_retrieval(
    engine: SemanticSearch,
    cases: Sequence[EvaluationCase],
) -> RetrievalMetrics:
    """Measure how often a relevant chunk occurs within the first k results."""
    if not cases:
        raise ValueError("at least one evaluation case is required")

    hits_at_1 = 0
    hits_at_3 = 0
    for case in cases:
        results = engine.search(case.question, top_k=3)
        relevant = [case.expected_text.lower() in result.chunk.text.lower() for result in results]
        hits_at_1 += int(bool(relevant) and relevant[0])
        hits_at_3 += int(any(relevant))

    count = len(cases)
    return RetrievalMetrics(
        recall_at_1=hits_at_1 / count,
        recall_at_3=hits_at_3 / count,
    )

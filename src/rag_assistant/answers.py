from dataclasses import dataclass
from typing import Sequence

from rag_assistant.search import SearchResult


@dataclass(frozen=True)
class Citation:
    number: int
    source: str
    page: int | None
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    grounded: bool
    mode: str = "extractive"


class ExtractiveAnswerGenerator:
    """Free, deterministic fallback that returns retrieved evidence verbatim."""

    def generate(self, question: str, results: Sequence[SearchResult]) -> Answer:
        # A conservative threshold is preferable to presenting unrelated text
        # as evidence. This value should later be tuned on a larger benchmark.
        useful = [result for result in results if result.score >= 0.2]
        if not useful:
            return Answer(
                text="I could not find enough evidence in the uploaded documents.",
                citations=[],
                grounded=False,
            )
        citations = [
            Citation(
                number=number,
                source=result.chunk.source,
                page=result.chunk.page,
                text=result.chunk.text,
                score=result.score,
            )
            for number, result in enumerate(useful, start=1)
        ]
        return Answer(
            text=(
                "I found the following relevant evidence in your documents. "
                "Review the cited passages below."
            ),
            citations=citations,
            grounded=True,
        )

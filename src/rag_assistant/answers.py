from dataclasses import dataclass
import json
from typing import Protocol
from typing import Sequence

import httpx

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


class AnswerGenerator(Protocol):
    def generate(self, question: str, results: Sequence[SearchResult]) -> Answer: ...


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


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server or selected model is unavailable."""


class OllamaAnswerGenerator:
    """Create grounded answers with a local open-source model through Ollama."""

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "integer"}},
            "grounded": {"type": "boolean"},
        },
        "required": ["answer", "citations", "grounded"],
    }

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        base_url: str = "http://localhost:11434",
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=180)
        self._fallback = ExtractiveAnswerGenerator()

    def generate(self, question: str, results: Sequence[SearchResult]) -> Answer:
        evidence = self._fallback.generate(question, results)
        if not evidence.grounded:
            return evidence

        context = "\n\n".join(
            f"[{item.number}] {item.source}, page {item.page}\n{item.text}"
            for item in evidence.citations
        )
        prompt = (
            f"Question:\n{question}\n\nRetrieved evidence:\n{context}\n\n"
            "Return a concise answer based only on this evidence. The citations "
            "array must contain the evidence numbers supporting the answer."
        )
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an evidence-grounded research assistant. "
                                "Never use outside knowledge or invent facts. If the "
                                "evidence is insufficient, set grounded to false."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "format": self.RESPONSE_SCHEMA,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 350},
                },
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            payload = json.loads(content)
        except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OllamaUnavailableError(
                "The local Ollama model could not answer. Start Ollama and run "
                f"'ollama pull {self.model}'."
            ) from exc

        valid_numbers = {citation.number for citation in evidence.citations}
        raw_citations = payload.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []
        requested_numbers = {
            number
            for number in raw_citations
            if isinstance(number, int)
        }
        used_numbers = requested_numbers & valid_numbers

        # Never return a generated claim unless it points to retrieved evidence.
        answer_text = str(payload.get("answer", "")).strip()
        if not payload.get("grounded") or not answer_text or not used_numbers:
            return Answer(
                text="I could not find enough evidence in the uploaded documents.",
                citations=[],
                grounded=False,
            )

        citations = [
            citation
            for citation in evidence.citations
            if citation.number in used_numbers
        ]
        return Answer(
            text=answer_text,
            citations=citations,
            grounded=True,
            mode="ollama",
        )

    def is_available(self) -> bool:
        """Check whether the configured model is installed in local Ollama."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            names = {model.get("name") for model in models}
            return self.model in names or f"{self.model}:latest" in names
        except (httpx.HTTPError, KeyError, TypeError):
            return False

import json

import httpx
import pytest

from rag_assistant.answers import OllamaAnswerGenerator, OllamaUnavailableError
from rag_assistant.chunking import Chunk
from rag_assistant.search import SearchResult


def search_result(text, page, score=0.8):
    return SearchResult(Chunk(text, "paper.pdf", page - 1, page=page), score)


def client_returning(payload):
    def handler(request):
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(payload)}},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_answer_preserves_used_citation_numbers():
    generator = OllamaAnswerGenerator(
        client=client_returning(
            {
                "answer": "The reported accuracy was 96%.",
                "citations": [2],
                "grounded": True,
            }
        )
    )
    results = [
        search_result("The model uses a CNN.", 2),
        search_result("The model reached 96% accuracy.", 7),
    ]

    answer = generator.generate("What accuracy was reported?", results)

    assert answer.mode == "ollama"
    assert [citation.number for citation in answer.citations] == [2]
    assert [citation.page for citation in answer.citations] == [7]


def test_ollama_rejects_generated_answer_without_valid_citation():
    generator = OllamaAnswerGenerator(
        client=client_returning(
            {
                "answer": "An unsupported answer.",
                "citations": [99],
                "grounded": True,
            }
        )
    )

    answer = generator.generate(
        "What happened?",
        [search_result("Relevant evidence.", 1)],
    )

    assert not answer.grounded
    assert answer.citations == []


def test_ollama_connection_error_has_actionable_message():
    def handler(request):
        raise httpx.ConnectError("not running", request=request)

    generator = OllamaAnswerGenerator(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(OllamaUnavailableError, match="ollama pull"):
        generator.generate(
            "What happened?",
            [search_result("Relevant evidence.", 1)],
        )

from rag_assistant.answers import ExtractiveAnswerGenerator
from rag_assistant.chunking import Chunk
from rag_assistant.search import SearchResult


def test_answer_contains_traceable_citation():
    result = SearchResult(
        Chunk("Evidence text.", "paper.pdf", 0, page=4),
        0.8,
    )

    answer = ExtractiveAnswerGenerator().generate("Question?", [result])

    assert answer.grounded
    assert answer.citations[0].page == 4
    assert answer.citations[0].source == "paper.pdf"


def test_answer_abstains_for_weak_results():
    result = SearchResult(Chunk("Unrelated.", "paper.pdf", 0), 0.05)
    answer = ExtractiveAnswerGenerator().generate("Question?", [result])

    assert not answer.grounded
    assert answer.citations == []

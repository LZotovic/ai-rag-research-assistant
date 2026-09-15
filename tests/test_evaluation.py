import unittest

from rag_assistant.chunking import Chunk
from rag_assistant.evaluation import EvaluationCase, evaluate_retrieval
from rag_assistant.search import SearchResult


class StubSearch:
    def search(self, query: str, *, top_k: int = 3) -> list[SearchResult]:
        results = {
            "easy": ["the target evidence", "other", "unrelated"],
            "hard": ["other", "the second target", "unrelated"],
        }
        return [
            SearchResult(Chunk(text, "notes.txt", index), 1.0 - index / 10)
            for index, text in enumerate(results[query][:top_k])
        ]


class EvaluationTests(unittest.TestCase):
    def test_computes_recall_at_different_cutoffs(self) -> None:
        metrics = evaluate_retrieval(
            StubSearch(),  # type: ignore[arg-type]
            [
                EvaluationCase("easy", "target evidence"),
                EvaluationCase("hard", "second target"),
            ],
        )

        self.assertEqual(metrics.recall_at_1, 0.5)
        self.assertEqual(metrics.recall_at_3, 1.0)


if __name__ == "__main__":
    unittest.main()

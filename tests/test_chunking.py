import unittest

from rag_assistant.chunking import chunk_paragraphs, chunk_words


class ChunkWordsTests(unittest.TestCase):
    def test_chunks_overlap_and_preserve_metadata(self) -> None:
        chunks = chunk_words(
            "one two three four five six",
            source="notes.txt",
            chunk_size=4,
            overlap=2,
        )

        self.assertEqual([chunk.text for chunk in chunks], [
            "one two three four",
            "three four five six",
        ])
        self.assertEqual(chunks[1].source, "notes.txt")
        self.assertEqual(chunks[1].index, 1)

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            chunk_words("some text", source="x", chunk_size=3, overlap=3)

    def test_paragraph_chunking_preserves_author_boundaries(self) -> None:
        chunks = chunk_paragraphs(
            "First paragraph.\nStill first.\n\nSecond paragraph.",
            source="paper.pdf",
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["First paragraph. Still first.", "Second paragraph."],
        )
        self.assertEqual(chunks[1].index, 1)


if __name__ == "__main__":
    unittest.main()

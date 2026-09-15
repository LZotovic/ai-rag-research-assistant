from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from threading import RLock
from uuid import uuid4

import numpy as np

from rag_assistant.chunking import Chunk, chunk_pages
from rag_assistant.embeddings import Embedder
from rag_assistant.search import SearchResult


@dataclass(frozen=True)
class Document:
    id: str
    filename: str
    pages: int
    chunks: int
    created_at: str


class DocumentIndex:
    """Small vector index with optional SQLite persistence.

    NumPy performs the similarity search in memory because this keeps the code
    understandable and is fast enough for a student portfolio demo. SQLite
    preserves documents and embeddings between restarts. At larger scale this
    class is the seam where pgvector or a managed vector database would fit.
    """

    def __init__(self, embedder: Embedder, database_path: str | Path | None = None):
        self._embedder = embedder
        self._documents: dict[str, Document] = {}
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0), dtype=np.float64)
        self._lock = RLock()
        self._database: sqlite3.Connection | None = None

        if database_path is not None:
            path = Path(database_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._database = sqlite3.connect(path, check_same_thread=False)
            self._create_schema()
            self._load_from_database()

    def _create_schema(self) -> None:
        """Create the tiny local persistence schema on first launch."""
        assert self._database is not None
        self._database.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                pages INTEGER NOT NULL,
                chunks INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER,
                vector BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                PRIMARY KEY (document_id, chunk_index),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
            """
        )
        self._database.commit()

    def _load_from_database(self) -> None:
        """Restore metadata, chunks, and embeddings without re-embedding PDFs."""
        assert self._database is not None
        for row in self._database.execute(
            "SELECT id, filename, pages, chunks, created_at FROM documents"
        ):
            document = Document(*row)
            self._documents[document.id] = document

        vectors: list[np.ndarray] = []
        query = (
            "SELECT document_id, chunk_index, text, source, page, vector, dimensions "
            "FROM chunks ORDER BY rowid"
        )
        for document_id, index, text, source, page, blob, dimensions in self._database.execute(query):
            self._chunks.append(
                Chunk(text, source, index, page=page, document_id=document_id)
            )
            vectors.append(np.frombuffer(blob, dtype=np.float64, count=dimensions))
        if vectors:
            self._vectors = np.vstack(vectors)

    def add_document(self, filename: str, pages: list[str]) -> Document:
        document_id = str(uuid4())
        chunks = chunk_pages(
            pages,
            source=filename,
            document_id=document_id,
        )
        if not chunks:
            raise ValueError("The PDF did not produce any searchable chunks.")
        vectors = self._embedder.encode([chunk.text for chunk in chunks])
        document = Document(
            id=document_id,
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._documents[document_id] = document
            self._chunks.extend(chunks)
            self._vectors = vectors if not self._vectors.size else np.vstack((self._vectors, vectors))
            if self._database is not None:
                # One transaction ensures metadata and vectors cannot get out of sync.
                self._database.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    (
                        document.id,
                        document.filename,
                        document.pages,
                        document.chunks,
                        document.created_at,
                    ),
                )
                self._database.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            document_id,
                            chunk.index,
                            chunk.text,
                            chunk.source,
                            chunk.page,
                            np.asarray(vector, dtype=np.float64).tobytes(),
                            len(vector),
                        )
                        for chunk, vector in zip(chunks, vectors)
                    ],
                )
                self._database.commit()
        return document

    def list_documents(self) -> list[Document]:
        with self._lock:
            return list(self._documents.values())

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._documents:
                return False
            keep = [
                index
                for index, chunk in enumerate(self._chunks)
                if chunk.document_id != document_id
            ]
            self._chunks = [self._chunks[index] for index in keep]
            if keep:
                self._vectors = self._vectors[keep]
            else:
                self._vectors = np.empty((0, 0), dtype=np.float64)
            if self._database is not None:
                self._database.execute(
                    "DELETE FROM chunks WHERE document_id = ?", (document_id,)
                )
                self._database.execute(
                    "DELETE FROM documents WHERE id = ?", (document_id,)
                )
                self._database.commit()
            del self._documents[document_id]
            return True

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Question cannot be empty.")
        with self._lock:
            if not self._chunks:
                return []
            query_vector = self._embedder.encode([query])[0]
            # Embeddings are normalized, so dot product equals cosine similarity.
            semantic_scores = self._vectors @ query_vector

            # Semantic models can underweight exact identifiers such as
            # "macro-F1", model names, or measurements. A small lexical bonus
            # makes those precise terms matter without taking over the ranking.
            query_terms = self._important_terms(query)
            lexical_scores = np.asarray(
                [
                    len(query_terms & self._important_terms(chunk.text))
                    / max(len(query_terms), 1)
                    for chunk in self._chunks
                ],
                dtype=np.float64,
            )
            scores = 0.85 * semantic_scores + 0.15 * lexical_scores
            indices = np.argsort(scores)[::-1][:top_k]
            return [
                SearchResult(self._chunks[index], float(scores[index]))
                for index in indices
            ]

    @staticmethod
    def _important_terms(text: str) -> set[str]:
        """Normalize words used by the lightweight keyword part of retrieval."""
        stop_words = {
            "a", "an", "and", "did", "do", "does", "for", "in", "is",
            "of", "on", "the", "to", "was", "what", "which", "with",
        }
        return {
            term
            for term in re.findall(r"[a-z0-9]+", text.lower())
            if term not in stop_words and len(term) > 1
        }

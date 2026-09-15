from dataclasses import asdict
from functools import lru_cache
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_assistant.answers import ExtractiveAnswerGenerator
from rag_assistant.embeddings import SentenceTransformerEmbedder
from rag_assistant.index import DocumentIndex
from rag_assistant.pdf import PdfExtractionError, extract_pdf_pages


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


@lru_cache
def get_index() -> DocumentIndex:
    # A local SQLite file keeps uploads available after restarting the server.
    database_path = os.getenv("CITEWISE_DB_PATH", "data/citewise.db")
    return DocumentIndex(SentenceTransformerEmbedder(), database_path)


def create_app(index: DocumentIndex | None = None) -> FastAPI:
    app = FastAPI(
        title="CiteWise API",
        description="Semantic research assistant for cited PDF question answering.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # The current checkpoint is intentionally retrieval-only. A local,
    # open-source generator will be added in the next commit.
    answer_generator = ExtractiveAnswerGenerator()

    def active_index() -> DocumentIndex:
        return index if index is not None else get_index()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/documents")
    async def documents() -> list[dict]:
        return [asdict(document) for document in active_index().list_documents()]

    @app.post("/api/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...)) -> dict:
        filename = Path(file.filename or "document.pdf").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(415, "Only PDF files are supported.")
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(413, "PDF files must be smaller than 25 MB.")
        try:
            pages = extract_pdf_pages(content)
            document = active_index().add_document(filename, pages)
        except (PdfExtractionError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return asdict(document)

    @app.delete("/api/documents/{document_id}", status_code=204)
    async def delete_document(document_id: str) -> None:
        if not active_index().delete_document(document_id):
            raise HTTPException(404, "Document not found.")

    @app.post("/api/questions")
    async def ask_question(request: QuestionRequest) -> dict:
        results = active_index().search(request.question, request.top_k)
        if not results:
            raise HTTPException(409, "Upload at least one PDF before asking a question.")
        answer = answer_generator.generate(request.question, results)
        return {
            "answer": answer.text,
            "grounded": answer.grounded,
            "mode": answer.mode,
            "citations": [asdict(citation) for citation in answer.citations],
        }

    frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend.exists():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            candidate = frontend / path
            return FileResponse(candidate if candidate.is_file() else frontend / "index.html")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("rag_assistant.api:app", host="0.0.0.0", port=8000, reload=True)

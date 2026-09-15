import pytest

from rag_assistant.pdf import PdfExtractionError, extract_pdf_pages


def test_rejects_non_pdf_content():
    with pytest.raises(PdfExtractionError, match="not a readable PDF"):
        extract_pdf_pages(b"this is not a PDF")

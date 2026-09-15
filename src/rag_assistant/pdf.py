from io import BytesIO

from pypdf import PdfReader


class PdfExtractionError(ValueError):
    pass


def extract_pdf_pages(content: bytes) -> list[str]:
    """Extract text from a PDF byte stream, retaining one item per page."""
    if not content:
        raise PdfExtractionError("The uploaded PDF is empty.")
    try:
        # BytesIO makes uploaded bytes behave like a normal file for pypdf.
        reader = PdfReader(BytesIO(content))
        # Keep empty pages in the list so later indexes still match PDF pages.
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise PdfExtractionError("The uploaded file is not a readable PDF.") from exc

    if not pages or not any(pages):
        raise PdfExtractionError(
            "No text was found. Scanned PDFs require OCR, which is not enabled yet."
        )
    return pages

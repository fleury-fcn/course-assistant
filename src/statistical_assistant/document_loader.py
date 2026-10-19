from hashlib import sha256
from pathlib import Path

import pymupdf

from .models import DocumentDiagnostic, Page
from .text_cleaner import clean_text


class DocumentLoadError(RuntimeError):
    """Erreur explicite de lecture d'un document PDF."""


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    return sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())


def document_id(pdf_path: Path) -> str:
    return sha256(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:16]


def load_pdf(pdf_path: Path, use_ocr: bool = False) -> tuple[list[Page], DocumentDiagnostic]:
    pages: list[Page] = []
    try:
        with pymupdf.open(pdf_path) as document:
            for number, pdf_page in enumerate(document, start=1):
                raw = pdf_page.get_text("text")
                if not raw.strip() and use_ocr:
                    raw = _ocr_page(pdf_page)
                cleaned = clean_text(raw)
                pages.append(Page(document_id(pdf_path), pdf_path.name, number, cleaned, raw))
    except (pymupdf.FileDataError, OSError) as error:
        raise DocumentLoadError(f"Impossible de lire « {pdf_path.name} » : {error}") from error

    text_pages = sum(bool(page.text) for page in pages)
    warnings: list[str] = []
    if not text_pages:
        warnings.append("Aucune page ne contient de texte exploitable ; le PDF semble scanné.")
    elif sum(len(page.text) for page in pages) < 100:
        warnings.append("Le texte extrait est très court ; vérifiez le contenu du PDF.")
    return pages, DocumentDiagnostic(
        str(pdf_path), len(pages), text_pages, sum(len(page.text) for page in pages), tuple(warnings)
    )


def _ocr_page(pdf_page: pymupdf.Page) -> str:
    """Utilise pytesseract uniquement s'il est installé par l'utilisateur."""
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        image = pdf_page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        return pytesseract.image_to_string(pil_image(image))
    except (OSError, RuntimeError, ValueError):
        return ""


def pil_image(pixmap: pymupdf.Pixmap):
    from PIL import Image
    from io import BytesIO

    return Image.open(BytesIO(pixmap.tobytes("png")))

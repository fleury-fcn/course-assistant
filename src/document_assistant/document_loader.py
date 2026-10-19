from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

import pymupdf

from .models import DocumentDiagnostic, Page
from .text_cleaner import clean_text


class DocumentLoadError(RuntimeError):
    """Erreur explicite de lecture d'un document."""


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml", ".html",
    ".htm", ".yaml", ".yml", ".py", ".js", ".ts", ".css", ".log",
}
PYMUPDF_EXTENSIONS = {".pdf", ".xps", ".epub", ".mobi", ".cbz", ".fb2", ".svg", ".xhtml"}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PYMUPDF_EXTENSIONS | OFFICE_EXTENSIONS


def discover_documents(document_dir: Path) -> list[Path]:
    return sorted(
        path for path in document_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
        and path.name.lower() != "readme.md"
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """Compatibilité avec l'ancienne API."""
    return discover_documents(pdf_dir)


def document_id(document_path: Path) -> str:
    return sha256(str(document_path.resolve()).encode("utf-8")).hexdigest()[:16]


def load_document(document_path: Path, use_ocr: bool = False) -> tuple[list[Page], DocumentDiagnostic]:
    suffix = document_path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        pages = [_page(document_path, 1, _read_text(document_path))]
    elif suffix in OFFICE_EXTENSIONS:
        pages = [_page(document_path, 1, _read_office_xml(document_path))]
    else:
        pages = _load_pymupdf(document_path, use_ocr)
    return _diagnostic(document_path, pages)


def load_pdf(pdf_path: Path, use_ocr: bool = False) -> tuple[list[Page], DocumentDiagnostic]:
    """Compatibilité avec l'ancienne API."""
    return load_document(pdf_path, use_ocr)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _read_office_xml(path: Path) -> str:
    try:
        with ZipFile(path) as archive:
            values = []
            for name in archive.namelist():
                if name.endswith(".xml"):
                    xml = archive.read(name).decode("utf-8", errors="ignore")
                    value = re.sub(r"<[^>]+>", " ", xml)
                    value = re.sub(r"\s+", " ", value).strip()
                    if value:
                        values.append(value)
            return "\n".join(values)
    except (BadZipFile, OSError) as error:
        raise DocumentLoadError(f"Impossible de lire « {path.name} » : {error}") from error


def _load_pymupdf(path: Path, use_ocr: bool) -> list[Page]:
    pages: list[Page] = []
    try:
        with pymupdf.open(path) as document:
            for number, pdf_page in enumerate(document, start=1):
                raw = pdf_page.get_text("text")
                if not raw.strip() and use_ocr:
                    raw = _ocr_page(pdf_page)
                pages.append(_page(path, number, raw))
    except (pymupdf.FileDataError, OSError) as error:
        raise DocumentLoadError(f"Impossible de lire « {path.name} » : {error}") from error
    return pages


def _page(path: Path, number: int, raw: str) -> Page:
    return Page(document_id(path), path.name, number, clean_text(raw), raw)


def _diagnostic(path: Path, pages: list[Page]) -> tuple[list[Page], DocumentDiagnostic]:
    text_pages = sum(bool(page.text) for page in pages)
    warnings: list[str] = []
    if not text_pages:
        warnings.append("Aucun texte exploitable ; activez l'OCR si le document est une image.")
    elif sum(len(page.text) for page in pages) < 100:
        warnings.append("Le texte extrait est très court ; vérifiez le contenu du document.")
    return pages, DocumentDiagnostic(
        str(path), len(pages), text_pages, sum(len(page.text) for page in pages), tuple(warnings)
    )


def _ocr_page(pdf_page: pymupdf.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        pixmap = pdf_page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        return pytesseract.image_to_string(Image.open(BytesIO(pixmap.tobytes("png"))))
    except (OSError, RuntimeError, ValueError):
        return ""

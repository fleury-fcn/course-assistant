from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    document_id: str
    document_name: str
    page_number: int
    text: str
    raw_text: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    lexical_score: float = 0.0
    character_score: float = 0.0


@dataclass(frozen=True)
class DocumentDiagnostic:
    path: str
    page_count: int
    text_pages: int
    character_count: int
    warnings: tuple[str, ...] = ()

from .models import Chunk, Page


def chunk_pages(pages: list[Page], size: int = 800, overlap: int = 120) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size doit être positif et overlap compris entre 0 et size - 1.")
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        start = 0
        part = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                if boundary > start + size // 2:
                    end = boundary
            value = text[start:end].strip()
            if value:
                chunks.append(Chunk(
                    f"{page.document_id}-{page.page_number}-{part}",
                    page.document_id, page.document_name, page.page_number, value, start, end
                ))
                part += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from .document_loader import load_document


def summarize_document(
    path: Path, sentence_count: int = 5, use_ocr: bool = False
) -> tuple[str, tuple[str, ...]]:
    """Create a local extractive summary and return it with loader warnings."""
    if sentence_count <= 0:
        raise ValueError("sentence_count doit être positif.")

    pages, diagnostic = load_document(path, use_ocr=use_ocr)
    text = "\n".join(page.text for page in pages if page.text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    if not sentences:
        return "", diagnostic.warnings
    if len(sentences) <= sentence_count:
        return " ".join(sentences), diagnostic.warnings

    try:
        matrix = TfidfVectorizer(
            strip_accents="unicode", lowercase=True, stop_words=None
        ).fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:sentence_count]), diagnostic.warnings

    scores = matrix.sum(axis=1).A1
    selected = sorted(
        sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)[:sentence_count]
    )
    return " ".join(sentences[index] for index in selected), diagnostic.warnings

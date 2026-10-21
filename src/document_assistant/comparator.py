from dataclasses import dataclass
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .document_loader import load_document


@dataclass(frozen=True)
class DocumentComparison:
    common_sentences: tuple[str, ...]
    first_unique_sentences: tuple[str, ...]
    second_unique_sentences: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def compare_documents(
    first_path: Path,
    second_path: Path,
    passage_count: int = 5,
    use_ocr: bool = False,
) -> DocumentComparison:
    if first_path == second_path:
        raise ValueError("Sélectionnez deux documents différents.")
    if passage_count <= 0:
        raise ValueError("passage_count doit être positif.")

    first_sentences, first_warnings = _sentences(first_path, use_ocr)
    second_sentences, second_warnings = _sentences(second_path, use_ocr)
    if not first_sentences or not second_sentences:
        return DocumentComparison(
            (),
            tuple(first_sentences[:passage_count]),
            tuple(second_sentences[:passage_count]),
            first_warnings + second_warnings,
        )

    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True)
    matrix = vectorizer.fit_transform(first_sentences + second_sentences)
    similarities = cosine_similarity(
        matrix[: len(first_sentences)], matrix[len(first_sentences) :]
    )

    common_pairs = [
        (float(similarities[first_index, second_index]), first_index, second_index)
        for first_index in range(len(first_sentences))
        for second_index in range(len(second_sentences))
        if similarities[first_index, second_index] >= 0.25
    ]
    common_pairs.sort(reverse=True)
    common = tuple(
        f"{first_sentences[first_index]} ↔ {second_sentences[second_index]}"
        for _, first_index, second_index in common_pairs[:passage_count]
    )
    first_unique = _unique_sentences(first_sentences, similarities.max(axis=1), passage_count)
    second_unique = _unique_sentences(second_sentences, similarities.max(axis=0), passage_count)
    return DocumentComparison(
        common,
        first_unique,
        second_unique,
        first_warnings + second_warnings,
    )


def _sentences(path: Path, use_ocr: bool) -> tuple[list[str], tuple[str, ...]]:
    pages, diagnostic = load_document(path, use_ocr=use_ocr)
    text = "\n".join(page.text for page in pages if page.text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    return sentences, diagnostic.warnings


def _unique_sentences(
    sentences: list[str], similarities, count: int
) -> tuple[str, ...]:
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: float(similarities[index]),
    )
    return tuple(sentences[index] for index in ranked[:count])

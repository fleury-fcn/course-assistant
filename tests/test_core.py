from pathlib import Path

from src.statistical_assistant.chunker import chunk_pages
from src.statistical_assistant.document_loader import load_pdf
from src.statistical_assistant.indexing import TfidfIndex
from src.statistical_assistant.models import Page
from src.statistical_assistant.retrieval import search_course
from src.statistical_assistant.text_cleaner import clean_text


def test_cleaner_preserves_math_symbols_and_removes_word_hyphenation():
    assert clean_text("test-\nstatistique  − 2") == "teststatistique − 2"


def test_chunking_preserves_physical_page_number():
    chunks = chunk_pages([Page("doc", "cours.pdf", 7, "estimateur et paramètre")], 100, 10)
    assert chunks[0].page_number == 7


def test_index_search_returns_only_relevant_results():
    chunks = chunk_pages([
        Page("a", "cours.pdf", 1, "La moyenne empirique est un estimateur."),
        Page("a", "cours.pdf", 2, "La variance mesure la dispersion."),
    ], 100, 10)
    results = search_course(TfidfIndex.build(chunks), "estimateur moyenne", 5, 0.01)
    assert results and results[0].chunk.page_number == 1
    assert all(result.score > 0 for result in results)


def test_hybrid_search_handles_a_partial_word():
    chunks = chunk_pages([Page("a", "cours.pdf", 3, "L'estimation ponctuelle d'un paramètre.")], 100, 10)
    results = search_course(TfidfIndex.build(chunks), "estim param", 1, 0.01)
    assert results and results[0].chunk.page_number == 3


def test_pdf_has_pages():
    pages, diagnostic = load_pdf(Path("data/raw/cours_statistiques.pdf"))
    assert pages and diagnostic.page_count == len(pages)

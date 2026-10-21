from pathlib import Path

from src.document_assistant.chunker import chunk_pages
from src.document_assistant.comparator import compare_documents
from src.document_assistant.document_loader import discover_documents, load_document
from src.document_assistant.indexing import TfidfIndex
from src.document_assistant.indexing import index_is_current
from src.document_assistant.models import Page
from src.document_assistant.retrieval import search_course
from src.document_assistant.summarizer import summarize_document
from src.document_assistant.text_cleaner import clean_text


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


def test_index_tracks_source_changes(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("La moyenne décrit le centre.", encoding="utf-8")
    index = TfidfIndex.build(
        chunk_pages([Page("a", source.name, 1, source.read_text())], 100, 10)
    )
    index.save(tmp_path / "index", [source])

    assert index_is_current(tmp_path / "index", [source]) is True
    source.write_text("La variance décrit la dispersion.", encoding="utf-8")
    assert index_is_current(tmp_path / "index", [source]) is False


def test_hybrid_search_handles_a_partial_word():
    chunks = chunk_pages([Page("a", "cours.pdf", 3, "L'estimation ponctuelle d'un paramètre.")], 100, 10)
    results = search_course(TfidfIndex.build(chunks), "estim param", 1, 0.01)
    assert results and results[0].chunk.page_number == 3


def test_pdf_has_pages(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "sample.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Une page de statistiques.")
        document.save(pdf_path)

    pages, diagnostic = load_document(pdf_path)
    assert pages and diagnostic.page_count == len(pages)


def test_text_document_is_discovered_and_loaded(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Une note\n\nUn contenu à explorer.", encoding="utf-8")
    assert discover_documents(tmp_path) == [source]
    pages, diagnostic = load_document(source)
    assert diagnostic.text_pages == 1
    assert "contenu" in pages[0].text


def test_document_diagnostic_reports_empty_text_warning(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")

    _, diagnostic = load_document(source)

    assert diagnostic.page_count == 1
    assert diagnostic.text_pages == 0
    assert diagnostic.warnings


def test_document_summary_keeps_source_order(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(
        "La moyenne décrit le centre. La variance décrit la dispersion. "
        "Un estimateur doit être étudié avec ses propriétés.",
        encoding="utf-8",
    )

    summary, warnings = summarize_document(source, sentence_count=2)

    assert summary.startswith("La moyenne décrit le centre.")
    assert summary.count(".") == 2
    assert summary.index(".") < summary.rindex(".")
    assert not warnings


def test_document_comparison_finds_common_and_unique_passages(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "La moyenne décrit le centre. La médiane résiste aux valeurs extrêmes.",
        encoding="utf-8",
    )
    second.write_text(
        "La moyenne décrit le centre. La variance décrit la dispersion.",
        encoding="utf-8",
    )

    comparison = compare_documents(first, second, passage_count=2)

    assert comparison.common_sentences
    assert "La médiane" in comparison.first_unique_sentences[0]
    assert "La variance" in comparison.second_unique_sentences[0]

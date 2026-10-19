from .models import SearchResult


def format_citation(result: SearchResult) -> str:
    return f"{result.chunk.document_name}, page {result.chunk.page_number}"


def citations_for(results: list[SearchResult]) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for result in results:
        citation = format_citation(result)
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations

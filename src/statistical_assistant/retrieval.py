from sklearn.metrics.pairwise import cosine_similarity

from .indexing import TfidfIndex
from .models import SearchResult


def search_course(
    index: TfidfIndex, question: str, number_of_results: int = 5, threshold: float = 0.05
) -> list[SearchResult]:
    if not question.strip():
        return []
    query = index.vectorizer.transform([question])
    character_query = (
        index.character_vectorizer.transform([question])
        if index.character_vectorizer is not None
        else None
    )
    if not query.nnz and (character_query is None or not character_query.nnz):
        return []
    import numpy as np

    lexical_scores = (
        cosine_similarity(query, index.matrix).ravel()
        if query.nnz
        else np.zeros(len(index.chunks))
    )
    character_scores = (
        cosine_similarity(character_query, index.character_matrix).ravel()
        if character_query is not None and character_query.nnz
        else np.zeros(len(index.chunks))
    )
    scores = 0.7 * lexical_scores + 0.3 * character_scores
    ranked = scores.argsort()[::-1]
    return [
        SearchResult(
            index.chunks[i],
            float(scores[i]),
            float(lexical_scores[i]),
            float(character_scores[i]),
        )
        for i in ranked[:number_of_results]
        if scores[i] >= threshold
    ]

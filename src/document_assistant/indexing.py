import json
from pathlib import Path
import pickle
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer

from .models import Chunk
from .storage import file_fingerprint


class TfidfIndex:
    """Index hybride local : mots pour les termes exacts, caractères pour les variantes."""

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix,
        chunks: list[Chunk],
        character_vectorizer: TfidfVectorizer | None = None,
        character_matrix=None,
    ):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks
        self.character_vectorizer = character_vectorizer
        self.character_matrix = character_matrix

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "TfidfIndex":
        if not chunks:
            raise ValueError("Impossible de construire un index sans passage.")
        vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        except ValueError as error:
            raise ValueError("Le vocabulaire du document est insuffisant pour TF-IDF.") from error
        character_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )
        character_matrix = character_vectorizer.fit_transform([chunk.text for chunk in chunks])
        return cls(vectorizer, matrix, chunks, character_vectorizer, character_matrix)

    def save(self, directory: Path, source_paths: list[Path] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "tfidf.pkl").open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sources = [
            {"name": path.name, "fingerprint": file_fingerprint(path)}
            for path in source_paths or []
        ]
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "chunks": len(self.chunks),
                    "documents": sorted({c.document_name for c in self.chunks}),
                    "sources": sources,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "TfidfIndex":
        with (directory / "tfidf.pkl").open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise ValueError("Format d'index inattendu.")
        return loaded


def index_metadata(directory: Path) -> dict:
    path = directory / "metadata.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Les métadonnées de l'index doivent contenir un objet.")
    return data


def index_is_current(directory: Path, source_paths: list[Path]) -> bool | None:
    """Return None when the index predates source tracking."""
    metadata = index_metadata(directory)
    sources = metadata.get("sources")
    if not isinstance(sources, list):
        return None
    expected = {
        path.name: file_fingerprint(path)
        for path in source_paths
    }
    actual = {
        item.get("name"): item.get("fingerprint")
        for item in sources
        if isinstance(item, dict)
    }
    return actual == expected

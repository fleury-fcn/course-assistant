from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.document_assistant.chunker import chunk_pages
from src.document_assistant.config import Settings
from src.document_assistant.document_loader import DocumentLoadError, discover_documents, load_document
from src.document_assistant.indexing import TfidfIndex


def main() -> None:
    settings = Settings.from_project_root(Path(__file__).resolve().parents[1])
    chunks = []
    source_paths = discover_documents(settings.pdf_dir)
    for path in source_paths:
        try:
            pages, diagnostic = load_document(path)
        except DocumentLoadError as error:
            print(error, file=sys.stderr)
            raise SystemExit(1) from error
        if diagnostic.warnings:
            print(f"{path.name}: {'; '.join(diagnostic.warnings)}")
        chunks.extend(chunk_pages(pages, settings.chunk_size, settings.chunk_overlap))
    if not chunks:
        print("Aucun passage textuel à indexer.", file=sys.stderr)
        raise SystemExit(1)
    index = TfidfIndex.build(chunks)
    index.save(settings.index_dir, source_paths)
    print(f"Index créé : {len(index.chunks)} passages dans {settings.index_dir}")


if __name__ == "__main__":
    main()

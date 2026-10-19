from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistical_assistant.chunker import chunk_pages
from src.statistical_assistant.config import Settings
from src.statistical_assistant.document_loader import DocumentLoadError, discover_pdfs, load_pdf
from src.statistical_assistant.indexing import TfidfIndex


def main() -> None:
    settings = Settings.from_project_root(Path(__file__).resolve().parents[1])
    chunks = []
    for path in discover_pdfs(settings.pdf_dir):
        try:
            pages, diagnostic = load_pdf(path)
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
    index.save(settings.index_dir)
    print(f"Index créé : {len(index.chunks)} passages dans {settings.index_dir}")


if __name__ == "__main__":
    main()

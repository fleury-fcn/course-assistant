from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.document_assistant.config import Settings
from src.document_assistant.document_loader import DocumentLoadError, discover_documents, load_document


def main() -> None:
    settings = Settings.from_project_root(Path(__file__).resolve().parents[1])
    documents = discover_documents(settings.pdf_dir)
    if not documents:
        print("Aucun document trouvé.")
        return
    for path in documents:
        try:
            pages, diagnostic = load_document(path)
        except DocumentLoadError as error:
            print(error)
            continue
        print(f"{path.name}: {diagnostic.page_count} pages, {diagnostic.text_pages} avec texte")
        print(f"  Caractères : {diagnostic.character_count}")
        for warning in diagnostic.warnings:
            print(f"  Avertissement : {warning}")
        for page in pages[:2]:
            print(f"  Page {page.page_number}: {page.text[:200] or '[vide]'}")


if __name__ == "__main__":
    main()

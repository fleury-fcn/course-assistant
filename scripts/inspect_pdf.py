from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistical_assistant.config import Settings
from src.statistical_assistant.document_loader import DocumentLoadError, discover_pdfs, load_pdf


def main() -> None:
    settings = Settings.from_project_root(Path(__file__).resolve().parents[1])
    pdfs = discover_pdfs(settings.pdf_dir)
    if not pdfs:
        print("Aucun PDF trouvé.")
        return
    for path in pdfs:
        try:
            pages, diagnostic = load_pdf(path)
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

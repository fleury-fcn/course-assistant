from pathlib import Path
import sys

from src.document_assistant.config import Settings
from src.document_assistant.document_loader import discover_documents


def main() -> None:
    """Afficher un diagnostic rapide des documents disponibles."""
    settings = Settings.from_project_root(Path(__file__).resolve().parent)
    documents = discover_documents(settings.pdf_dir)
    if not documents:
        print(f"Aucun document trouvé dans {settings.pdf_dir}")
        sys.exit(0)
    for document in documents:
        print(document.name)


if __name__ == "__main__":
    main()
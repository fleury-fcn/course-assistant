from pathlib import Path
import sys

from src.statistical_assistant.config import Settings
from src.statistical_assistant.document_loader import discover_pdfs


def main() -> None:
    """Afficher un diagnostic rapide des documents disponibles."""
    settings = Settings.from_project_root(Path(__file__).resolve().parent)
    pdf_files = discover_pdfs(settings.pdf_dir)
    if not pdf_files:
        print(f"Aucun PDF trouvé dans {settings.pdf_dir}")
        sys.exit(0)
    for pdf_file in pdf_files:
        print(pdf_file.name)


if __name__ == "__main__":
    main()
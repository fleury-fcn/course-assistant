import re


def clean_text(text: str) -> str:
    """Nettoyage prudent : espaces et césures de mots uniquement."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", normalized)
    return re.sub(r"[ \t]+", " ", normalized).strip()

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    pdf_dir: Path
    index_dir: Path
    chunk_size: int = 800
    chunk_overlap: int = 120
    default_results: int = 5
    relevance_threshold: float = 0.05
    max_upload_mb: int = 50
    use_ocr: bool = False

    @classmethod
    def from_project_root(cls, root: Path) -> "Settings":
        return cls(
            project_root=root,
            pdf_dir=root / "data" / "raw",
            index_dir=root / "data" / "index",
        )

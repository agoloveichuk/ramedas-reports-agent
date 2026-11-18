from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Юридичні особи групи компаній, які НЕ є навчальними закладами і мають ігноруватися як facility_hint
IGNORED_LEGAL_ENTITIES: tuple[str, ...] = (
    "рамедас",
    "рамедас україна",
    "тов рамедас",
    "фондако",
    "мистецтво смаку",
    "школярик",
    "лерчія",
    "січових стрільців",
    "склад луцьк",
)


def _default_viber_downloads() -> Path:
    docs_onedrive = Path.home() / "OneDrive" / "Documents" / "ViberDownloads"
    if docs_onedrive.exists():
        return docs_onedrive
    return Path.home() / "Documents" / "ViberDownloads"


def _default_viber_pc_dir() -> Path:
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    viber_root = appdata / "ViberPC"
    if viber_root.exists():
        for child in viber_root.iterdir():
            if child.is_dir() and child.name.isdigit():
                return child
    return viber_root


@dataclass(slots=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest").strip()
    )
    viber_downloads_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("VIBER_DOWNLOADS_DIR", str(_default_viber_downloads()))
        )
    )
    viber_pc_dir: Path = field(
        default_factory=lambda: Path(os.getenv("VIBER_PC_DIR", str(_default_viber_pc_dir())))
    )
    facilities_json_path: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / os.getenv("FACILITIES_JSON_PATH", "./data/facilities.json")
    )
    storage_root: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("STORAGE_ROOT", "./storage")
    )
    batch_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("BATCH_WINDOW_SECONDS", "45"))
    )
    silent_mode: bool = field(
        default_factory=lambda: os.getenv("SILENT_MODE", "true").lower() in ("1", "true", "yes")
    )
    confidence_review_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_REVIEW_THRESHOLD", "0.78"))
    )
    ignored_entities: tuple[str, ...] = IGNORED_LEGAL_ENTITIES

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Юридичні особи групи компаній, які НЕ є навчальними закладами і мають ігноруватися як facility_hint
IGNORED_LEGAL_ENTITIES: tuple[str, ...] = (
    "рамедас",
    "рамадас",
    "рамедас україна",
    "тов рамедас",
    "дженерал девелопмент",
    "дженерал девелопмент україна",
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


def _env_str(key: str, default: str = "") -> str:
    val = os.getenv(key)
    return val.strip() if val and val.strip() else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    return int(raw) if raw else default


def _env_path(key: str, default: str) -> Path:
    raw = os.getenv(key, "").strip() or default
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_optional_path(key: str) -> Optional[Path]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(slots=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: _env_str("GEMINI_API_KEY"))
    gemini_model: str = field(
        default_factory=lambda: _env_str("GEMINI_MODEL", "models/gemini-flash-lite-latest")
    )
    viber_downloads_dir: Path = field(
        default_factory=lambda: Path(
            _env_str("VIBER_DOWNLOADS_DIR") or str(_default_viber_downloads())
        )
    )
    viber_pc_dir: Path = field(
        default_factory=lambda: Path(_env_str("VIBER_PC_DIR") or str(_default_viber_pc_dir()))
    )
    viber_retention_days: int = field(
        default_factory=lambda: _env_int("VIBER_RETENTION_DAYS", 21)
    )
    viber_archive_dir: Optional[Path] = field(
        default_factory=lambda: _env_optional_path("VIBER_ARCHIVE_DIR")
    )
    facilities_json_path: Path = field(
        default_factory=lambda: _env_path("FACILITIES_JSON_PATH", "./data/facilities.json")
    )
    storage_root: Path = field(
        default_factory=lambda: _env_path("STORAGE_ROOT", "./storage")
    )
    reports_output_dir: Optional[Path] = field(
        default_factory=lambda: _env_optional_path("REPORTS_OUTPUT_DIR")
    )
    document_date_max_age_days: int = field(
        default_factory=lambda: _env_int("DOCUMENT_DATE_MAX_AGE_DAYS", 14)
    )
    batch_window_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_WINDOW_SECONDS", 45)
    )
    silent_mode: bool = field(
        default_factory=lambda: os.getenv("SILENT_MODE", "true").lower() in ("1", "true", "yes")
    )
    confidence_review_threshold: float = field(
        default_factory=lambda: float(_env_str("CONFIDENCE_REVIEW_THRESHOLD", "0.78"))
    )
    telegram_bot_token: str = field(default_factory=lambda: _env_str("TELEGRAM_BOT_TOKEN"))
    tech_alert_chat_id: str = field(default_factory=lambda: _env_str("TECH_ALERT_CHAT_ID"))
    telegram_api_id: int = field(default_factory=lambda: _env_int("TELEGRAM_API_ID", 0))
    telegram_api_hash: str = field(default_factory=lambda: _env_str("TELEGRAM_API_HASH"))
    telegram_session_path: Path = field(
        default_factory=lambda: _env_path("TELEGRAM_SESSION_PATH", "./storage/telegram_user.session")
    )
    telegram_downloads_dir: Path = field(
        default_factory=lambda: _env_path("TELEGRAM_DOWNLOADS_DIR", "./storage/telegram_downloads")
    )
    telegram_lookback_hours: int = field(
        default_factory=lambda: _env_int("TELEGRAM_LOOKBACK_HOURS", 72)
    )
    telegram_allowed_chats: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            part.strip()
            for part in os.getenv("TELEGRAM_ALLOWED_CHATS", "").split(",")
            if part.strip()
        )
    )
    work_hours_start: str = field(default_factory=lambda: _env_str("WORK_HOURS_START", "09:00"))
    work_hours_end: str = field(default_factory=lambda: _env_str("WORK_HOURS_END", "18:00"))
    work_days: str = field(default_factory=lambda: _env_str("WORK_DAYS", "1,2,3,4,5"))
    silence_alert_minutes: int = field(
        default_factory=lambda: _env_int("SILENCE_ALERT_MINUTES", 120)
    )
    alert_cooldown_minutes: int = field(
        default_factory=lambda: _env_int("ALERT_COOLDOWN_MINUTES", 60)
    )
    daily_summary_time: str = field(
        default_factory=lambda: _env_str("DAILY_SUMMARY_TIME", "18:30")
    )
    ignored_entities: tuple[str, ...] = IGNORED_LEGAL_ENTITIES

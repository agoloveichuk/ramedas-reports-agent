from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Optional

from src.connectors.viber_local import compute_file_sha256
from src.logger import get_logger
from src.models import FacilityMatch, IncomingMediaItem, VisionAnalysisResult

log = get_logger("storage")

COMPLETED_STATUSES = (
    "matched",
    "needs_review",
    "unmatched",
    "ignored_non_report",
    "duplicate",
)
MAX_RETRY_COUNT = 5

IGNORED_SURNAMES = frozenset(
    {
        "трачук",
        "павлючик",
        "павлюк",
        "павлович",
        "карпович",
        "оззо",
        "ззсо",
        "тов",
        "фоп",
        "рамедас",
        "директор",
        "кухар",
        # Поширені власні імена (щоб із "Світлана Ковальчук" чи "Анатолій Лашик" бралося саме прізвище)
        "олена",
        "тетяна",
        "наталія",
        "наташа",
        "світлана",
        "людмила",
        "галина",
        "валентина",
        "марія",
        "ніна",
        "ольга",
        "ірина",
        "оксана",
        "анна",
        "ганна",
        "катерина",
        "надія",
        "вікторія",
        "юлія",
        "інна",
        "алла",
        "василина",
        "раїса",
        "тамара",
        "рима",
        "лілія",
        "леся",
        "зоя",
        "віра",
        "любов",
        "марина",
        "аліна",
        "володимир",
        "сергій",
        "олександр",
        "микола",
        "василь",
        "іван",
        "петро",
        "андрій",
        "юрій",
        "віктор",
        "анатолій",
        "григорій",
        "михайло",
        "денис",
        "максим",
        "віталій",
        "дмитро",
        "ігор",
        "павло",
        "роман",
        "богдан",
        "ярослав",
        "владислав",
        "руслан",
        "валерій",
        "степан",
        "леонід",
    }
)


def _extract_clean_surname(person_hint: str) -> str:
    cleaned = re.sub(r"\(.*?\)", "", person_hint or "").strip().lower()
    cleaned = cleaned.replace("’", "'").replace("`", "'")
    parts = [p.strip(".,;:\"'") for p in re.split(r"\s+", cleaned) if p]
    for part in parts:
        # Пропускаємо ініціали, по батькові (-івна/-ович) та імена зі списку IGNORED_SURNAMES
        if (
            len(part) >= 4
            and "." not in part
            and part.isalpha()
            and part not in IGNORED_SURNAMES
            and not part.endswith(("івна", "ївна", "ович", "йович"))
        ):
            return part
    return ""


def _safe_dirname(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", (name or "").strip())
    return re.sub(r"\s+", " ", cleaned)[:100].strip(" .") or "unknown"


class ReportStorage:
    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root
        self.reports_dir = storage_root / "reports"
        self.needs_review_dir = storage_root / "needs_review"
        self.unmatched_dir = storage_root / "unmatched"
        self.ignored_dir = storage_root / "ignored_non_reports"
        self.db_path = storage_root / "state.db"

        for directory in (
            self.storage_root,
            self.reports_dir,
            self.needs_review_dir,
            self.unmatched_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_media (
                    media_id TEXT PRIMARY KEY,
                    file_sha256 TEXT,
                    status TEXT DEFAULT 'matched',
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    channel TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    saved_path TEXT,
                    is_report INTEGER NOT NULL DEFAULT 1,
                    document_type TEXT,
                    document_date TEXT,
                    facility_id TEXT,
                    canonical_name TEXT,
                    region TEXT,
                    match_score REAL,
                    ai_confidence REAL,
                    legibility TEXT DEFAULT 'clear',
                    needs_review INTEGER DEFAULT 0,
                    used_model TEXT,
                    latency_ms INTEGER,
                    match_reason TEXT,
                    vision_json TEXT,
                    processed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stamp_registry (
                    stamp_number TEXT PRIMARY KEY,
                    facility_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    region TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learned_contacts (
                    surname TEXT NOT NULL,
                    full_hint TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    region TEXT NOT NULL,
                    source_doc_type TEXT NOT NULL,
                    occurrences INTEGER DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (surname, facility_id)
                )
                """
            )
            self._migrate_columns(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_sha256 ON processed_media(file_sha256)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_status ON processed_media(status)"
            )

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(processed_media)").fetchall()
        }
        migrations = [
            ("file_sha256", "TEXT"),
            ("status", "TEXT DEFAULT 'matched'"),
            ("retry_count", "INTEGER DEFAULT 0"),
            ("last_error", "TEXT"),
            ("ai_confidence", "REAL"),
            ("legibility", "TEXT DEFAULT 'clear'"),
            ("needs_review", "INTEGER DEFAULT 0"),
            ("used_model", "TEXT"),
            ("latency_ms", "INTEGER"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE processed_media ADD COLUMN {col_name} {col_type}")

        # Нормалізуємо старий дефолтний статус 'processed' -> 'matched'/'unmatched'
        conn.execute(
            """
            UPDATE processed_media
            SET status = CASE
                WHEN facility_id IS NOT NULL THEN 'matched'
                WHEN is_report = 0 THEN 'ignored_non_report'
                ELSE 'unmatched'
            END
            WHERE status = 'processed' OR status IS NULL
            """
        )

        rows_missing_hash = conn.execute(
            "SELECT media_id, source_path FROM processed_media WHERE file_sha256 IS NULL"
        ).fetchall()
        for media_id, src_path in rows_missing_hash:
            path = Path(src_path)
            if path.exists():
                sha = compute_file_sha256(path)
                conn.execute(
                    "UPDATE processed_media SET file_sha256 = ? WHERE media_id = ?",
                    (sha, media_id),
                )

    def is_completed(self, media_id: str) -> bool:
        """
        Повертає True лише якщо файл уже успішно завершив обробку або вичерпав ліміт спроб.
        Файли зі статусом 'processing' (перервані збоєм) або 'failed' (retry_count < 5)
        повертають False і автоматично підхоплюються повторно.
        """
        placeholders = ",".join("?" for _ in COMPLETED_STATUSES)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"""
                SELECT status, COALESCE(retry_count, 0)
                FROM processed_media
                WHERE media_id = ?
                """,
                (media_id,),
            ).fetchone()
            if not row:
                return False
            status, retries = row
            if status in COMPLETED_STATUSES:
                return True
            if status == "failed" and retries >= MAX_RETRY_COUNT:
                return True
            return False

    def mark_processing(self, item: IncomingMediaItem) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO processed_media (
                    media_id, file_sha256, status, channel, source_path, is_report, processed_at
                ) VALUES (?, ?, 'processing', ?, ?, 1, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    status = 'processing',
                    processed_at = excluded.processed_at
                """,
                (
                    item.media_id,
                    item.file_sha256,
                    item.channel,
                    str(item.file_path),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def mark_failed(self, item: IncomingMediaItem, error_msg: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(retry_count, 0) FROM processed_media WHERE media_id = ?",
                (item.media_id,),
            ).fetchone()
            retries = (row[0] if row else 0) + 1
            conn.execute(
                """
                INSERT INTO processed_media (
                    media_id, file_sha256, status, retry_count, last_error,
                    channel, source_path, is_report, processed_at
                ) VALUES (?, ?, 'failed', ?, ?, ?, ?, 1, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    status = 'failed',
                    retry_count = ?,
                    last_error = excluded.last_error,
                    processed_at = excluded.processed_at
                """,
                (
                    item.media_id,
                    item.file_sha256,
                    retries,
                    error_msg[:500],
                    item.channel,
                    str(item.file_path),
                    datetime.now().isoformat(timespec="seconds"),
                    retries,
                ),
            )
            return retries

    def get_unfinished_stats(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM processed_media GROUP BY status"
            ).fetchall()
            return {status: count for status, count in rows}

    def find_duplicate_by_sha256(
        self, file_sha256: Optional[str], current_media_id: str
    ) -> Optional[tuple[str, str]]:
        if not file_sha256:
            return None
        placeholders = ",".join("?" for _ in COMPLETED_STATUSES if _ != "duplicate")
        valid_statuses = [s for s in COMPLETED_STATUSES if s != "duplicate"]
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"""
                SELECT media_id, COALESCE(saved_path, source_path)
                FROM processed_media
                WHERE file_sha256 = ?
                  AND media_id != ?
                  AND status IN ({placeholders})
                LIMIT 1
                """,
                (file_sha256, current_media_id, *valid_statuses),
            ).fetchone()
            return (row[0], row[1]) if row else None

    def record_duplicate(self, item: IncomingMediaItem, original_media_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_media (
                    media_id, file_sha256, status, retry_count, last_error,
                    channel, source_path, saved_path, is_report, match_reason, processed_at
                ) VALUES (?, ?, 'duplicate', 0, NULL, ?, ?, NULL, 0, ?, ?)
                """,
                (
                    item.media_id,
                    item.file_sha256,
                    item.channel,
                    str(item.file_path),
                    f"duplicate_of:{original_media_id}",
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def lookup_stamp(self, stamp_number: Optional[str]) -> Optional[FacilityMatch]:
        if not stamp_number:
            return None
        num = re.sub(r"\D+", "", stamp_number)
        if not num or len(num) > 3:
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT facility_id, canonical_name, region FROM stamp_registry WHERE stamp_number = ?",
                (num,),
            ).fetchone()
        if row:
            return FacilityMatch(
                facility_id=row[0],
                canonical_name=row[1],
                region=row[2],
                score=0.95,
                legibility="clear",
                needs_review=False,
                match_reason=f"stamp_registry:№{num}",
            )
        return None

    def register_stamp(self, stamp_number: Optional[str], match: FacilityMatch) -> None:
        if not stamp_number or match.score < 0.80 or match.needs_review:
            return
        num = re.sub(r"\D+", "", stamp_number)
        if not num or len(num) > 3:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO stamp_registry (stamp_number, facility_id, canonical_name, region, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(stamp_number) DO UPDATE SET
                    facility_id = excluded.facility_id,
                    canonical_name = excluded.canonical_name,
                    region = excluded.region,
                    updated_at = excluded.updated_at
                """,
                (
                    num,
                    match.facility_id,
                    match.canonical_name,
                    match.region,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def learn_contacts(self, vision: VisionAnalysisResult, match: FacilityMatch) -> list[str]:
        """
        Автоматично зберігає прізвища кухарів/директорів із чітких внутрішніх бланків закладу
        (акти, меню, заявки з високим рейтингом впевненості), щоб використовувати їх
        для подальшого розпізнавання складних рукописних звітів цього закладу.
        """
        if (
            match.needs_review
            or match.score < 0.92
            or vision.legibility != "clear"
            or vision.document_type not in ("act", "menu", "application")
            or not vision.person_hints
        ):
            return []

        learned_now: list[str] = []
        now_iso = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as conn:
            for raw_person in vision.person_hints:
                surname = _extract_clean_surname(raw_person)
                if not surname:
                    continue
                conn.execute(
                    """
                    INSERT INTO learned_contacts (
                        surname, full_hint, facility_id, canonical_name,
                        region, source_doc_type, occurrences, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(surname, facility_id) DO UPDATE SET
                        occurrences = learned_contacts.occurrences + 1,
                        full_hint = excluded.full_hint,
                        updated_at = excluded.updated_at
                    """,
                    (
                        surname,
                        raw_person.strip(),
                        match.facility_id,
                        match.canonical_name,
                        match.region,
                        vision.document_type,
                        now_iso,
                    ),
                )
                learned_now.append(surname)
        return learned_now

    def get_learned_contacts_map(self) -> dict[str, list[str]]:
        """Повертає словник {facility_id: [surname, ...]} для прізвищ, які не є наскрізними між >2 школами."""
        with sqlite3.connect(self.db_path) as conn:
            # Відсікаємо прізвища, які зустрічаються більш ніж у 2 закладах (водії/експедитори/поширені однофамільці)
            rows = conn.execute(
                """
                SELECT facility_id, surname
                FROM learned_contacts
                WHERE surname IN (
                    SELECT surname
                    FROM learned_contacts
                    GROUP BY surname
                    HAVING COUNT(DISTINCT facility_id) <= 2
                )
                """
            ).fetchall()
        result: dict[str, list[str]] = {}
        for fac_id, surname in rows:
            result.setdefault(fac_id, []).append(surname)
        return result

    def backfill_learned_contacts_from_history(self) -> int:
        """Заповнює таблицю learned_contacts з уже оброблених чітких актів/меню/заявок у state.db."""
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT facility_id, canonical_name, region, match_score, legibility, document_type, vision_json
                FROM processed_media
                WHERE status = 'matched'
                  AND legibility = 'clear'
                  AND match_score >= 0.92
                  AND document_type IN ('act', 'menu', 'application')
                  AND vision_json IS NOT NULL
                """
            ).fetchall()

        for fac_id, canon_name, region, score, leg, doc_type, vjson in rows:
            vis = VisionAnalysisResult.model_validate(json.loads(vjson))
            match = FacilityMatch(
                facility_id=fac_id,
                canonical_name=canon_name,
                region=region,
                score=score,
                legibility=leg,
                needs_review=False,
                match_reason="history",
            )
            added = self.learn_contacts(vis, match)
            count += len(added)
        return count

    def save_item(
        self,
        item: IncomingMediaItem,
        vision: VisionAnalysisResult,
        match: Optional[FacilityMatch],
        *,
        used_model: str = "",
        latency_ms: int = 0,
        copy_files: bool = True,
    ) -> Optional[Path]:
        date_folder = item.created_at.strftime("%Y-%m-%d")
        ext = item.file_path.suffix or ".jpg"
        short_id = item.media_id[-12:] if len(item.media_id) > 12 else item.media_id

        saved_path: Optional[Path] = None

        if not vision.is_report:
            status = "ignored_non_report"
            filename = f"{date_folder}_{vision.document_type}_{short_id}{ext}"
            target_dir = self.ignored_dir / date_folder
        elif match and not match.needs_review:
            status = "matched"
            filename = f"{date_folder}_{vision.document_type}_{short_id}{ext}"
            region_dir = _safe_dirname(match.region)
            fac_dir = _safe_dirname(f"{match.facility_id}_{match.canonical_name}")
            target_dir = self.reports_dir / region_dir / fac_dir / date_folder
        elif match and match.needs_review:
            status = "needs_review"
            cand_prefix = _safe_dirname(f"{match.facility_id}_{match.canonical_name[:30]}")
            filename = f"REVIEW_{cand_prefix}_{vision.document_type}_{short_id}{ext}"
            target_dir = self.needs_review_dir / date_folder
        else:
            status = "unmatched"
            filename = f"{date_folder}_{vision.document_type}_{short_id}{ext}"
            target_dir = self.unmatched_dir / date_folder

        if copy_files:
            # Якщо файл уже був збережений раніше в іншу папку (наприклад, до консенсусу пакета), прибираємо стару копію
            with sqlite3.connect(self.db_path) as conn:
                prev = conn.execute(
                    "SELECT saved_path FROM processed_media WHERE media_id = ?",
                    (item.media_id,),
                ).fetchone()
                if prev and prev[0]:
                    prev_path = Path(prev[0])
                    if prev_path.exists() and prev_path != (target_dir / filename):
                        prev_path.unlink(missing_ok=True)

            target_dir.mkdir(parents=True, exist_ok=True)
            saved_path = target_dir / filename
            shutil.copy2(item.file_path, saved_path)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_media (
                    media_id, file_sha256, status, retry_count, last_error,
                    channel, source_path, saved_path,
                    is_report, document_type, document_date,
                    facility_id, canonical_name, region,
                    match_score, ai_confidence, legibility, needs_review,
                    used_model, latency_ms, match_reason, vision_json, processed_at
                ) VALUES (?, ?, ?, 0, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.media_id,
                    item.file_sha256,
                    status,
                    item.channel,
                    str(item.file_path),
                    str(saved_path) if saved_path else None,
                    1 if vision.is_report else 0,
                    vision.document_type,
                    vision.document_date,
                    match.facility_id if match else None,
                    match.canonical_name if match else None,
                    match.region if match else None,
                    match.score if match else None,
                    vision.confidence,
                    vision.legibility,
                    1 if (match and match.needs_review) else 0,
                    used_model,
                    latency_ms,
                    match.match_reason if match else None,
                    json.dumps(vision.model_dump(), ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        return saved_path

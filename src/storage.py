from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterator, Optional

from src.connectors.viber_local import compute_file_sha256
from src.date_utils import (
    DOCUMENT_TYPE_LABELS,
    format_month_folder,
    shorten_facility_dirname,
    validate_document_date,
)
from src.image_utils import save_oriented_image
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
        "дженерал",
        "девелопмент",
        "директор",
        "кухар",
        # Внутрішні бухгалтери/операціоністи Рамедаса (з Органіграми, не кухарі шкіл)
        "рачкінда",
        "балаушко",
        "ляшко",
        "тихонюк",
        "горайчук",
        "клюєва",
        "кучарська",
        "антіпова",
        "васькевич",
        "велісейко",
        "оласюк",
        "ярощук",
        "марковець",
        "євтушок",
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


IGNORED_STAMP_NUMBERS = frozenset({"201"})
STAMP_ELIGIBLE_DOC_TYPES = frozenset(
    {
        "act",
        "menu",
        "application",
        "daily_report",
        "waybill_internal",
        "waybill_external",
    }
)


def _normalize_stamp_number(stamp_number: Optional[str]) -> Optional[str]:
    digits = re.sub(r"\D+", "", stamp_number or "")
    if not digits or len(digits) > 3:
        return None
    val = int(digits)
    if val <= 0:
        return None
    norm = str(val)
    if norm in IGNORED_STAMP_NUMBERS:
        return None
    return norm


def _extract_clean_surname(person_hint: str) -> str:
    if re.match(r"^\s*фоп\b", person_hint or "", flags=re.IGNORECASE):
        return ""
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
    def __init__(
        self,
        storage_root: Path,
        *,
        reports_output_dir: Optional[Path] = None,
        document_date_max_age_days: int = 14,
    ) -> None:
        self.storage_root = storage_root
        self.reports_output_dir = reports_output_dir
        self.document_date_max_age_days = max(1, document_date_max_age_days)
        self.reports_root = reports_output_dir or (storage_root / "reports")
        self.reports_dir = self.reports_root / "Щоденні документи"
        self.needs_review_dir = storage_root / "needs_review"
        self.unmatched_dir = storage_root / "unmatched"
        self.ignored_dir = storage_root / "ignored_non_reports"
        self.db_path = storage_root / "state.db"

        local_dirs = [self.storage_root, self.needs_review_dir, self.unmatched_dir]
        if reports_output_dir is None:
            local_dirs.append(self.reports_dir)
        for directory in local_dirs:
            directory.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
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
                    rotation_degrees INTEGER DEFAULT 0,
                    used_model TEXT,
                    latency_ms INTEGER,
                    match_reason TEXT,
                    vision_json TEXT,
                    received_at TEXT,
                    processed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_key TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    delivered_telegram INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
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
                    occurrences INTEGER NOT NULL DEFAULT 1,
                    is_ambiguous INTEGER NOT NULL DEFAULT 0,
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_received_at ON processed_media(received_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alert_events_key_time ON alert_events(alert_key, created_at)"
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
            ("rotation_degrees", "INTEGER DEFAULT 0"),
            ("used_model", "TEXT"),
            ("latency_ms", "INTEGER"),
            ("received_at", "TEXT"),
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
            try:
                if path.exists():
                    sha = compute_file_sha256(path)
                    conn.execute(
                        "UPDATE processed_media SET file_sha256 = ? WHERE media_id = ?",
                        (sha, media_id),
                    )
            except OSError:
                pass

        rows_missing_received = conn.execute(
            "SELECT media_id, saved_path, processed_at FROM processed_media WHERE received_at IS NULL"
        ).fetchall()
        for media_id, saved_path, processed_at in rows_missing_received:
            received_iso = processed_at
            if saved_path:
                try:
                    sp = Path(saved_path)
                    if sp.exists():
                        received_iso = datetime.fromtimestamp(sp.stat().st_mtime).isoformat(
                            timespec="seconds"
                        )
                except OSError:
                    pass
            conn.execute(
                "UPDATE processed_media SET received_at = ? WHERE media_id = ?",
                (received_iso, media_id),
            )

        stamp_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(stamp_registry)").fetchall()
        }
        if "occurrences" not in stamp_cols:
            conn.execute(
                "ALTER TABLE stamp_registry ADD COLUMN occurrences INTEGER NOT NULL DEFAULT 1"
            )
        newly_added_ambiguity = "is_ambiguous" not in stamp_cols
        if newly_added_ambiguity:
            conn.execute(
                "ALTER TABLE stamp_registry ADD COLUMN is_ambiguous INTEGER NOT NULL DEFAULT 0"
            )
            self._reconcile_legacy_stamp_conflicts(conn)

        if IGNORED_STAMP_NUMBERS:
            placeholders = ",".join("?" for _ in IGNORED_STAMP_NUMBERS)
            conn.execute(
                f"DELETE FROM stamp_registry WHERE stamp_number IN ({placeholders})",
                tuple(IGNORED_STAMP_NUMBERS),
            )

    @staticmethod
    def _reconcile_legacy_stamp_conflicts(conn: sqlite3.Connection) -> None:
        """Позначає неоднозначними старі штампи, які в історії траплялися на різних закладах."""
        facilities_by_stamp: dict[str, set[str]] = {}
        rows = conn.execute(
            """
            SELECT facility_id, vision_json
            FROM processed_media
            WHERE facility_id IS NOT NULL AND vision_json IS NOT NULL
            """
        ).fetchall()
        for facility_id, raw_json in rows:
            try:
                stamp_raw = json.loads(raw_json).get("stamp_number")
            except (TypeError, json.JSONDecodeError):
                continue
            norm = _normalize_stamp_number(stamp_raw)
            if norm:
                facilities_by_stamp.setdefault(norm, set()).add(facility_id)

        for norm_stamp, fids in facilities_by_stamp.items():
            if len(fids) > 1:
                conn.execute(
                    "UPDATE stamp_registry SET is_ambiguous = 1 WHERE stamp_number = ?",
                    (norm_stamp,),
                )

    def is_completed(self, media_id: str) -> bool:
        """
        Повертає True лише якщо файл уже успішно завершив обробку або вичерпав ліміт спроб.
        Файли зі статусом 'processing' (перервані збоєм) або 'failed' (retry_count < 5)
        повертають False і автоматично підхоплюються повторно.
        """
        with self._connect() as conn:
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
        received_iso = item.created_at.isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_media (
                    media_id, file_sha256, status, channel, source_path, is_report, received_at, processed_at
                ) VALUES (?, ?, 'processing', ?, ?, 1, ?, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    status = 'processing',
                    received_at = COALESCE(processed_media.received_at, excluded.received_at),
                    processed_at = excluded.processed_at
                """,
                (
                    item.media_id,
                    item.file_sha256,
                    item.channel,
                    str(item.file_path),
                    received_iso,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def mark_failed(self, item: IncomingMediaItem, error_msg: str) -> int:
        received_iso = item.created_at.isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(retry_count, 0) FROM processed_media WHERE media_id = ?",
                (item.media_id,),
            ).fetchone()
            retries = (row[0] if row else 0) + 1
            conn.execute(
                """
                INSERT INTO processed_media (
                    media_id, file_sha256, status, retry_count, last_error,
                    channel, source_path, is_report, received_at, processed_at
                ) VALUES (?, ?, 'failed', ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(media_id) DO UPDATE SET
                    status = 'failed',
                    retry_count = ?,
                    last_error = excluded.last_error,
                    received_at = COALESCE(processed_media.received_at, excluded.received_at),
                    processed_at = excluded.processed_at
                """,
                (
                    item.media_id,
                    item.file_sha256,
                    retries,
                    error_msg[:500],
                    item.channel,
                    str(item.file_path),
                    received_iso,
                    datetime.now().isoformat(timespec="seconds"),
                    retries,
                ),
            )
            return retries

    def get_unfinished_stats(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM processed_media GROUP BY status"
            ).fetchall()
            return {status: count for status, count in rows}

    def get_final_status_media_ids(self) -> set[str]:
        """Повертає множину media_id, які вже мають фінальний статус обробки у state.db."""
        placeholders = ",".join("?" for _ in COMPLETED_STATUSES)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT media_id FROM processed_media WHERE status IN ({placeholders})",
                COMPLETED_STATUSES,
            ).fetchall()
            return {row[0] for row in rows}

    def find_duplicate_by_sha256(
        self, file_sha256: Optional[str], current_media_id: str
    ) -> Optional[tuple[str, str]]:
        if not file_sha256:
            return None
        placeholders = ",".join("?" for _ in COMPLETED_STATUSES if _ != "duplicate")
        valid_statuses = [s for s in COMPLETED_STATUSES if s != "duplicate"]
        with self._connect() as conn:
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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_media (
                    media_id, file_sha256, status, retry_count, last_error,
                    channel, source_path, saved_path, is_report, match_reason, received_at, processed_at
                ) VALUES (?, ?, 'duplicate', 0, NULL, ?, ?, NULL, 0, ?, ?, ?)
                """,
                (
                    item.media_id,
                    item.file_sha256,
                    item.channel,
                    str(item.file_path),
                    f"duplicate_of:{original_media_id}",
                    item.created_at.isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def record_alert(
        self,
        alert_key: str,
        severity: str,
        message: str,
        *,
        delivered_telegram: bool,
        created_at: Optional[datetime] = None,
    ) -> None:
        ts = (created_at or datetime.now()).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alert_events (alert_key, severity, message, delivered_telegram, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alert_key, severity, message[:1000], 1 if delivered_telegram else 0, ts),
            )

    def get_last_alert_time(self, alert_key: str) -> Optional[datetime]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(created_at) FROM alert_events WHERE alert_key = ?",
                (alert_key,),
            ).fetchone()
        if row and row[0]:
            try:
                return datetime.fromisoformat(row[0])
            except ValueError:
                return None
        return None

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT alert_key, severity, message, delivered_telegram, created_at
                FROM alert_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "alert_key": r[0],
                "severity": r[1],
                "message": r[2],
                "delivered_telegram": bool(r[3]),
                "created_at": r[4],
            }
            for r in rows
        ]

    def get_last_arrival_time(self) -> Optional[datetime]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(COALESCE(received_at, processed_at)) FROM processed_media"
            ).fetchone()
        if row and row[0]:
            try:
                return datetime.fromisoformat(row[0][:19])
            except ValueError:
                return None
        return None

    def get_activity_stats(
        self,
        *,
        days: int = 14,
        work_start_hour: int = 9,
        work_end_hour: int = 18,
        now: Optional[datetime] = None,
    ) -> dict:
        """
        Повертає статистику надходження файлів за останні N днів:
        - розподіл по годинах доби (00..23) та частка у робочий час;
        - статистика по громадах/регіонах (кількість, останній прихід, пікова година);
        - кількість надходжень за сьогодні.
        """
        ref_now = now or datetime.now()
        since_iso = (ref_now - timedelta(days=days)).isoformat(timespec="seconds")
        today_prefix = ref_now.strftime("%Y-%m-%d")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(received_at, processed_at) AS ts, region, status
                FROM processed_media
                WHERE COALESCE(received_at, processed_at) >= ?
                """,
                (since_iso,),
            ).fetchall()

        hourly_counts: dict[int, int] = {h: 0 for h in range(24)}
        today_count = 0
        work_hours_count = 0
        total_count = 0
        last_arrival: Optional[str] = None
        by_region: dict[str, dict] = {}

        for ts_str, region, _status in rows:
            if not ts_str:
                continue
            ts_clean = ts_str[:19]
            try:
                dt = datetime.fromisoformat(ts_clean)
            except ValueError:
                continue

            total_count += 1
            hourly_counts[dt.hour] += 1
            if work_start_hour <= dt.hour < work_end_hour:
                work_hours_count += 1
            if ts_clean.startswith(today_prefix):
                today_count += 1
            if last_arrival is None or ts_clean > last_arrival:
                last_arrival = ts_clean

            reg_key = region or "Невизначено / Інше"
            reg_info = by_region.setdefault(
                reg_key,
                {"count": 0, "last_arrival": ts_clean, "hours": {}},
            )
            reg_info["count"] += 1
            if ts_clean > reg_info["last_arrival"]:
                reg_info["last_arrival"] = ts_clean
            reg_info["hours"][dt.hour] = reg_info["hours"].get(dt.hour, 0) + 1

        region_summary = []
        for reg_name, info in sorted(
            by_region.items(), key=lambda kv: kv[1]["count"], reverse=True
        ):
            peak_hour = max(info["hours"], key=info["hours"].get)
            region_summary.append(
                {
                    "region": reg_name,
                    "count": info["count"],
                    "last_arrival": info["last_arrival"],
                    "peak_hour": peak_hour,
                }
            )

        return {
            "days": days,
            "total_count": total_count,
            "today_count": today_count,
            "work_hours_count": work_hours_count,
            "last_arrival": last_arrival,
            "hourly_counts": hourly_counts,
            "by_region": region_summary,
        }

    def get_daily_summary(self, target_date: Optional[date] = None) -> dict:
        """Повертає зведену статистику оброблених фото за конкретну дату (YYYY-MM-DD)."""
        day_str = (target_date or date.today()).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, document_type, facility_id, region,
                       COALESCE(received_at, processed_at) AS ts
                FROM processed_media
                WHERE COALESCE(received_at, processed_at) LIKE ?
                """,
                (f"{day_str}%",),
            ).fetchall()

        status_counts: dict[str, int] = {}
        doc_counts: dict[str, int] = {}
        region_counts: dict[str, int] = {}
        facilities: set[str] = set()
        last_ts: Optional[str] = None

        for status, doc_type, fac_id, region, ts in rows:
            st = status or "unknown"
            status_counts[st] = status_counts.get(st, 0) + 1
            if ts and (last_ts is None or ts > last_ts):
                last_ts = ts
            if st == "matched":
                if fac_id:
                    facilities.add(fac_id)
                if doc_type:
                    doc_counts[doc_type] = doc_counts.get(doc_type, 0) + 1
                if region:
                    region_counts[region] = region_counts.get(region, 0) + 1

        return {
            "date": day_str,
            "total": len(rows),
            "matched": status_counts.get("matched", 0),
            "needs_review": status_counts.get("needs_review", 0),
            "unmatched": status_counts.get("unmatched", 0),
            "ignored_non_report": status_counts.get("ignored_non_report", 0),
            "duplicate": status_counts.get("duplicate", 0),
            "failed": status_counts.get("failed", 0),
            "processing": status_counts.get("processing", 0),
            "unique_facilities": len(facilities),
            "unique_regions": len(region_counts),
            "doc_types": sorted(doc_counts.items(), key=lambda kv: (-kv[1], kv[0])),
            "top_regions": sorted(region_counts.items(), key=lambda kv: (-kv[1], kv[0])),
            "last_arrival": last_ts[:19] if last_ts else None,
        }

    def lookup_stamp(self, stamp_number: Optional[str]) -> Optional[FacilityMatch]:
        num = _normalize_stamp_number(stamp_number)
        if not num:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT facility_id, canonical_name, region
                FROM stamp_registry
                WHERE stamp_number = ? AND COALESCE(is_ambiguous, 0) = 0
                """,
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

    def register_stamp(
        self,
        stamp_number: Optional[str],
        match: FacilityMatch,
        *,
        document_type: Optional[str] = None,
    ) -> None:
        if match.score < 0.80 or match.needs_review:
            return
        if match.match_reason.startswith("stamp_registry:"):
            return
        if document_type and document_type not in STAMP_ELIGIBLE_DOC_TYPES:
            return
        num = _normalize_stamp_number(stamp_number)
        if not num:
            return

        now_iso = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT facility_id FROM stamp_registry WHERE stamp_number = ?",
                (num,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO stamp_registry (
                        stamp_number, facility_id, canonical_name, region, occurrences, is_ambiguous, updated_at
                    ) VALUES (?, ?, ?, ?, 1, 0, ?)
                    """,
                    (num, match.facility_id, match.canonical_name, match.region, now_iso),
                )
                return

            if row[0] == match.facility_id:
                conn.execute(
                    """
                    UPDATE stamp_registry
                    SET canonical_name = ?,
                        region = ?,
                        occurrences = COALESCE(occurrences, 1) + 1,
                        updated_at = ?
                    WHERE stamp_number = ?
                    """,
                    (match.canonical_name, match.region, now_iso, num),
                )
            else:
                conn.execute(
                    """
                    UPDATE stamp_registry
                    SET is_ambiguous = 1,
                        occurrences = COALESCE(occurrences, 1) + 1,
                        updated_at = ?
                    WHERE stamp_number = ?
                    """,
                    (now_iso, num),
                )
                log.warning(
                    "Конфлікт штампа №%s (%s vs %s) — позначено як неоднозначний",
                    num,
                    row[0],
                    match.facility_id,
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
        with self._connect() as conn:
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

    def get_learned_contacts_map(
        self,
        *,
        min_occurrences: int = 2,
        max_age_days: int = 90,
    ) -> dict[str, list[str]]:
        """
        Повертає словник {facility_id: [surname, ...]} для надійних прізвищ кухарів:
        - зустрічались щонайменше min_occurrences разів (відсікає поодинокі опечатки/заміни);
        - оновлювались протягом останніх max_age_days днів (захист при переході кухаря в іншу школу);
        - не є наскрізними між >2 школами (відсікає водіїв/поширені прізвища).
        """
        cutoff_iso = (datetime.now() - timedelta(days=max(1, max_age_days))).isoformat(
            timespec="seconds"
        )
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT facility_id, surname
                FROM learned_contacts
                WHERE occurrences >= ?
                  AND updated_at >= ?
                  AND surname IN (
                      SELECT surname
                      FROM learned_contacts
                      WHERE updated_at >= ?
                      GROUP BY surname
                      HAVING COUNT(DISTINCT facility_id) <= 2
                  )
                """,
                (max(1, min_occurrences), cutoff_iso, cutoff_iso),
            ).fetchall()
        result: dict[str, list[str]] = {}
        for fac_id, surname in rows:
            result.setdefault(fac_id, []).append(surname)
        return result

    def backfill_learned_contacts_from_history(self, *, force: bool = False) -> int:
        """Заповнює таблицю learned_contacts з уже оброблених чітких актів/меню/заявок у state.db."""
        with self._connect() as conn:
            if not force:
                has_any = conn.execute("SELECT 1 FROM learned_contacts LIMIT 1").fetchone()
                if has_any:
                    return 0
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

        count = 0
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
        received_date = item.created_at.strftime("%Y-%m-%d")
        doc_date_str, _ = validate_document_date(
            vision.document_date,
            item.created_at,
            max_past_days=self.document_date_max_age_days,
        )
        doc_date_obj = date.fromisoformat(doc_date_str)
        month_folder = format_month_folder(doc_date_obj)
        doc_label = DOCUMENT_TYPE_LABELS.get(vision.document_type, vision.document_type)
        ext = item.file_path.suffix or ".jpg"
        short_id = item.media_id[-12:] if len(item.media_id) > 12 else item.media_id

        saved_path: Optional[Path] = None

        if not vision.is_report:
            status = "ignored_non_report"
            filename = f"{received_date}_{doc_label}_{short_id}{ext}"
            target_dir = self.ignored_dir / received_date
        elif match and not match.needs_review:
            status = "matched"
            filename = f"{doc_date_str}_{doc_label}_{short_id}{ext}"
            region_dir = _safe_dirname(match.region)
            fac_dir = _safe_dirname(shorten_facility_dirname(match.canonical_name))
            target_dir = (
                self.reports_dir / region_dir / fac_dir / month_folder / doc_date_str
            )
            if vision.document_type == "food_photo":
                target_dir = target_dir / "Контрольна порція"
        elif match and match.needs_review:
            status = "needs_review"
            cand_prefix = _safe_dirname(
                f"{match.facility_id}_{shorten_facility_dirname(match.canonical_name, max_len=35)}"
            )
            filename = f"REVIEW_{cand_prefix}_{doc_label}_{short_id}{ext}"
            target_dir = self.needs_review_dir / doc_date_str
        else:
            status = "unmatched"
            filename = f"{doc_date_str}_{doc_label}_{short_id}{ext}"
            target_dir = self.unmatched_dir / doc_date_str
            if vision.document_type == "food_photo":
                target_dir = target_dir / "Контрольна порція"

        if copy_files:
            # Якщо файл уже був збережений раніше в іншу папку (наприклад, до консенсусу пакета), прибираємо стару копію
            with self._connect() as conn:
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
            save_oriented_image(
                item.file_path,
                saved_path,
                rotation_degrees=vision.rotation_degrees,
            )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_media (
                    media_id, file_sha256, status, retry_count, last_error,
                    channel, source_path, saved_path,
                    is_report, document_type, document_date,
                    facility_id, canonical_name, region,
                    match_score, ai_confidence, legibility, needs_review,
                    rotation_degrees, used_model, latency_ms, match_reason, vision_json, received_at, processed_at
                ) VALUES (?, ?, ?, 0, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    vision.rotation_degrees,
                    used_model,
                    latency_ms,
                    match.match_reason if match else None,
                    json.dumps(vision.model_dump(), ensure_ascii=False),
                    item.created_at.isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        return saved_path

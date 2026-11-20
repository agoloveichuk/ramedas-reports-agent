from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import shutil
from typing import Callable, Optional

from src.connectors.base import BaseConnector
from src.models import IncomingMediaItem

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(slots=True)
class ViberRotationStats:
    """Статистика очищення/архівації старих файлів із папок завантажень."""

    scanned: int = 0
    rotated: int = 0
    archived: int = 0
    deleted: int = 0
    skipped_unprocessed: int = 0
    freed_bytes: int = 0
    errors: int = 0

    def merge(self, other: ViberRotationStats) -> None:
        self.scanned += other.scanned
        self.rotated += other.rotated
        self.archived += other.archived
        self.deleted += other.deleted
        self.skipped_unprocessed += other.skipped_unprocessed
        self.freed_bytes += other.freed_bytes
        self.errors += other.errors


def compute_file_sha256(file_path: Path) -> str:
    """Обчислює SHA-256 хеш байтового вмісту файлу для надійної дедуплікації."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ViberLocalConnector(BaseConnector):
    """
    Конектор до локального клієнта Viber Desktop на Windows.
    Читає автоматично завантажені групові фото з папки ViberDownloads (read-only),
    обчислює SHA-256 хеш для відсіювання дублікатів та групує їх у пакети.
    """

    channel_name = "viber"

    def __init__(
        self,
        downloads_dir: Path,
        is_processed_fn: Callable[[str], bool],
    ) -> None:
        self.downloads_dir = downloads_dir
        self.is_processed_fn = is_processed_fn

    def fetch_pending_items(self) -> list[IncomingMediaItem]:
        if not self.downloads_dir.exists():
            return []

        items: list[IncomingMediaItem] = []
        for path in self.downloads_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            stat = path.stat()
            if stat.st_size < 1024:
                continue

            media_id = path.stem
            if self.is_processed_fn(media_id):
                continue

            created_at = datetime.fromtimestamp(stat.st_mtime)
            file_sha256 = compute_file_sha256(path)
            items.append(
                IncomingMediaItem(
                    channel="viber",
                    media_id=media_id,
                    file_path=path,
                    file_sha256=file_sha256,
                    created_at=created_at,
                )
            )

        items.sort(key=lambda x: x.created_at)
        return items

    def rotate_old_files(
        self,
        *,
        retention_days: int,
        final_media_ids: set[str],
        archive_dir: Optional[Path] = None,
        dry_run: bool = False,
        now: Optional[datetime] = None,
    ) -> ViberRotationStats:
        """
        Видаляє або переносить в архів файли з ViberDownloads, старші за retention_days,
        ЛИШЕ якщо їхній media_id вже має фінальний статус у state.db.
        """
        return rotate_directory_old_files(
            self.downloads_dir,
            retention_days=retention_days,
            final_media_ids=final_media_ids,
            archive_dir=archive_dir,
            dry_run=dry_run,
            now=now,
        )


def rotate_directory_old_files(
    downloads_dir: Path,
    *,
    retention_days: int,
    final_media_ids: set[str],
    archive_dir: Optional[Path] = None,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> ViberRotationStats:
    """
    Видаляє або переносить в архів зображення у downloads_dir, старші за retention_days,
    ЛИШЕ якщо їхній media_id (path.stem) вже має фінальний статус у state.db.
    """
    stats = ViberRotationStats()
    if retention_days <= 0 or not downloads_dir.exists():
        return stats

    cutoff = (now or datetime.now()) - timedelta(days=retention_days)

    for path in downloads_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stats.scanned += 1
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        if mtime > cutoff:
            continue

        if path.stem not in final_media_ids:
            stats.skipped_unprocessed += 1
            continue

        if dry_run:
            stats.rotated += 1
            stats.freed_bytes += stat.st_size
            if archive_dir is not None:
                stats.archived += 1
            else:
                stats.deleted += 1
            continue

        try:
            if archive_dir is not None:
                month_dir = archive_dir / mtime.strftime("%Y-%m")
                month_dir.mkdir(parents=True, exist_ok=True)
                dest = month_dir / path.name
                if dest.exists():
                    if dest.stat().st_size == stat.st_size:
                        path.unlink()
                    else:
                        dest = month_dir / f"{path.stem}_{int(stat.st_mtime)}{path.suffix}"
                        shutil.move(str(path), str(dest))
                else:
                    shutil.move(str(path), str(dest))
                stats.archived += 1
            else:
                path.unlink()
                stats.deleted += 1

            stats.rotated += 1
            stats.freed_bytes += stat.st_size
        except OSError:
            stats.errors += 1

    return stats


def group_into_burst_batches(
    items: list[IncomingMediaItem], max_gap_seconds: int = 6
) -> list[list[IncomingMediaItem]]:
    """
    Групує послідовні фото з ViberDownloads у пакети (burst-батчі),
    якщо інтервал між сусідніми файлами не перевищує max_gap_seconds.
    """
    if not items:
        return []

    sorted_items = sorted(items, key=lambda x: x.created_at)
    batches: list[list[IncomingMediaItem]] = [[sorted_items[0]]]

    for item in sorted_items[1:]:
        prev = batches[-1][-1]
        gap = abs((item.created_at - prev.created_at).total_seconds())
        if gap <= max_gap_seconds:
            batches[-1].append(item)
        else:
            batches.append([item])

    return batches

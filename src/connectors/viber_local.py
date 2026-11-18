from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Callable

from src.connectors.base import BaseConnector
from src.models import IncomingMediaItem


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
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
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

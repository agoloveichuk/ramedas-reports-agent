from __future__ import annotations

from abc import ABC, abstractmethod
from src.models import IncomingMediaItem


class BaseConnector(ABC):
    """Базовий інтерфейс джерела фотозвітів."""

    channel_name: str

    @abstractmethod
    def fetch_pending_items(self) -> list[IncomingMediaItem]:
        """Повертає список нових медіафайлів, впорядкованих за часом створення."""
        raise NotImplementedError

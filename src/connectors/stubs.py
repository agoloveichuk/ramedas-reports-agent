from __future__ import annotations

from src.connectors.base import BaseConnector
from src.models import IncomingMediaItem


class TelegramConnectorStub(BaseConnector):
    """Заготовка для майбутнього підключення груп Telegram (aiogram / Telethon)."""

    channel_name = "telegram"

    def fetch_pending_items(self) -> list[IncomingMediaItem]:
        return []


class WahaConnectorStub(BaseConnector):
    """Заготовка для майбутнього підключення WhatsApp через WAHA HTTP API."""

    channel_name = "waha"

    def fetch_pending_items(self) -> list[IncomingMediaItem]:
        return []

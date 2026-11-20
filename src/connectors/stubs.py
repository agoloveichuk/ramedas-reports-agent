from __future__ import annotations

from src.connectors.base import BaseConnector
from src.connectors.telegram_user import TelegramUserConnector
from src.models import IncomingMediaItem

TelegramConnectorStub = TelegramUserConnector


class WahaConnectorStub(BaseConnector):
    """Заготовка для майбутнього підключення WhatsApp через WAHA HTTP API."""

    channel_name = "waha"

    def fetch_pending_items(self) -> list[IncomingMediaItem]:
        return []

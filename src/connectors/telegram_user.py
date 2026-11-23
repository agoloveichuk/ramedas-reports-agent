"""
Конектор до Telegram через User API (MTProto / Telethon) для читання фотозвітів
із групових чатів під робочим номером (без потреби додавати бота в кожну групу).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.connectors.base import BaseConnector
from src.connectors.viber_local import compute_file_sha256
from src.logger import get_logger
from src.models import IncomingMediaItem

log = get_logger("telegram")

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def build_telegram_media_id(chat_id: int, message_id: int) -> str:
    return f"tg_{chat_id}_{message_id}"


def dialog_matches_filter(
    dialog_id: int,
    title: str,
    *,
    is_group: bool,
    is_channel: bool,
    allowed_chats: tuple[str, ...] = (),
) -> bool:
    """
    Визначає, чи потрібно читати фото з цього діалогу.
    Якщо allowed_chats порожній — беруться лише групові чати/супергрупи (без особистих 1:1 діалогів).
    """
    if not allowed_chats:
        return bool(is_group)

    id_str = str(dialog_id)
    title_low = (title or "").lower()
    for pattern in allowed_chats:
        pat = pattern.strip()
        if not pat:
            continue
        if pat.lstrip("-").isdigit():
            if id_str == pat or id_str.lstrip("-") == pat.lstrip("-"):
                return True
        elif pat.lower() in title_low:
            return True
    return False


def is_supported_media_message(msg: Any) -> tuple[bool, str]:
    """Перевіряє, чи містить повідомлення Telegram фото або нестиснуте зображення-документ."""
    if getattr(msg, "photo", None) is not None:
        return True, ".jpg"

    doc = getattr(msg, "document", None)
    if doc is not None:
        mime = (getattr(doc, "mime_type", "") or "").lower()
        if mime in _MIME_TO_EXT:
            return True, _MIME_TO_EXT[mime]
        if mime.startswith("image/"):
            return True, ".jpg"
    return False, ""


def extract_sender_info(sender: Any) -> tuple[Optional[str], Optional[str]]:
    """Дістає ім'я/назву та номер телефону (якщо доступний) відправника повідомлення."""
    if sender is None:
        return None, None
    first = (getattr(sender, "first_name", None) or "").strip()
    last = (getattr(sender, "last_name", None) or "").strip()
    full_name = " ".join(part for part in (first, last) if part).strip()
    if not full_name:
        full_name = (
            (getattr(sender, "title", None) or "").strip()
            or (getattr(sender, "username", None) or "").strip()
        )
    phone = getattr(sender, "phone", None) or None
    return (full_name or None), phone


def _to_local_naive(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime.now()
    if dt.tzinfo is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


class TelegramUserConnector(BaseConnector):
    """
    Читає вхідні фотозвіти з групових чатів Telegram через обліковий запис користувача (Telethon).
    Якщо TELEGRAM_API_ID / TELEGRAM_API_HASH не задані або сесія ще не авторизована —
    безпечно повертає порожній список, не блокуючи основний пайплайн Viber.
    """

    channel_name = "telegram"

    def __init__(
        self,
        *,
        api_id: int,
        api_hash: str,
        session_path: Path,
        downloads_dir: Path,
        is_processed_fn: Callable[[str], bool],
        lookback_hours: int = 72,
        allowed_chats: tuple[str, ...] = (),
        client_factory: Optional[Callable[[str, int, str], Any]] = None,
    ) -> None:
        self.api_id = api_id
        self.api_hash = (api_hash or "").strip()
        self.session_path = session_path
        self.downloads_dir = downloads_dir
        self.is_processed_fn = is_processed_fn
        self.lookback_hours = max(0, lookback_hours)
        self.allowed_chats = allowed_chats
        self._client_factory = client_factory
        self._warned_no_session = False

    @property
    def is_configured(self) -> bool:
        return self.api_id > 0 and bool(self.api_hash)

    @property
    def has_session(self) -> bool:
        if self.session_path.exists():
            return True
        alt = Path(f"{self.session_path}.session")
        return alt.exists()

    @property
    def status_label(self) -> str:
        if not self.is_configured:
            return "disabled (set TELEGRAM_API_ID & TELEGRAM_API_HASH)"
        if not self.has_session:
            return "needs login (run --telegram-login)"
        return f"ready ({self.session_path.name})"

    def _create_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(str(self.session_path), self.api_id, self.api_hash)
        from telethon import TelegramClient

        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        return TelegramClient(str(self.session_path), self.api_id, self.api_hash)

    def fetch_pending_items(self) -> list[IncomingMediaItem]:
        if not self.is_configured:
            return []
        if self._client_factory is None and not self.has_session:
            if not self._warned_no_session:
                log.warning(
                    "Telegram API налаштовано, але файл сесії відсутній (%s). "
                    "Запустіть --telegram-login для одноразової авторизації.",
                    self.session_path,
                )
                self._warned_no_session = True
            return []

        try:
            return asyncio.run(self._fetch_pending_async())
        except Exception as exc:
            log.error("Помилка читання повідомлень із Telegram User API: %s", exc)
            return []

    async def _fetch_pending_async(self) -> list[IncomingMediaItem]:
        client = self._create_client()
        items: list[IncomingMediaItem] = []
        # lookback_hours <= 0 означає повне вивантаження всієї історії чатів без часового ліміту
        cutoff_utc = (
            datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
            if self.lookback_hours > 0
            else None
        )
        msg_limit = 250 if self.lookback_hours > 0 else None

        try:
            await client.connect()
            if not await client.is_user_authorized():
                if not self._warned_no_session:
                    log.warning(
                        "Telegram-сесія не авторизована. Запустіть --telegram-login."
                    )
                    self._warned_no_session = True
                return []

            self.downloads_dir.mkdir(parents=True, exist_ok=True)

            async for dialog in client.iter_dialogs():
                if not dialog_matches_filter(
                    dialog.id,
                    dialog.title or "",
                    is_group=bool(getattr(dialog, "is_group", False)),
                    is_channel=bool(getattr(dialog, "is_channel", False)),
                    allowed_chats=self.allowed_chats,
                ):
                    continue

                async for msg in client.iter_messages(dialog.entity, limit=msg_limit):
                    msg_dt = getattr(msg, "date", None)
                    if cutoff_utc is not None and msg_dt is not None:
                        msg_utc = (
                            msg_dt.astimezone(timezone.utc)
                            if msg_dt.tzinfo
                            else msg_dt.replace(tzinfo=timezone.utc)
                        )
                        if msg_utc < cutoff_utc:
                            break

                    supported, ext = is_supported_media_message(msg)
                    if not supported:
                        continue

                    media_id = build_telegram_media_id(dialog.id, msg.id)
                    if self.is_processed_fn(media_id):
                        continue

                    dest_path = self.downloads_dir / f"{media_id}{ext}"
                    if not dest_path.exists():
                        downloaded = await client.download_media(msg, file=str(dest_path))
                        if not downloaded:
                            continue
                        dest_path = Path(downloaded)

                    if not dest_path.exists() or dest_path.stat().st_size < 512:
                        continue

                    sender = await msg.get_sender()
                    sender_name, sender_phone = extract_sender_info(sender)

                    items.append(
                        IncomingMediaItem(
                            channel="telegram",
                            media_id=media_id,
                            file_path=dest_path,
                            file_sha256=compute_file_sha256(dest_path),
                            created_at=_to_local_naive(msg_dt),
                            chat_title=(dialog.title or "").strip() or None,
                            sender_name=sender_name,
                            sender_phone=sender_phone,
                            caption=(getattr(msg, "message", None) or "").strip() or None,
                        )
                    )
        finally:
            await client.disconnect()

        items.sort(key=lambda x: x.created_at)
        return items

    def interactive_login(self) -> bool:
        """Одноразова інтерактивна авторизація робочого акаунта Telegram (телефон + код)."""
        if not self.is_configured:
            raise RuntimeError(
                "Спочатку заповніть TELEGRAM_API_ID та TELEGRAM_API_HASH у .env"
            )
        return asyncio.run(self._interactive_login_async())

    async def _interactive_login_async(self) -> bool:
        client = self._create_client()
        try:
            await client.start()
            me = await client.get_me()
            name = " ".join(
                p for p in (getattr(me, "first_name", ""), getattr(me, "last_name", "")) if p
            ).strip()
            phone = getattr(me, "phone", "")
            log.info(
                "Telegram User API авторизовано: %s (+%s), сесія: %s",
                name or getattr(me, "username", "unknown"),
                phone or "?",
                self.session_path,
            )
            return True
        finally:
            await client.disconnect()

    def list_dialogs(self) -> list[dict[str, Any]]:
        """Повертає список групових чатів і каналів для налаштування TELEGRAM_ALLOWED_CHATS."""
        if not self.is_configured:
            raise RuntimeError(
                "Спочатку заповніть TELEGRAM_API_ID та TELEGRAM_API_HASH у .env"
            )
        return asyncio.run(self._list_dialogs_async())

    async def _list_dialogs_async(self) -> list[dict[str, Any]]:
        client = self._create_client()
        result: list[dict[str, Any]] = []
        try:
            await client.connect()
            if not await client.is_user_authorized():
                raise RuntimeError(
                    "Telegram-сесія не авторизована. Спочатку запустіть --telegram-login."
                )
            async for dialog in client.iter_dialogs():
                is_group = bool(getattr(dialog, "is_group", False))
                is_channel = bool(getattr(dialog, "is_channel", False))
                if not (is_group or is_channel):
                    continue
                result.append(
                    {
                        "id": dialog.id,
                        "title": dialog.title or "",
                        "is_group": is_group,
                        "is_channel": is_channel,
                        "active": dialog_matches_filter(
                            dialog.id,
                            dialog.title or "",
                            is_group=is_group,
                            is_channel=is_channel,
                            allowed_chats=self.allowed_chats,
                        ),
                    }
                )
        finally:
            await client.disconnect()
        return result

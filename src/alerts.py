"""
Клієнт технічних сповіщень у Telegram.
Працює безпечно навіть без налаштованого бота: якщо TELEGRAM_BOT_TOKEN або TECH_ALERT_CHAT_ID
порожні, алерти не падають з помилкою, а фіксуються у логах та локальній базі state.db.
Підтримує hot-reload токенів із .env без перезапуску процесу.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from dotenv import dotenv_values
import httpx

from src.config import PROJECT_ROOT, Settings
from src.logger import get_logger

log = get_logger("alerts")


class TelegramNotifier:
    """Відправка технічних сповіщень у Telegram Bot API з підтримкою hot-reload .env."""

    def __init__(
        self,
        settings: Settings,
        *,
        env_path: Optional[Path] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.settings = settings
        self.env_path = env_path if env_path is not None else (PROJECT_ROOT / ".env")
        self.timeout_seconds = timeout_seconds
        self._bot_token: str = settings.telegram_bot_token.strip()
        self._chat_id: str = settings.tech_alert_chat_id.strip()
        self._last_env_mtime: float = (
            self.env_path.stat().st_mtime if self.env_path.exists() else 0.0
        )

    def _reload_env_if_changed(self) -> None:
        if not self.env_path.exists():
            return
        mtime = self.env_path.stat().st_mtime
        if mtime == self._last_env_mtime:
            return
        self._last_env_mtime = mtime
        values = dotenv_values(self.env_path)
        token = (values.get("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id = (values.get("TECH_ALERT_CHAT_ID") or "").strip()
        if token != self._bot_token or chat_id != self._chat_id:
            self._bot_token = token
            self._chat_id = chat_id
            state = "активовано" if self.is_configured else "вимкнено"
            log.info("[Hot-Reload] Telegram-сповіщення: %s", state)

    @property
    def is_configured(self) -> bool:
        self._reload_env_if_changed()
        return bool(self._bot_token and self._chat_id)

    def send(self, text: str) -> bool:
        """
        Надсилає повідомлення у Telegram.
        Повертає True лише якщо повідомлення реально доставлене через Telegram Bot API.
        Якщо бот ще не підключений — логує повідомлення локально та повертає False.
        """
        self._reload_env_if_changed()
        if not (self._bot_token and self._chat_id):
            log.warning("[ALERT][LOCAL-ONLY, бот не підключений] %s", text)
            return False

        endpoint = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
            log.info("[ALERT][TELEGRAM-SENT] %s", text.splitlines()[0][:120])
            return True
        except Exception as exc:
            log.error("[ALERT][TELEGRAM-ERROR] Не вдалося надіслати в Telegram: %s", exc)
            return False

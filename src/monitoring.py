"""
Моніторинг активності груп у робочий час (09:00–18:00), контроль працездатності Viber.exe
та відправка технічних алертів із захистом від спаму (cooldown у state.db).
"""

from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta
from typing import Optional

from src.alerts import TelegramNotifier
from src.config import Settings
from src.logger import get_logger
from src.storage import ReportStorage
from src.viber_watchdog import ViberProcessWatchdog

log = get_logger("monitor")


def parse_work_time(hhmm: str, fallback: dt_time) -> dt_time:
    try:
        parts = (hhmm or "").strip().split(":")
        return dt_time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return fallback


def parse_work_days(csv_days: str) -> set[int]:
    days: set[int] = set()
    for token in (csv_days or "").split(","):
        token = token.strip()
        if token.isdigit():
            val = int(token)
            if 1 <= val <= 7:
                days.add(val)
    return days or {1, 2, 3, 4, 5}


class ActivityMonitor:
    """Контроль активності чатів у робочий час та здоров'я Viber/LLM-пайплайну."""

    def __init__(
        self,
        settings: Settings,
        storage: ReportStorage,
        watchdog: ViberProcessWatchdog,
        notifier: TelegramNotifier,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.watchdog = watchdog
        self.notifier = notifier
        self.work_start = parse_work_time(settings.work_hours_start, dt_time(9, 0))
        self.work_end = parse_work_time(settings.work_hours_end, dt_time(18, 0))
        self.work_days = parse_work_days(settings.work_days)

    def is_within_work_hours(self, now: Optional[datetime] = None) -> bool:
        current = now or datetime.now()
        if current.isoweekday() not in self.work_days:
            return False
        cur_t = current.time()
        return self.work_start <= cur_t <= self.work_end

    def _can_emit(self, alert_key: str, now: datetime) -> bool:
        cooldown_min = max(0, self.settings.alert_cooldown_minutes)
        if cooldown_min == 0:
            return True
        last_dt = self.storage.get_last_alert_time(alert_key)
        if last_dt is None:
            return True
        return (now - last_dt) >= timedelta(minutes=cooldown_min)

    def emit_alert(
        self,
        alert_key: str,
        severity: str,
        message: str,
        *,
        now: Optional[datetime] = None,
        force: bool = False,
    ) -> bool:
        current = now or datetime.now()
        if not force and not self._can_emit(alert_key, current):
            return False

        delivered = self.notifier.send(message)
        self.storage.record_alert(
            alert_key=alert_key,
            severity=severity,
            message=message,
            delivered_telegram=delivered,
            created_at=current,
        )
        return True

    def check_viber_health(self, *, now: Optional[datetime] = None) -> bool:
        current = now or datetime.now()
        running, restarted = self.watchdog.check_and_recover()
        if restarted:
            self.emit_alert(
                "viber_restarted",
                "warning",
                (
                    f"[Ramedas] Viber.exe був закритий/упав і автоматично перезапущений "
                    f"({current.strftime('%Y-%m-%d %H:%M')}, всього перезапусків: {self.watchdog.restart_count})."
                ),
                now=current,
            )
        elif not running:
            self.emit_alert(
                "viber_down",
                "critical",
                (
                    f"[Ramedas] КРИТИЧНО: Viber.exe не запущений і не вдалося перезапустити "
                    f"({current.strftime('%Y-%m-%d %H:%M')}). Перевірте робочий ноутбук/RDP."
                ),
                now=current,
            )
        return running

    def check_silence(
        self,
        *,
        now: Optional[datetime] = None,
        viber_running: Optional[bool] = None,
    ) -> bool:
        """
        Перевіряє, чи не було аномальної тиші у ViberDownloads під час робочого вікна (09:00–18:00).
        Повертає True, якщо алерт спрацював (або був зафіксований).
        """
        current = now or datetime.now()
        threshold_min = self.settings.silence_alert_minutes
        if threshold_min <= 0 or not self.is_within_work_hours(current):
            return False

        work_start_today = datetime.combine(current.date(), self.work_start)
        last_arrival = self.storage.get_last_arrival_time()
        reference_dt = (
            max(work_start_today, last_arrival) if last_arrival else work_start_today
        )

        silent_for = current - reference_dt
        if silent_for < timedelta(minutes=threshold_min):
            return False

        is_alive = (
            viber_running
            if viber_running is not None
            else self.watchdog.is_viber_running()
        )
        last_str = (
            last_arrival.strftime("%Y-%m-%d %H:%M")
            if last_arrival
            else "даних ще немає"
        )
        silent_minutes = int(silent_for.total_seconds() // 60)
        viber_state = "running" if is_alive else "STOPPED"
        msg = (
            f"[Ramedas] Аномальна тиша у робочий час ({current.strftime('%H:%M')}): "
            f"нових фото з Viber немає вже {silent_minutes} хв "
            f"(останнє надходження: {last_str}, Viber.exe: {viber_state}). "
            f"Перевірте, чи не завис клієнт Viber або інтернет."
        )
        return self.emit_alert("work_hours_silence", "warning", msg, now=current)

    def notify_llm_failure(
        self,
        filename: str,
        error_msg: str,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        current = now or datetime.now()
        msg = (
            f"[Ramedas] Помилка розпізнавання фото {filename[:32]} "
            f"({current.strftime('%H:%M')}): {error_msg[:200]}"
        )
        return self.emit_alert("llm_failure", "error", msg, now=current)

"""
Моніторинг активності груп у робочий час (09:00–18:00), контроль працездатності Viber.exe
та відправка технічних алертів із захистом від спаму (cooldown у state.db).
"""

from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta
import html
from typing import Optional

from src.alerts import TelegramNotifier, strip_html_tags
from src.config import Settings
from src.date_utils import DOCUMENT_TYPE_LABELS
from src.logger import get_logger
from src.storage import ReportStorage
from src.viber_watchdog import ViberProcessWatchdog

log = get_logger("monitor")


def _format_ua_date(iso_date: str) -> str:
    try:
        return date.fromisoformat(iso_date).strftime("%d.%m.%Y")
    except ValueError:
        return iso_date


def parse_work_time(hhmm: str, fallback: dt_time) -> dt_time:
    try:
        parts = (hhmm or "").strip().split(":")
        return dt_time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return fallback


def parse_optional_time(hhmm: str) -> Optional[dt_time]:
    cleaned = (hhmm or "").strip().lower()
    if not cleaned or cleaned in ("0", "off", "false", "disabled", "none"):
        return None
    try:
        parts = cleaned.split(":")
        return dt_time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return None


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
            message=strip_html_tags(message),
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
                    "⚠️ <b>Viber автоматично перезапущено</b>\n"
                    "Клієнт <code>Viber.exe</code> закрився або завис - агент підняв його повторно.\n\n"
                    f"🕒 Час: <b>{current.strftime('%H:%M')}</b>\n"
                    f"🔁 Перезапусків за сесію: <b>{self.watchdog.restart_count}</b>"
                ),
                now=current,
            )
        elif not running:
            self.emit_alert(
                "viber_down",
                "critical",
                (
                    "🚨 <b>КРИТИЧНО: Viber не працює</b>\n"
                    "Не вдалося автоматично запустити <code>Viber.exe</code> - нові фото не завантажуються.\n\n"
                    f"🕒 Час: <b>{current.strftime('%H:%M')}</b>\n"
                    "👉 Перевірте робочий ПК або RDP-сесію."
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
        Перевіряє, чи не було аномальної тиші у ViberDownloads під час робочого вікна (09:00-18:00).
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
            last_arrival.strftime("%d.%m %H:%M")
            if last_arrival
            else "даних ще немає"
        )
        silent_minutes = int(silent_for.total_seconds() // 60)
        viber_state = "🟢 працює" if is_alive else "🔴 зупинений"
        msg = (
            f"⏳ <b>Аномальна тиша в робочий час</b> ({current.strftime('%H:%M')})\n"
            f"Нових фото з Viber немає вже <b>{silent_minutes} хв</b>.\n\n"
            f"• Останнє фото: <b>{last_str}</b>\n"
            f"• <code>Viber.exe</code>: {viber_state}\n"
            "👉 Перевірте інтернет або синхронізацію груп Viber."
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
        safe_name = html.escape(filename[:40], quote=False)
        safe_err = html.escape(error_msg[:180], quote=False)
        msg = (
            f"❌ <b>Помилка розпізнавання ШІ</b> ({current.strftime('%H:%M')})\n"
            "Усі моделі не змогли обробити фото.\n\n"
            f"• Файл: <code>{safe_name}</code>\n"
            f"• Причина: <code>{safe_err}</code>"
        )
        return self.emit_alert("llm_failure", "error", msg, now=current)

    @staticmethod
    def format_daily_summary(
        stats: dict,
        *,
        viber_running: Optional[bool] = None,
        pending_count: int = 0,
    ) -> str:
        """Форматує короткий HTML-звіт за день для відправки в Telegram."""
        day_human = _format_ua_date(stats["date"])
        total = stats["total"]
        lines = [f"📊 <b>Підсумок дня · {day_human}</b>", ""]

        if total == 0:
            lines.append("Сьогодні нових фото не надходило (<b>0</b>).")
            if pending_count > 0:
                lines.append(f"• У черзі на обробку: <b>{pending_count}</b>")
            if viber_running is not None:
                v_state = "🟢 працює" if viber_running else "🔴 зупинений"
                lines.extend(["", f"<code>Viber.exe</code>: {v_state}"])
            return "\n".join(lines)

        matched = stats["matched"]
        pct = round(matched * 100.0 / total, 1)
        lines.append("<b>Огляд:</b>")
        lines.append(
            f"• Отримано фото: <b>{total}</b> "
            f"(<b>{stats['unique_facilities']}</b> закладів · <b>{stats['unique_regions']}</b> громад)"
        )
        lines.append(f"• Розкладено в папки: <b>{matched}</b> (<b>{pct:.1f}%</b>)")
        lines.append(f"• На перевірку очима: <b>{stats['needs_review']}</b>")
        lines.append(f"• Без реквізитів: <b>{stats['unmatched']}</b>")
        lines.append(
            f"• Не-звіти / дублікати: <b>{stats['ignored_non_report']}</b> / <b>{stats['duplicate']}</b>"
        )
        if stats["failed"] > 0:
            lines.append(f"• Помилки розпізнавання: <b>{stats['failed']}</b>")
        if pending_count > 0:
            lines.append(f"• У черзі на обробку: <b>{pending_count}</b>")

        doc_types = stats.get("doc_types") or []
        if doc_types:
            doc_parts = [
                f"{DOCUMENT_TYPE_LABELS.get(dt, dt).replace('_', ' ')} ({cnt})"
                for dt, cnt in doc_types
            ]
            lines.extend(["", "<b>Документи:</b>", ", ".join(doc_parts)])

        top_regions = stats.get("top_regions") or []
        if top_regions:
            lines.extend(["", "<b>Топ громад:</b>"])
            for reg, cnt in top_regions[:5]:
                lines.append(f"• {html.escape(reg, quote=False)} - <b>{cnt}</b>")
            rest = len(top_regions) - 5
            if rest > 0:
                lines.append(f"<i>+ ще {rest} громад</i>")

        last_ts = stats.get("last_arrival")
        last_hhmm = last_ts[11:16] if last_ts and len(last_ts) >= 16 else "-"
        footer = f"🕒 Останнє фото: <b>{last_hhmm}</b>"
        if viber_running is not None:
            v_state = "🟢 працює" if viber_running else "🔴 зупинений"
            footer += f" | Viber: {v_state}"
        lines.extend(["", footer])
        return "\n".join(lines)

    def send_daily_summary(
        self,
        *,
        target_date: Optional[date] = None,
        now: Optional[datetime] = None,
        viber_running: Optional[bool] = None,
        pending_count: int = 0,
        manual: bool = False,
    ) -> tuple[bool, str]:
        """Формує та надсилає щоденне зведення в Telegram із записом в alert_events."""
        current = now or datetime.now()
        day = target_date or current.date()
        stats = self.storage.get_daily_summary(day)
        is_alive = (
            viber_running
            if viber_running is not None
            else self.watchdog.is_viber_running()
        )
        msg = self.format_daily_summary(
            stats,
            viber_running=is_alive,
            pending_count=pending_count,
        )
        alert_key = "manual_daily_summary" if manual else f"daily_summary:{stats['date']}"
        delivered = self.notifier.send(msg)
        self.storage.record_alert(
            alert_key=alert_key,
            severity="info",
            message=strip_html_tags(msg),
            delivered_telegram=delivered,
            created_at=current,
        )
        return delivered, msg

    def check_daily_summary(
        self,
        *,
        now: Optional[datetime] = None,
        viber_running: Optional[bool] = None,
        pending_count: int = 0,
    ) -> bool:
        """Раз на добу після настання DAILY_SUMMARY_TIME надсилає підсумок дня в Telegram."""
        summary_time = parse_optional_time(self.settings.daily_summary_time)
        if summary_time is None:
            return False

        current = now or datetime.now()
        if current.time() < summary_time:
            return False

        day_str = current.date().isoformat()
        alert_key = f"daily_summary:{day_str}"
        if self.storage.get_last_alert_time(alert_key) is not None:
            return False

        stats = self.storage.get_daily_summary(current.date())
        if stats["total"] == 0 and current.isoweekday() not in self.work_days:
            return False

        self.send_daily_summary(
            target_date=current.date(),
            now=current,
            viber_running=viber_running,
            pending_count=pending_count,
            manual=False,
        )
        return True

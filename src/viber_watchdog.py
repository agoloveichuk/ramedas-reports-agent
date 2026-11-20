"""
Watchdog для контролю процесу Viber Desktop (Viber.exe) на Windows.
Перевіряє, чи запущений клієнт Viber, і автоматично перезапускає його у фоні у разі падіння.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

from src.logger import get_logger

logger = get_logger("watchdog")


def _default_viber_exe_path() -> Path:
    local_appdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local_appdata / "Viber" / "Viber.exe"


class ViberProcessWatchdog:
    """Моніторинг та автоматичний перезапуск клієнта Viber Desktop."""

    def __init__(self, viber_exe_path: Path | None = None) -> None:
        self.viber_exe_path = viber_exe_path or _default_viber_exe_path()
        self.restart_count: int = 0
        self.last_restart_ts: float = 0.0

    def is_viber_running(self) -> bool:
        """Перевіряє наявність активного процесу Viber.exe через системний tasklist."""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Viber.exe", "/NH"],
                capture_output=True,
                text=True,
                encoding="cp866",
                errors="ignore",
                timeout=10,
                check=False,
            )
            return "viber.exe" in result.stdout.lower()
        except Exception as exc:
            logger.error("Помилка перевірки процесу Viber.exe: %s", exc)
            return False

    def ensure_running(self) -> bool:
        """
        Якщо Viber.exe запущений — повертає True.
        Якщо процес відсутній — запускає Viber.exe у фоні й логує подію.
        """
        running, _ = self.check_and_recover()
        return running

    def check_and_recover(self) -> tuple[bool, bool]:
        """
        Перевіряє, чи запущений Viber.exe, і при падінні намагається його перезапустити.
        Повертає кортеж (is_running, was_restarted).
        """
        if self.is_viber_running():
            return True, False

        logger.warning("Процес Viber.exe НЕ знайдено! Ініціюємо автоматичний запуск...")
        if not self.viber_exe_path.exists():
            logger.error(
                "Не вдалося запустити Viber: виконуваний файл не знайдено за шляхом %s",
                self.viber_exe_path,
            )
            return False, False

        # Захист від занадто частих перезапусків (не частіше 1 разу на 60 секунд)
        now = time.time()
        if now - self.last_restart_ts < 60.0:
            logger.warning("Пропуск перезапуску Viber.exe: минуло менше 60с після попередньої спроби.")
            return False, False

        try:
            subprocess.Popen(
                [str(self.viber_exe_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0x00000008),
            )
            self.restart_count += 1
            self.last_restart_ts = now
            time.sleep(3.0)
            running = self.is_viber_running()
            if running:
                logger.info(
                    "Viber.exe успішно перезапущено (всього автоперезапусків: %d).",
                    self.restart_count,
                )
            else:
                logger.error("Команду запуску Viber.exe виконано, але процес не з'явився.")
            return running, running
        except Exception as exc:
            logger.error("Критична помилка при спробі запустити Viber.exe: %s", exc)
            return False, False

"""
Централізоване налаштування логування агента.
Записує читабельні логи одночасно в консоль та у ротований файл `logs/agent.log` (UTF-8).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

from src.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "agent.log"

_INITIALIZED = False


def get_logger(name: str = "ramedas") -> logging.Logger:
    """Повертає налаштований логер із виводом у консоль та файл logs/agent.log."""
    global _INITIALIZED
    root_logger = logging.getLogger("ramedas")

    if not _INITIALIZED:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Ротований файловий лог (до 5 файлів по 5 МБ кожен)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Консольний вивід (UTF-8)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        _INITIALIZED = True

    if name == "ramedas" or name.startswith("ramedas."):
        return logging.getLogger(name)
    return logging.getLogger(f"ramedas.{name}")

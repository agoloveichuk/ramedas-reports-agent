from __future__ import annotations

import argparse
import sys
import time

from src.config import Settings
from src.connectors.viber_local import group_into_burst_batches
from src.logger import LOG_FILE, get_logger
from src.pipeline import ReportPipeline

log = get_logger("main")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ramedas Reports Agent")
    parser.add_argument("--status", action="store_true", help="Показати поточний стан і чергу")
    parser.add_argument("--once", action="store_true", help="Обробити поточну чергу фото")
    parser.add_argument("--limit", type=int, default=0, help="Ліміт фото за один прогін")
    parser.add_argument("--dry-run", action="store_true", help="Тестовий прогін без запису")
    parser.add_argument("--watch", action="store_true", help="Фоновий моніторинг")
    args = parser.parse_args()

    settings = Settings()
    pipeline = ReportPipeline(settings)
    viber_alive = pipeline.watchdog.ensure_running()

    if args.status or not (args.once or args.watch):
        pending = pipeline.viber.fetch_pending_items()
        batches = group_into_burst_batches(pending, max_gap_seconds=6)
        stats = pipeline.storage.get_unfinished_stats()
        chain_str = " -> ".join(
            f"{p.name}[{p.model}]" for p in pipeline.vision.active_providers
        )
        print("Ramedas Reports Agent - статус:")
        print(f"  viber_process    : {'running' if viber_alive else 'stopped'}")
        print(f"  llm_chain        : {chain_str or 'none'}")
        print(f"  routing_config   : {pipeline.vision.routing_path}")
        print(f"  review_threshold : {settings.confidence_review_threshold}")
        print(
            f"  catalog          : {len(pipeline.catalog.facilities)} закладів "
            f"({len(pipeline.catalog._by_phone)} тел., {len(pipeline.catalog._by_surname)} прізвищ, "
            f"автовивчено для {len(pipeline.catalog._learned_by_facility)} закладів)"
        )
        print(f"  viber_downloads  : {settings.viber_downloads_dir}")
        print(f"  pending_queue    : {len(pending)} фото ({len(batches)} пакетів)")
        print(f"  db_stats         : {stats}")
        print(f"  storage_root     : {settings.storage_root}")
        print(f"  log_file         : {LOG_FILE}")
        if not (args.once or args.watch):
            return 0

    if args.once:
        pending = pipeline.viber.fetch_pending_items()
        if args.limit > 0:
            pending = pending[: args.limit]
        batches = group_into_burst_batches(pending, max_gap_seconds=6)
        log.info(
            "Старт обробки черги: %d фото (%d пакетів), dry_run=%s",
            len(pending),
            len(batches),
            args.dry_run,
        )
        for idx, batch in enumerate(batches, 1):
            log.info("Пакет %d/%d (%d фото)", idx, len(batches), len(batch))
            pipeline.process_batch(batch, dry_run=args.dry_run)
        log.info("Обробку завершено.")
        return 0

    if args.watch:
        log.info(
            "Запущено моніторинг %s (batch_window=%ds)",
            settings.viber_downloads_dir,
            settings.batch_window_seconds,
        )
        last_watchdog_ts = time.time()
        try:
            while True:
                now_ts = time.time()
                if now_ts - last_watchdog_ts >= 30.0:
                    pipeline.watchdog.ensure_running()
                    last_watchdog_ts = now_ts

                pending = pipeline.viber.fetch_pending_items()
                if pending:
                    newest_age = now_ts - pending[-1].created_at.timestamp()
                    if newest_age >= settings.batch_window_seconds:
                        batches = group_into_burst_batches(pending, max_gap_seconds=6)
                        log.info("У черзі %d фото (%d пакетів)", len(pending), len(batches))
                        for idx, batch in enumerate(batches, 1):
                            log.info("Пакет %d/%d (%d фото)", idx, len(batches), len(batch))
                            pipeline.process_batch(batch, dry_run=args.dry_run)
                time.sleep(5)
        except KeyboardInterrupt:
            log.info("Зупинено користувачем.")
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
import time

from src.alerts import strip_html_tags
from src.config import Settings
from src.llm.router import _sanitize_error_text
from src.logger import LOG_FILE, get_logger
from src.models import IncomingMediaItem
from src.pipeline import ReportPipeline

log = get_logger("main")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ramedas Reports Agent")
    parser.add_argument("--status", action="store_true", help="Показати поточний стан і чергу")
    parser.add_argument("--once", action="store_true", help="Обробити поточну чергу фото")
    parser.add_argument(
        "--all-history",
        action="store_true",
        help="Повний прогін по всій історії груп (без 72-год ліміту Telegram, розширене вікно дат 365 дн. і без видалення файлів)",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Не виконувати автоматичну ротацію ViberDownloads після --once",
    )
    parser.add_argument("--limit", type=int, default=0, help="Ліміт фото за один прогін")
    parser.add_argument("--dry-run", action="store_true", help="Тестовий прогін без запису")
    parser.add_argument("--watch", action="store_true", help="Фоновий моніторинг")
    parser.add_argument(
        "--rotate-old",
        action="store_true",
        help="Очистити або заархівувати старі оброблені файли з ViberDownloads",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Поріг давності файлів у днях для ротації ViberDownloads (перевизначає VIBER_RETENTION_DAYS)",
    )
    parser.add_argument(
        "--activity-stats",
        action="store_true",
        help="Показати статистику активності по годинах доби та громадах із state.db",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Кількість днів для --activity-stats (за замовчуванням 14)",
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        help="Надіслати тестовий технічний алерт (у Telegram або в локальний лог, якщо бот не підключений)",
    )
    parser.add_argument(
        "--daily-summary",
        action="store_true",
        help="Сформувати і надіслати в Telegram підсумкову статистику дня (із --dry-run лише вивід у консоль)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Дата YYYY-MM-DD для --daily-summary (за замовчуванням сьогодні)",
    )
    parser.add_argument(
        "--telegram-login",
        action="store_true",
        help="Одноразова авторизація робочого акаунта Telegram (User API / Telethon)",
    )
    parser.add_argument(
        "--telegram-chats",
        action="store_true",
        help="Показати список груп Telegram та їхні ID (для TELEGRAM_ALLOWED_CHATS)",
    )
    parser.add_argument(
        "--test-models",
        action="store_true",
        help="Порівняльний read-only тест усіх активних LLM-провайдерів із .env на реальних фото (без запису в БД)",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Шлях до конкретного фото для --test-models",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Кількість тестових фото для --test-models (за замовчуванням 3)",
    )
    return parser


def _print_activity_stats(pipeline: ReportPipeline, settings: Settings, days: int) -> None:
    act = pipeline.storage.get_activity_stats(
        days=days,
        work_start_hour=pipeline.monitor.work_start.hour,
        work_end_hour=pipeline.monitor.work_end.hour,
    )
    print(f"Статистика активності за останні {act['days']} дн.:")
    print(f"  total_files      : {act['total_count']}")
    print(
        f"  work_hours       : {act['work_hours_count']} "
        f"({settings.work_hours_start}-{settings.work_hours_end})"
    )
    print(f"  today_files      : {act['today_count']}")
    print(f"  last_arrival     : {act['last_arrival'] or 'none'}")
    print("  by_hour (00..23) :")
    for hour, cnt in act["hourly_counts"].items():
        if cnt > 0:
            print(f"    {hour:02d}:00 | {cnt:4d} {'#' * min(30, cnt)}")
    if act["by_region"]:
        print("  by_region        :")
        for reg in act["by_region"]:
            print(
                f"    - {reg['region']}: {reg['count']} фото "
                f"(пік {reg['peak_hour']:02d}:00, останній {reg['last_arrival']})"
            )
    recent_alerts = pipeline.storage.get_recent_alerts(limit=5)
    if recent_alerts:
        print("  recent_alerts    :")
        for al in recent_alerts:
            tg_flag = "tg" if al["delivered_telegram"] else "local"
            print(
                f"    [{al['created_at']}] ({al['severity']}/{tg_flag}) "
                f"{al['alert_key']}: {al['message'][:80]}"
            )


def _print_status(pipeline: ReportPipeline, settings: Settings, *, viber_alive: bool) -> None:
    pending, batches = pipeline.fetch_pending_batches()
    stats = pipeline.storage.get_unfinished_stats()
    last_arrival = pipeline.storage.get_last_arrival_time()
    chain_str = " -> ".join(f"{p.name}[{p.model}]" for p in pipeline.vision.active_providers)
    esc_str = " -> ".join(f"{p.name}[{p.model}]" for p in pipeline.vision.escalation_providers)
    retention_mode = (
        f"archive -> {settings.viber_archive_dir}" if settings.viber_archive_dir else "delete"
    )
    retention_str = (
        f"{settings.viber_retention_days} дн. ({retention_mode})"
        if settings.viber_retention_days > 0
        else "disabled"
    )
    tg_status = (
        "configured"
        if pipeline.notifier.is_configured
        else "local-only (set TELEGRAM_BOT_TOKEN & TECH_ALERT_CHAT_ID in .env)"
    )

    print("Ramedas Reports Agent - статус:")
    print(f"  viber_process    : {'running' if viber_alive else 'stopped'}")
    print(f"  telegram_ingest  : {pipeline.telegram.status_label}")
    print(f"  telegram_alerts  : {tg_status}")
    print(
        f"  work_hours       : {settings.work_hours_start}-{settings.work_hours_end} "
        f"(тиша > {settings.silence_alert_minutes} хв, cooldown {settings.alert_cooldown_minutes} хв)"
    )
    print(f"  daily_summary    : {settings.daily_summary_time or 'disabled'}")
    print(
        "  last_arrival     : "
        f"{last_arrival.strftime('%Y-%m-%d %H:%M:%S') if last_arrival else 'none'}"
    )
    print(f"  llm_chain        : {chain_str or 'none'}")
    print(f"  escalation_chain : {esc_str or 'disabled (no API key)'}")
    print(f"  routing_config   : {pipeline.vision.routing_path}")
    print(f"  review_threshold : {settings.confidence_review_threshold}")
    print(
        f"  catalog          : {len(pipeline.catalog.facilities)} закладів "
        f"({len(pipeline.catalog._by_phone)} тел., {len(pipeline.catalog._by_surname)} прізвищ, "
        f"автовивчено для {len(pipeline.catalog._learned_by_facility)} закладів)"
    )
    print(f"  viber_downloads  : {settings.viber_downloads_dir}")
    print(f"  viber_retention  : {retention_str}")
    print(f"  pending_queue    : {len(pending)} фото ({len(batches)} пакетів)")
    print(f"  db_stats         : {stats}")
    print(f"  storage_root     : {settings.storage_root}")
    print(f"  reports_output   : {pipeline.storage.reports_dir}")
    print(f"  log_file         : {LOG_FILE}")


def _process_batches(
    pipeline: ReportPipeline,
    batches: list[list[IncomingMediaItem]],
    *,
    dry_run: bool,
) -> None:
    for idx, batch in enumerate(batches, 1):
        log.info("Пакет %d/%d (%d фото)", idx, len(batches), len(batch))
        pipeline.process_batch(batch, dry_run=dry_run)


def _run_once(pipeline: ReportPipeline, args: argparse.Namespace) -> None:
    pending, batches = pipeline.fetch_pending_batches(limit=args.limit)
    log.info(
        "Старт обробки черги: %d фото (%d пакетів), dry_run=%s, all_history=%s",
        len(pending),
        len(batches),
        args.dry_run,
        args.all_history,
    )
    _process_batches(pipeline, batches, dry_run=args.dry_run)
    should_rotate = (
        not args.no_rotate
        and (not args.all_history or args.rotate_old)
        and (args.limit == 0 or args.rotate_old)
    )
    if should_rotate:
        pipeline.rotate_viber_downloads(
            retention_days=args.retention_days,
            dry_run=args.dry_run,
        )
    log.info("Обробку завершено.")


def _run_watch(
    pipeline: ReportPipeline,
    settings: Settings,
    args: argparse.Namespace,
    *,
    viber_alive: bool,
) -> None:
    log.info(
        "Запущено моніторинг %s (telegram=%s, batch_window=%ds, retention=%dd)",
        settings.viber_downloads_dir,
        pipeline.telegram.status_label,
        settings.batch_window_seconds,
        settings.viber_retention_days,
    )
    last_watchdog_ts = time.time()
    last_silence_ts = 0.0
    last_summary_ts = 0.0
    last_rotate_ts = 0.0
    last_tg_poll_ts = 0.0
    try:
        while True:
            now_ts = time.time()
            if now_ts - last_watchdog_ts >= 30.0:
                viber_alive = pipeline.monitor.check_viber_health()
                last_watchdog_ts = now_ts

            if now_ts - last_silence_ts >= 60.0:
                pipeline.monitor.check_silence(viber_running=viber_alive)
                last_silence_ts = now_ts

            if now_ts - last_rotate_ts >= 21600.0:
                pipeline.rotate_viber_downloads(
                    retention_days=args.retention_days,
                    dry_run=args.dry_run,
                )
                last_rotate_ts = now_ts

            poll_tg = (now_ts - last_tg_poll_ts) >= 30.0
            if poll_tg:
                last_tg_poll_ts = now_ts

            pending, batches = pipeline.fetch_pending_batches(include_telegram=poll_tg)
            if pending:
                newest_age = now_ts - pending[-1].created_at.timestamp()
                if newest_age >= settings.batch_window_seconds:
                    log.info("У черзі %d фото (%d пакетів)", len(pending), len(batches))
                    _process_batches(pipeline, batches, dry_run=args.dry_run)
                    pending = []

            if now_ts - last_summary_ts >= 60.0:
                pipeline.monitor.check_daily_summary(
                    viber_running=viber_alive,
                    pending_count=len(pending),
                )
                last_summary_ts = now_ts
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("Зупинено користувачем.")


def _pick_sample_images(pipeline: ReportPipeline, settings: Settings, max_samples: int) -> list[Path]:
    picked: list[Path] = []
    seen: set[Path] = set()
    reports_dir = pipeline.storage.reports_dir
    if reports_dir.exists():
        all_imgs = [
            p for p in reports_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]
        # Намагаємося взяти по одному типовому документу кожного класу (ТТН, Щоденний звіт, Розрахунок сировини)
        for token in ("_ТТН_", "_Щоденний_звіт_", "_Розрахунок_сировини_", "_Акт_", "_Меню_"):
            for p in all_imgs:
                if token in p.name and p not in seen:
                    picked.append(p)
                    seen.add(p)
                    break
            if len(picked) >= max_samples:
                return picked
        for p in all_imgs:
            if p not in seen:
                picked.append(p)
                seen.add(p)
            if len(picked) >= max_samples:
                return picked

    if len(picked) < max_samples and settings.viber_downloads_dir.exists():
        for p in settings.viber_downloads_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p not in seen:
                picked.append(p)
                seen.add(p)
            if len(picked) >= max_samples:
                break
    return picked


def _run_test_models(
    pipeline: ReportPipeline,
    settings: Settings,
    *,
    sample_path: str | None,
    max_samples: int,
) -> int:
    providers = []
    seen_keys: set[tuple[str, str]] = set()
    for prov in list(pipeline.vision.active_providers) + list(pipeline.vision.escalation_providers):
        key = (prov.provider_type, prov.model)
        if key not in seen_keys:
            seen_keys.add(key)
            providers.append(prov)

    if not providers:
        print("Не знайдено жодного активного LLM-маршруту з ключем у .env.")
        return 1

    if sample_path:
        p = Path(sample_path)
        if not p.exists():
            print(f"Файл не знайдено: {p}")
            return 2
        samples = [p]
    else:
        samples = _pick_sample_images(pipeline, settings, max(1, max_samples))

    if not samples:
        print("Не знайдено жодного фото для тестування (передайте --sample PATH).")
        return 2

    print(f"Тестування {len(providers)} активних моделей на {len(samples)} фото (read-only, без запису в БД):")
    for prov in providers:
        print(f"  - {prov.name} [{prov.provider_type}:{prov.model}]")

    for idx, img_path in enumerate(samples, 1):
        folder_hint = img_path.parent.parent.parent.name if len(img_path.parents) >= 3 else img_path.parent.name
        print(f"\n[{idx}/{len(samples)}] {img_path.name} (папка: {folder_hint})")
        for prov in providers:
            t0 = time.monotonic()
            try:
                vis = prov.analyze(img_path, pipeline.vision.system_prompt)
                pipeline._normalize_report_flags(vis)
                match = pipeline.catalog.match(vis) if vis.is_report else None
                ms = int((time.monotonic() - t0) * 1000)
                if not vis.is_report:
                    res_str = f"NON_REPORT ({vis.document_type})"
                elif match:
                    flag = " [REVIEW]" if match.needs_review else ""
                    res_str = f"{match.facility_id} score={match.score:.2f}{flag} ({match.canonical_name[:30]})"
                else:
                    res_str = f"UNMATCHED (hint={vis.facility_hint!r})"
                print(
                    f"  OK  | {prov.name:27s} | {ms:5d}ms | leg={vis.legibility:6s} "
                    f"conf={vis.confidence:.2f} rot={vis.rotation_degrees:3d} | {res_str}",
                    flush=True,
                )
            except Exception as exc:
                ms = int((time.monotonic() - t0) * 1000)
                safe_err = _sanitize_error_text(str(exc)).replace("\n", " ")[:120]
                print(
                    f"  ERR | {prov.name:27s} | {ms:5d}ms | {exc.__class__.__name__}: {safe_err}",
                    flush=True,
                )
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args()
    settings = Settings()
    if args.all_history:
        settings.telegram_lookback_hours = 0
        settings.document_date_max_age_days = max(settings.document_date_max_age_days, 365)
    pipeline = ReportPipeline(settings)

    if args.telegram_login:
        pipeline.telegram.interactive_login()
        return 0

    if args.telegram_chats:
        dialogs = pipeline.telegram.list_dialogs()
        print(f"Доступні групові чати/канали Telegram ({len(dialogs)}):")
        for dlg in dialogs:
            mark = "active" if dlg["active"] else "filtered_out"
            kind = "group" if dlg["is_group"] else "channel"
            print(f"  {dlg['id']} | {kind:7s} | {mark:12s} | {dlg['title']}")
        return 0

    if args.activity_stats:
        _print_activity_stats(pipeline, settings, args.days)
        return 0

    if args.test_models:
        return _run_test_models(
            pipeline,
            settings,
            sample_path=args.sample,
            max_samples=args.samples,
        )

    if args.test_alert:
        tg_state = "підключений" if pipeline.notifier.is_configured else "ще не налаштований, запис локально"
        pipeline.monitor.emit_alert(
            "manual_test",
            "info",
            (
                "✅ <b>Ramedas Alerts на зв'язку</b>\n"
                f"Тестове технічне сповіщення (Telegram: <b>{tg_state}</b>)."
            ),
            force=True,
        )
        return 0

    if args.daily_summary:
        try:
            target_date = date.fromisoformat(args.date) if args.date else date.today()
        except ValueError:
            log.error("Некоректний формат --date %r (очікується YYYY-MM-DD)", args.date)
            return 2
        viber_alive = pipeline.watchdog.is_viber_running()
        pending, _ = pipeline.fetch_pending_batches(include_telegram=False)
        if args.dry_run:
            stats = pipeline.storage.get_daily_summary(target_date)
            msg = pipeline.monitor.format_daily_summary(
                stats,
                viber_running=viber_alive,
                pending_count=len(pending),
            )
        else:
            _, msg = pipeline.monitor.send_daily_summary(
                target_date=target_date,
                viber_running=viber_alive,
                pending_count=len(pending),
                manual=True,
            )
        print(strip_html_tags(msg))
        return 0

    viber_alive = pipeline.watchdog.ensure_running()
    run_queue = args.once or args.all_history
    has_action = run_queue or args.watch or args.rotate_old

    if args.status or not has_action:
        _print_status(pipeline, settings, viber_alive=viber_alive)
        if not has_action:
            return 0

    if args.rotate_old and not (run_queue or args.watch):
        rot_stats = pipeline.rotate_viber_downloads(
            retention_days=args.retention_days,
            dry_run=args.dry_run,
        )
        if rot_stats.rotated == 0 and rot_stats.skipped_unprocessed == 0:
            log.info("Старих оброблених файлів у ViberDownloads для ротації не знайдено.")
        return 0

    if run_queue:
        _run_once(pipeline, args)
    elif args.watch:
        _run_watch(pipeline, settings, args, viber_alive=viber_alive)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

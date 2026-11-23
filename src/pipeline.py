from __future__ import annotations

import time
from typing import Optional

from src.alerts import TelegramNotifier
from src.catalog import FacilityCatalog
from src.config import Settings
from src.connectors.telegram_user import TelegramUserConnector
from src.connectors.viber_local import (
    ViberLocalConnector,
    ViberRotationStats,
    group_into_burst_batches,
    rotate_directory_old_files,
)
from src.date_utils import validate_document_date
from src.image_utils import normalize_rotation_degrees
from src.logger import get_logger
from src.models import FacilityMatch, IncomingMediaItem, VisionAnalysisResult
from src.monitoring import ActivityMonitor
from src.storage import ReportStorage
from src.viber_watchdog import ViberProcessWatchdog
from src.vision import VisionModelRouter

log = get_logger("pipeline")


class ReportPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = FacilityCatalog(
            settings.facilities_json_path,
            review_threshold=settings.confidence_review_threshold,
        )
        self.storage = ReportStorage(
            settings.storage_root,
            reports_output_dir=settings.reports_output_dir,
            document_date_max_age_days=settings.document_date_max_age_days,
        )
        self.storage.backfill_learned_contacts_from_history()
        self.catalog.update_learned_contacts(self.storage.get_learned_contacts_map())
        self.vision = VisionModelRouter(settings)
        self.watchdog = ViberProcessWatchdog()
        self.notifier = TelegramNotifier(settings)
        self.monitor = ActivityMonitor(
            settings,
            self.storage,
            self.watchdog,
            self.notifier,
        )
        self.viber = ViberLocalConnector(
            downloads_dir=settings.viber_downloads_dir,
            is_processed_fn=self.storage.is_completed,
        )
        self.telegram = TelegramUserConnector(
            api_id=settings.telegram_api_id,
            api_hash=settings.telegram_api_hash,
            session_path=settings.telegram_session_path,
            downloads_dir=settings.telegram_downloads_dir,
            is_processed_fn=self.storage.is_completed,
            lookback_hours=settings.telegram_lookback_hours,
            allowed_chats=settings.telegram_allowed_chats,
        )

    def fetch_pending_batches(
        self,
        *,
        include_telegram: bool = True,
        limit: int = 0,
    ) -> tuple[list[IncomingMediaItem], list[list[IncomingMediaItem]]]:
        """Збирає чергу нових фото з Viber (і Telegram, якщо підключений) та групує в пакети окремо по каналах."""
        viber_items = self.viber.fetch_pending_items()
        tg_items = (
            self.telegram.fetch_pending_items()
            if (include_telegram and self.telegram.is_configured)
            else []
        )

        combined = sorted(viber_items + tg_items, key=lambda x: x.created_at)
        if limit > 0:
            combined = combined[:limit]

        viber_slice = [item for item in combined if item.channel == "viber"]
        tg_slice = [item for item in combined if item.channel == "telegram"]

        batches = group_into_burst_batches(viber_slice, max_gap_seconds=6) + group_into_burst_batches(
            tg_slice, max_gap_seconds=6
        )
        batches.sort(key=lambda b: b[0].created_at)
        return combined, batches

    def rotate_viber_downloads(
        self,
        *,
        retention_days: Optional[int] = None,
        dry_run: bool = False,
    ) -> ViberRotationStats:
        days = (
            self.settings.viber_retention_days
            if retention_days is None
            else retention_days
        )
        if days <= 0:
            return ViberRotationStats()

        final_ids = self.storage.get_final_status_media_ids()
        stats = self.viber.rotate_old_files(
            retention_days=days,
            final_media_ids=final_ids,
            archive_dir=self.settings.viber_archive_dir,
            dry_run=dry_run,
        )
        if self.settings.telegram_downloads_dir.exists():
            tg_stats = rotate_directory_old_files(
                self.settings.telegram_downloads_dir,
                retention_days=days,
                final_media_ids=final_ids,
                archive_dir=self.settings.viber_archive_dir,
                dry_run=dry_run,
            )
            stats.merge(tg_stats)
        if stats.rotated or stats.skipped_unprocessed or stats.errors:
            action = "архів" if self.settings.viber_archive_dir else "видалення"
            prefix = "[DRY-RUN] " if dry_run else ""
            freed_mb = stats.freed_bytes / (1024 * 1024)
            log.info(
                "%s[ROTATE] Downloads (>%d дн., %s): оброблено=%d (%.2f МБ), пропущено_незавершених=%d, помилок=%d",
                prefix,
                days,
                action,
                stats.rotated,
                freed_mb,
                stats.skipped_unprocessed,
                stats.errors,
            )
        return stats

    def _resolve_facility_match(
        self,
        item: IncomingMediaItem,
        vis: VisionAnalysisResult,
    ) -> Optional[FacilityMatch]:
        if not vis.is_report:
            return None
        match = self.catalog.match(
            vis,
            sender_phone=item.sender_phone,
            sender_name=item.sender_name,
            chat_title=item.chat_title,
            caption=item.caption,
        )
        if not vis.stamp_number:
            return match

        stamp_hit = self.storage.lookup_stamp(vis.stamp_number)
        if not stamp_hit:
            return match
        if match is None:
            return stamp_hit
        if match.needs_review and stamp_hit.facility_id == match.facility_id:
            return FacilityMatch(
                facility_id=match.facility_id,
                canonical_name=match.canonical_name,
                region=match.region,
                score=max(match.score, stamp_hit.score),
                legibility=match.legibility,
                needs_review=False,
                match_reason=f"{match.match_reason}+{stamp_hit.match_reason}",
            )
        return match

    def _is_duplicate(
        self,
        item: IncomingMediaItem,
        seen_hashes_in_batch: dict[str, str],
        *,
        dry_run: bool,
    ) -> bool:
        if not item.file_sha256:
            return False

        existing_dup = self.storage.find_duplicate_by_sha256(item.file_sha256, item.media_id)
        if existing_dup:
            orig_id, orig_path = existing_dup
            log.info(
                "[DUPLICATE] %s збігається за SHA-256 з %s (%s)",
                item.file_path.name[:32],
                orig_id[:24],
                orig_path,
            )
            if not dry_run:
                self.storage.record_duplicate(item, orig_id)
            return True

        if item.file_sha256 in seen_hashes_in_batch:
            orig_id = seen_hashes_in_batch[item.file_sha256]
            log.info(
                "[DUPLICATE] %s дублює файл %s у поточному пакеті",
                item.file_path.name[:32],
                orig_id[:24],
            )
            if not dry_run:
                self.storage.record_duplicate(item, orig_id)
            return True

        seen_hashes_in_batch[item.file_sha256] = item.media_id
        return False

    _CONSOLIDATED_MULTI_FACILITY_MARKERS = (
        "зведена таблиця",
        "розподілу продуктів по закладах",
        "маршрутний лист",
        "графік розвозки",
        "графік водіїв",
    )

    @classmethod
    def _normalize_report_flags(cls, vis: VisionAnalysisResult) -> None:
        if not vis.matched_facility_id:
            combined_text = f"{vis.facility_hint or ''} {vis.summary or ''}".lower()
            if any(marker in combined_text for marker in cls._CONSOLIDATED_MULTI_FACILITY_MARKERS):
                vis.is_report = False
                return

        if vis.is_report:
            return
        always_report_types = (
            "act",
            "daily_report",
            "menu",
            "application",
            "raw_materials_calc",
            "waybill_internal",
            "receipt",
        )
        if vis.document_type in always_report_types or (
            vis.document_type == "waybill_external"
            and bool(vis.matched_facility_id or vis.facility_hint)
        ):
            vis.is_report = True

    @staticmethod
    def _is_better_match(
        current: Optional[FacilityMatch],
        candidate: Optional[FacilityMatch],
    ) -> bool:
        if candidate is None:
            return False
        if current is None:
            return True
        if current.needs_review and not candidate.needs_review:
            return True
        if not current.needs_review and candidate.needs_review:
            return False
        if candidate.score > current.score:
            return True
        if candidate.score == current.score:
            return candidate.legibility != "poor" or current.legibility == "poor"
        return False

    def _analyze_with_upright_retry(
        self, item: IncomingMediaItem
    ) -> tuple[VisionAnalysisResult, Optional[FacilityMatch]]:
        vis = self.vision.analyze_image(
            item.file_path,
            caption=item.caption,
            reference_date=item.created_at,
        )
        self._normalize_report_flags(vis)
        detected_rotation = vis.rotation_degrees
        match = self._resolve_facility_match(item, vis)

        needs_upright_retry = (
            detected_rotation != 0
            and vis.is_report
            and vis.document_type != "food_photo"
            and (
                detected_rotation in (90, 270)
                or match is None
                or match.needs_review
                or vis.legibility != "clear"
                or vis.confidence < 0.90
            )
        )
        if needs_upright_retry:
            try:
                vis_upright = self.vision.analyze_image(
                    item.file_path,
                    caption=item.caption,
                    rotation_degrees=detected_rotation,
                    reference_date=item.created_at,
                )
                self._normalize_report_flags(vis_upright)
                # Додаємо залишковий кут після перевороту (наприклад, 270° + 180° -> 90° для горизонтальних ТТН)
                combined_rotation = normalize_rotation_degrees(
                    detected_rotation + vis_upright.rotation_degrees
                )
                vis_upright.rotation_degrees = combined_rotation
                vis.rotation_degrees = combined_rotation
                match_upright = self._resolve_facility_match(item, vis_upright)
                if self._is_better_match(match, match_upright):
                    vis, match = vis_upright, match_upright
            except Exception as exc:
                log.warning(
                    "Повторний аналіз повернутого на %d° фото %s не вдався: %s",
                    detected_rotation,
                    item.file_path.name[:24],
                    exc,
                )

        # Tier-2 ескалація на сильнішу Vision-модель лише для складних документів
        # і тільки якщо в .env наявний API-ключ для хоча б одного іншого маршруту з escalation_routes
        tier1_model = getattr(self.vision, "last_used_model", "")
        exclude_models = {tier1_model} if tier1_model else set()
        needs_escalation = (
            self.vision.has_escalation_for(exclude_models)
            and vis.is_report
            and vis.document_type != "food_photo"
            and (match is None or match.needs_review)
        )
        if needs_escalation:
            tier1_route = self.vision.last_used_route
            try:
                esc_kwargs: dict = {
                    "caption": item.caption,
                    "rotation_degrees": vis.rotation_degrees,
                    "reference_date": item.created_at,
                    "escalate": True,
                }
                if exclude_models:
                    esc_kwargs["exclude_models"] = exclude_models
                vis_esc = self.vision.analyze_image(item.file_path, **esc_kwargs)
                self._normalize_report_flags(vis_esc)
                vis_esc.rotation_degrees = normalize_rotation_degrees(
                    vis.rotation_degrees + vis_esc.rotation_degrees
                )
                if not vis_esc.is_report and match is None:
                    log.info(
                        "[ESCALATION] %s (%s): перекласифіковано як non-report (%s)",
                        item.file_path.name[:24],
                        self.vision.last_used_route,
                        vis_esc.summary[:60],
                    )
                    return vis_esc, None

                match_esc = self._resolve_facility_match(item, vis_esc)
                if self._is_better_match(match, match_esc):
                    log.info(
                        "[ESCALATION] %s (%s): %s -> %s (score=%.2f, review=%s)",
                        item.file_path.name[:24],
                        self.vision.last_used_route,
                        match.facility_id if match else "unmatched",
                        match_esc.facility_id if match_esc else "unmatched",
                        match_esc.score if match_esc else 0.0,
                        match_esc.needs_review if match_esc else None,
                    )
                    return vis_esc, match_esc
                self.vision.last_used_route = tier1_route
            except Exception as exc:
                self.vision.last_used_route = tier1_route
                log.warning(
                    "Ескалація фото %s на сильнішу модель не вдалася (залишаємо Tier-1): %s",
                    item.file_path.name[:24],
                    exc,
                )

        return vis, match

    def _validate_item_date(self, item: IncomingMediaItem, vis: VisionAnalysisResult) -> None:
        raw_date = vis.document_date
        validated_date, date_reason = validate_document_date(
            raw_date,
            item.created_at,
            max_past_days=self.settings.document_date_max_age_days,
        )
        vis.raw_document_date = raw_date
        vis.document_date = validated_date
        if raw_date and validated_date != raw_date:
            log.info(
                "[DATE-FIX] %s: %s -> %s (%s)",
                item.file_path.name[:24],
                raw_date,
                validated_date,
                date_reason,
            )

    @staticmethod
    def _log_item_result(
        item: IncomingMediaItem,
        vis: VisionAnalysisResult,
        match: Optional[FacilityMatch],
        used_model: str,
        elapsed_ms: int,
    ) -> None:
        rot_info = f", rot={vis.rotation_degrees}°" if vis.rotation_degrees else ""
        if not vis.is_report:
            log.info(
                "%s (%dms, %s) -> type=%s%s -> IGNORED (non-report: %s)",
                item.file_path.name[:24],
                elapsed_ms,
                used_model,
                vis.document_type,
                rot_info,
                vis.summary[:55],
            )
        elif match:
            flag = " [NEEDS_REVIEW]" if match.needs_review else ""
            log.info(
                "%s (%dms, %s) -> type=%s, legibility=%s, ai_conf=%.2f, score=%.2f%s%s -> %s (%s)",
                item.file_path.name[:24],
                elapsed_ms,
                used_model,
                vis.document_type,
                vis.legibility,
                vis.confidence,
                match.score,
                rot_info,
                flag,
                match.facility_id,
                match.canonical_name[:35],
            )
        else:
            log.warning(
                "%s (%dms, %s) -> type=%s, legibility=%s%s, hint=%r -> UNMATCHED",
                item.file_path.name[:24],
                elapsed_ms,
                used_model,
                vis.document_type,
                vis.legibility,
                rot_info,
                vis.facility_hint or vis.settlement_or_address_hint,
            )

    def _apply_album_consensus(
        self,
        analyzed: list[
            tuple[IncomingMediaItem, VisionAnalysisResult, Optional[FacilityMatch], str, int]
        ],
        *,
        dry_run: bool,
    ) -> None:
        """
        Узгоджує схожі за префіксом назви закладів всередині одного одночасного альбому (<= 2.5с),
        щоб одна нерозбірлива сторінка не відлітала в іншу школу зі схожим початком назви.
        """
        album_span = abs((analyzed[-1][0].created_at - analyzed[0][0].created_at).total_seconds())
        known_matches = {
            m.facility_id: m for _, v, m, _, _ in analyzed if v.is_report and m is not None
        }
        if len(analyzed) < 3 or album_span > 2.5 or len(known_matches) <= 1:
            return

        counts: dict[str, int] = {}
        contact_backed: set[str] = set()
        for _, v, m, _, _ in analyzed:
            if v.is_report and m is not None:
                counts[m.facility_id] = counts.get(m.facility_id, 0) + 1
                if "contact:" in m.match_reason:
                    contact_backed.add(m.facility_id)

        best_fid = max(
            counts,
            key=lambda fid: (counts[fid], 1 if fid in contact_backed else 0),
        )
        if counts[best_fid] < len(analyzed) * 0.6 or best_fid not in contact_backed:
            return

        consensus = known_matches[best_fid]
        consensus_prefix = consensus.canonical_name.lower().replace('"', "").split()[0][:5]
        for idx, (item, vis, match, used_m, lat_ms) in enumerate(analyzed):
            if not vis.is_report or match is None or match.facility_id == best_fid:
                continue
            cand_prefix = (
                (vis.facility_hint or "").lower().replace('"', "").split()[0][:5]
                if vis.facility_hint
                else ""
            )
            if not cand_prefix or cand_prefix != consensus_prefix:
                continue

            reconciled = FacilityMatch(
                facility_id=consensus.facility_id,
                canonical_name=consensus.canonical_name,
                region=consensus.region,
                score=round(consensus.score * 0.95, 3),
                legibility=vis.legibility,
                needs_review=False,
                match_reason=f"album_consensus({consensus.match_reason})",
            )
            analyzed[idx] = (item, vis, reconciled, used_m, lat_ms)
            if not dry_run:
                self.storage.save_item(
                    item,
                    vis,
                    reconciled,
                    used_model=used_m,
                    latency_ms=lat_ms,
                    copy_files=True,
                )
            log.info(
                "Консенсус альбому для %s -> %s (%s)",
                item.file_path.name[:24],
                reconciled.facility_id,
                reconciled.canonical_name[:35],
            )

    def _apply_burst_inheritance(
        self,
        analyzed: list[
            tuple[IncomingMediaItem, VisionAnalysisResult, Optional[FacilityMatch], str, int]
        ],
        *,
        dry_run: bool,
    ) -> None:
        """
        Прив'язує фото без власних реквізитів (напр. контрольну порцію) до сусіднього документа
        в межах <= 2.5с того ж каналу (або того ж Telegram-чату).
        """
        matched_entries = [
            (m_item, m_vis, m_match)
            for m_item, m_vis, m_match, _, _ in analyzed
            if m_vis.is_report and m_match is not None
        ]
        if not matched_entries:
            return

        for idx, (item, vis, match, used_m, lat_ms) in enumerate(analyzed):
            has_own_hint = bool(vis.facility_hint or vis.settlement_or_address_hint)
            if not (vis.is_report and match is None and not has_own_hint):
                continue

            if item.channel == "telegram" and item.chat_title:
                donors = [
                    (d_item, d_vis, d_match)
                    for d_item, d_vis, d_match in matched_entries
                    if d_item.channel == "telegram" and d_item.chat_title == item.chat_title
                ]
            else:
                donors = [
                    (d_item, d_vis, d_match)
                    for d_item, d_vis, d_match in matched_entries
                    if d_item.channel == item.channel
                    and abs((d_item.created_at - item.created_at).total_seconds()) <= 2.5
                ]

            if len({d_match.facility_id for _, _, d_match in donors}) != 1:
                continue

            _, closest_vis, shared_match = min(
                donors,
                key=lambda d: abs((d[0].created_at - item.created_at).total_seconds()),
            )
            if not vis.raw_document_date and closest_vis.document_date:
                vis.document_date = closest_vis.document_date

            inherited = FacilityMatch(
                facility_id=shared_match.facility_id,
                canonical_name=shared_match.canonical_name,
                region=shared_match.region,
                score=round(shared_match.score * 0.9, 3),
                legibility=shared_match.legibility,
                needs_review=shared_match.needs_review,
                match_reason=f"burst_inherit({shared_match.match_reason})",
            )
            analyzed[idx] = (item, vis, inherited, used_m, lat_ms)
            if not dry_run:
                self.storage.save_item(
                    item,
                    vis,
                    inherited,
                    used_model=used_m,
                    latency_ms=lat_ms,
                    copy_files=True,
                )
            log.info(
                "Успадковано заклад з пакета для %s -> %s (%s)",
                item.file_path.name[:24],
                inherited.facility_id,
                inherited.canonical_name[:35],
            )

    def process_batch(
        self,
        batch: list[IncomingMediaItem],
        *,
        dry_run: bool = False,
    ) -> list[tuple[IncomingMediaItem, VisionAnalysisResult, Optional[FacilityMatch]]]:
        analyzed: list[
            tuple[IncomingMediaItem, VisionAnalysisResult, Optional[FacilityMatch], str, int]
        ] = []
        seen_hashes_in_batch: dict[str, str] = {}

        for item in batch:
            if self._is_duplicate(item, seen_hashes_in_batch, dry_run=dry_run):
                continue

            if not dry_run:
                self.storage.mark_processing(item)

            t0 = time.perf_counter()
            try:
                vis, match = self._analyze_with_upright_retry(item)
            except Exception as exc:
                if not dry_run:
                    retries = self.storage.mark_failed(item, str(exc))
                    log.error(
                        "[FAILED] Не вдалося обробити %s (спроба #%d): %s",
                        item.file_path.name[:32],
                        retries,
                        exc,
                    )
                    self.monitor.notify_llm_failure(item.file_path.name, str(exc))
                else:
                    log.error("[FAILED] Помилка на %s: %s", item.file_path.name[:32], exc)
                continue

            self._validate_item_date(item, vis)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            used_model = self.vision.last_used_route

            if vis.is_report and match and not match.needs_review:
                if vis.stamp_number:
                    self.storage.register_stamp(
                        vis.stamp_number,
                        match,
                        document_type=vis.document_type,
                    )
                if self.storage.learn_contacts(vis, match):
                    self.catalog.update_learned_contacts(
                        self.storage.get_learned_contacts_map()
                    )

            # Зберігаємо одразу по кожному файлу, щоб при збої прогрес не губився
            if not dry_run:
                self.storage.save_item(
                    item,
                    vis,
                    match,
                    used_model=used_model,
                    latency_ms=elapsed_ms,
                    copy_files=True,
                )

            self._log_item_result(item, vis, match, used_model, elapsed_ms)
            analyzed.append((item, vis, match, used_model, elapsed_ms))

        if len(analyzed) > 1:
            self._apply_album_consensus(analyzed, dry_run=dry_run)
            self._apply_burst_inheritance(analyzed, dry_run=dry_run)

        return [(item, vis, match) for item, vis, match, _, _ in analyzed]

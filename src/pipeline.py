from __future__ import annotations

import time
from typing import Optional

from src.catalog import FacilityCatalog
from src.config import Settings
from src.connectors.viber_local import ViberLocalConnector
from src.logger import get_logger
from src.models import FacilityMatch, IncomingMediaItem, VisionAnalysisResult
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
        self.storage = ReportStorage(settings.storage_root)
        self.storage.backfill_learned_contacts_from_history()
        self.catalog.update_learned_contacts(self.storage.get_learned_contacts_map())
        self.vision = VisionModelRouter(settings)
        self.watchdog = ViberProcessWatchdog()
        self.viber = ViberLocalConnector(
            downloads_dir=settings.viber_downloads_dir,
            is_processed_fn=self.storage.is_completed,
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
            if item.file_sha256:
                existing_dup = self.storage.find_duplicate_by_sha256(
                    item.file_sha256, item.media_id
                )
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
                    continue

                if item.file_sha256 in seen_hashes_in_batch:
                    orig_id = seen_hashes_in_batch[item.file_sha256]
                    log.info(
                        "[DUPLICATE] %s дублює файл %s у поточному пакеті",
                        item.file_path.name[:32],
                        orig_id[:24],
                    )
                    if not dry_run:
                        self.storage.record_duplicate(item, orig_id)
                    continue

                seen_hashes_in_batch[item.file_sha256] = item.media_id

            if not dry_run:
                self.storage.mark_processing(item)

            t0 = time.perf_counter()
            try:
                vis = self.vision.analyze_image(item.file_path, caption=item.caption)
            except Exception as exc:
                if not dry_run:
                    retries = self.storage.mark_failed(item, str(exc))
                    log.error(
                        "[FAILED] Не вдалося обробити %s (спроба #%d): %s",
                        item.file_path.name[:32],
                        retries,
                        exc,
                    )
                else:
                    log.error("[FAILED] Помилка на %s: %s", item.file_path.name[:32], exc)
                continue

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            used_model = self.vision.last_used_route

            match: Optional[FacilityMatch] = None
            if vis.is_report:
                match = self.catalog.match(
                    vis,
                    sender_phone=item.sender_phone,
                    sender_name=item.sender_name,
                    chat_title=item.chat_title,
                    caption=item.caption,
                )
                if match and vis.stamp_number and not match.needs_review:
                    self.storage.register_stamp(vis.stamp_number, match)
                elif not match and vis.stamp_number:
                    match = self.storage.lookup_stamp(vis.stamp_number)

                if match and not match.needs_review:
                    newly_learned = self.storage.learn_contacts(vis, match)
                    if newly_learned:
                        self.catalog.update_learned_contacts(
                            self.storage.get_learned_contacts_map()
                        )

            # Зберігаємо результат одразу по кожному файлу, щоб при раптовому збої прогрес не губився
            if not dry_run:
                self.storage.save_item(
                    item,
                    vis,
                    match,
                    used_model=used_model,
                    latency_ms=elapsed_ms,
                    copy_files=True,
                )

            if match:
                flag = " [NEEDS_REVIEW]" if match.needs_review else ""
                log.info(
                    "%s (%dms, %s) -> type=%s, legibility=%s, ai_conf=%.2f, score=%.2f%s -> %s (%s)",
                    item.file_path.name[:24],
                    elapsed_ms,
                    used_model,
                    vis.document_type,
                    vis.legibility,
                    vis.confidence,
                    match.score,
                    flag,
                    match.facility_id,
                    match.canonical_name[:35],
                )
            else:
                log.warning(
                    "%s (%dms, %s) -> type=%s, legibility=%s, hint=%r -> UNMATCHED",
                    item.file_path.name[:24],
                    elapsed_ms,
                    used_model,
                    vis.document_type,
                    vis.legibility,
                    vis.facility_hint or vis.settlement_or_address_hint,
                )

            analyzed.append((item, vis, match, used_model, elapsed_ms))

        if len(analyzed) <= 1:
            return [(item, vis, match) for item, vis, match, _, _ in analyzed]

        # Консенсус альбому застосовуємо ЛИШЕ коли в межах 2.5с назви-кандидати схожі за префіксом
        # (напр. прописне "Великояблуньківський" vs "Великоглушанський"), а не для різних шкіл при масовій синхронізації Viber
        album_span = abs((analyzed[-1][0].created_at - analyzed[0][0].created_at).total_seconds())
        known_matches = {m.facility_id: m for _, v, m, _, _ in analyzed if v.is_report and m is not None}

        if len(analyzed) >= 3 and album_span <= 2.5 and len(known_matches) > 1:
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
            if counts[best_fid] >= len(analyzed) * 0.6 and best_fid in contact_backed:
                consensus = known_matches[best_fid]
                consensus_prefix = consensus.canonical_name.lower().replace('"', "").split()[0][:5]
                for idx, (item, vis, match, used_m, lat_ms) in enumerate(analyzed):
                    if not vis.is_report or match is None or match.facility_id == best_fid:
                        continue
                    cand_prefix = (vis.facility_hint or "").lower().replace('"', "").split()[0][:5] if vis.facility_hint else ""
                    if cand_prefix and cand_prefix == consensus_prefix:
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

        # Успадкування закладу для фото без тексту (наприклад, фото страв у тому ж пакеті)
        if len(known_matches) == 1:
            shared_match = next(iter(known_matches.values()))
            for idx, (item, vis, match, used_m, lat_ms) in enumerate(analyzed):
                has_own_hint = bool(vis.facility_hint or vis.settlement_or_address_hint)
                if vis.is_report and match is None and not has_own_hint:
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

        return [(item, vis, match) for item, vis, match, _, _ in analyzed]

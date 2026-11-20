from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from src.alerts import TelegramNotifier
from src.config import Settings
from src.connectors.telegram_user import (
    TelegramUserConnector,
    build_telegram_media_id,
    dialog_matches_filter,
    extract_sender_info,
    is_supported_media_message,
)
from src.llm.router import VisionModelRouter
from src.models import FacilityMatch, IncomingMediaItem, VisionAnalysisResult
from src.monitoring import ActivityMonitor
from src.pipeline import ReportPipeline
from src.storage import ReportStorage
from src.viber_watchdog import ViberProcessWatchdog


def _save_random_image(path: Path, *, fmt: str = "JPEG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.frombytes(
        "RGB",
        (120, 120),
        bytes(random.getrandbits(8) for _ in range(120 * 120 * 3)),
    )
    im.save(path, format=fmt)
    return path


class StorageMonitoringAndConnectorsTests(unittest.TestCase):
    def test_viber_rotation_and_storage_closes_db_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            viber_dir = tmp_path / "ViberDownloads"
            archive_dir = tmp_path / "ViberArchive"
            viber_dir.mkdir()

            def make_img(name: str, age_days: int) -> Path:
                p = _save_random_image(viber_dir / name)
                ts = (datetime.now() - timedelta(days=age_days)).timestamp()
                os.utime(p, (ts, ts))
                return p

            f_old_matched = make_img("old_matched.jpg", 25)
            f_old_failed = make_img("old_failed.jpg", 25)
            f_new_matched = make_img("new_matched.jpg", 5)

            settings = Settings(
                storage_root=tmp_path / "storage",
                viber_downloads_dir=viber_dir,
                viber_pc_dir=tmp_path / "viber_pc",
                viber_retention_days=21,
                viber_archive_dir=archive_dir,
            )
            pipeline = ReportPipeline(settings)

            vis_ok = VisionAnalysisResult(is_report=True, document_type="act", confidence=0.95)
            match_ok = FacilityMatch(
                facility_id="FAC-0053",
                canonical_name="Карасинський ліцей",
                region="Камінь-Каширський район",
                score=0.98,
                match_reason="test",
            )
            for stem, p in [("old_matched", f_old_matched), ("new_matched", f_new_matched)]:
                pipeline.storage.save_item(
                    IncomingMediaItem(
                        media_id=stem,
                        file_path=p,
                        file_sha256=stem,
                        created_at=datetime.now(),
                    ),
                    vis_ok,
                    match_ok,
                    copy_files=False,
                )
            pipeline.storage.mark_failed(
                IncomingMediaItem(
                    media_id="old_failed",
                    file_path=f_old_failed,
                    file_sha256="fail",
                    created_at=datetime.now(),
                ),
                "timeout",
            )

            dry_stats = pipeline.rotate_viber_downloads(dry_run=True)
            self.assertEqual((dry_stats.rotated, dry_stats.skipped_unprocessed), (1, 1))
            self.assertTrue(f_old_matched.exists())

            arch_stats = pipeline.rotate_viber_downloads(dry_run=False)
            self.assertEqual((arch_stats.rotated, arch_stats.archived), (1, 1))
            self.assertFalse(f_old_matched.exists())
            self.assertTrue(f_old_failed.exists())
            self.assertTrue(f_new_matched.exists())

    def test_activity_monitoring_and_telegram_hot_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env"
            env_file.write_text("TELEGRAM_BOT_TOKEN=\nTECH_ALERT_CHAT_ID=\n", encoding="utf-8")

            settings = Settings(
                storage_root=tmp_path / "storage",
                viber_downloads_dir=tmp_path / "viber",
                viber_pc_dir=tmp_path / "viber_pc",
                telegram_bot_token="",
                tech_alert_chat_id="",
                silence_alert_minutes=120,
                alert_cooldown_minutes=60,
            )
            storage = ReportStorage(settings.storage_root)
            watchdog = ViberProcessWatchdog()
            notifier = TelegramNotifier(settings, env_path=env_file)
            monitor = ActivityMonitor(settings, storage, watchdog, notifier)

            self.assertFalse(monitor.is_within_work_hours(datetime(2026, 11, 19, 8, 59)))
            self.assertTrue(monitor.is_within_work_hours(datetime(2026, 11, 19, 14, 30)))
            self.assertFalse(monitor.is_within_work_hours(datetime(2026, 11, 21, 12, 0)))

            # Morning grace window: at 09:30 (30m after 09:00), silence threshold 120m shouldn't fire
            self.assertFalse(
                monitor.check_silence(now=datetime(2026, 11, 19, 9, 30), viber_running=True)
            )
            # At 11:05 (125m into workday), local alert is recorded
            self.assertTrue(
                monitor.check_silence(now=datetime(2026, 11, 19, 11, 5), viber_running=True)
            )
            # Cooldown suppresses repeat 10 min later
            self.assertFalse(
                monitor.check_silence(now=datetime(2026, 11, 19, 11, 15), viber_running=True)
            )

            env_file.write_text(
                "TELEGRAM_BOT_TOKEN=123:ABC\nTECH_ALERT_CHAT_ID=-100999\n",
                encoding="utf-8",
            )
            future_mtime = time.time() + 2
            os.utime(env_file, (future_mtime, future_mtime))
            self.assertTrue(notifier.is_configured)

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            with patch("src.alerts.httpx.Client") as mock_client_cls:
                mock_client = mock_client_cls.return_value.__enter__.return_value
                mock_client.post.return_value = mock_resp
                self.assertTrue(
                    monitor.check_silence(now=datetime(2026, 11, 19, 12, 10), viber_running=True)
                )
                mock_client.post.assert_called_once()

            recent = storage.get_recent_alerts(limit=5)
            self.assertEqual(len(recent), 2)
            self.assertTrue(recent[0]["delivered_telegram"])
            self.assertFalse(recent[1]["delivered_telegram"])

    def test_telegram_user_connector_and_pipeline_e2e(self) -> None:
        self.assertEqual(build_telegram_media_id(-1001, 42), "tg_-1001_42")
        self.assertTrue(dialog_matches_filter(-1001, "Кухарі", is_group=True, is_channel=False))
        self.assertFalse(dialog_matches_filter(55, "DM", is_group=False, is_channel=False))
        self.assertEqual(
            is_supported_media_message(SimpleNamespace(photo=None, document=SimpleNamespace(mime_type="image/png"))),
            (True, ".png"),
        )
        self.assertEqual(
            extract_sender_info(SimpleNamespace(first_name="Олена", last_name="Тест", phone="380501112233")),
            ("Олена Тест", "380501112233"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            viber_dir = tmp_path / "viber"
            tg_dir = tmp_path / "tg"
            out_dir = tmp_path / "NetworkPublic"
            viber_dir.mkdir()

            recv = datetime(2026, 11, 18, 14, 0, 0)
            f_act = _save_random_image(viber_dir / "act.jpg")
            f_food_close = _save_random_image(viber_dir / "food_close.jpg")
            f_food_far = _save_random_image(viber_dir / "food_far.jpg")
            for p, dt in [
                (f_act, recv),
                (f_food_close, recv + timedelta(seconds=1.5)),
                (f_food_far, recv + timedelta(seconds=5.0)),
            ]:
                ts = dt.timestamp()
                os.utime(p, (ts, ts))

            now_utc = datetime.now(timezone.utc)
            sender_obj = SimpleNamespace(first_name="Іван", last_name="Кухар", phone="380991112233")

            class FakeMsg:
                def __init__(self, mid: int, dt: datetime, text: str | None = None) -> None:
                    self.id = mid
                    self.date = dt
                    self.photo = object()
                    self.document = None
                    self.message = text

                async def get_sender(self):
                    return sender_obj

            class FakeClient:
                def __init__(self, session, api_id, api_hash) -> None:
                    pass

                async def connect(self) -> None:
                    pass

                async def disconnect(self) -> None:
                    pass

                async def is_user_authorized(self) -> bool:
                    return True

                async def iter_dialogs(self):
                    yield SimpleNamespace(id=-100500, title="Кухарі Маневичі", is_group=True, is_channel=False, entity="g")

                async def iter_messages(self, entity, limit=250):
                    yield FakeMsg(10, now_utc - timedelta(minutes=1), text="Акт")

                async def download_media(self, msg, file: str) -> str:
                    return str(_save_random_image(Path(file)))

            settings = Settings(
                storage_root=tmp_path / "storage",
                reports_output_dir=out_dir,
                viber_downloads_dir=viber_dir,
                viber_pc_dir=tmp_path / "viber_pc",
                telegram_api_id=123,
                telegram_api_hash="abc",
                telegram_session_path=tmp_path / "tg.session",
                telegram_downloads_dir=tg_dir,
            )
            pipeline = ReportPipeline(settings)
            pipeline.telegram._client_factory = FakeClient

            pending, batches = pipeline.fetch_pending_batches()
            self.assertEqual(len(pending), 4)
            self.assertEqual(len(batches), 2)

            def fake_analyze(path, caption=None, *, rotation_degrees=0, reference_date=None):
                if path.name == "act.jpg":
                    return VisionAnalysisResult(
                        is_report=True,
                        document_type="act",
                        document_date="2024-11-17",
                        matched_facility_id="FAC-0053",
                        facility_hint="Карасинський ліцей",
                        legibility="clear",
                        confidence=0.96,
                    )
                return VisionAnalysisResult(
                    is_report=True,
                    document_type="food_photo",
                    document_date=None,
                    legibility="clear",
                    confidence=0.92,
                )

            pipeline.vision.analyze_image = fake_analyze
            pipeline.vision.last_used_route = "gemini:test"

            viber_batch = [
                IncomingMediaItem(media_id="act_01", file_path=f_act, file_sha256="s1", created_at=recv),
                IncomingMediaItem(
                    media_id="food_cls_01",
                    file_path=f_food_close,
                    file_sha256="s2",
                    created_at=recv + timedelta(seconds=1.5),
                ),
                IncomingMediaItem(
                    media_id="food_far_01",
                    file_path=f_food_far,
                    file_sha256="s3",
                    created_at=recv + timedelta(seconds=5.0),
                ),
            ]
            pipeline.process_batch(viber_batch, dry_run=False)

            expected_act = (
                out_dir
                / "Щоденні документи"
                / "Камінь-Каширський район"
                / "Карасинський ліцей"
                / "2026-11 Листопад"
                / "2026-11-17"
                / "2026-11-17_Акт_act_01.jpg"
            )
            expected_close = (
                out_dir
                / "Щоденні документи"
                / "Камінь-Каширський район"
                / "Карасинський ліцей"
                / "2026-11 Листопад"
                / "2026-11-17"
                / "Контрольна порція"
                / "2026-11-17_Контрольна_порція_food_cls_01.jpg"
            )
            expected_far = (
                settings.storage_root
                / "unmatched"
                / "2026-11-18"
                / "Контрольна порція"
                / "2026-11-18_Контрольна_порція_food_far_01.jpg"
            )
            self.assertTrue(expected_act.exists())
            self.assertTrue(expected_close.exists())
            self.assertTrue(expected_far.exists())

            with sqlite3.connect(pipeline.storage.db_path) as conn:
                row = conn.execute(
                    "SELECT document_date, vision_json FROM processed_media WHERE media_id = 'act_01'"
                ).fetchone()
            conn.close()
            self.assertEqual(row[0], "2026-11-17")
            self.assertEqual(json.loads(row[1])["raw_document_date"], "2024-11-17")

    def test_router_cooldown_sleep_and_earliest_order(self) -> None:
        router = object.__new__(VisionModelRouter)
        router.system_prompt = "sys"
        router.cooldown_seconds = 60.0
        router.max_attempts_per_model = 1
        router.last_used_route = ""
        router.reload_if_changed = lambda force=False: False

        call_order: list[str] = []
        ok_result = VisionAnalysisResult(is_report=True, document_type="act", confidence=0.9)

        def make_provider(name: str, fail: bool) -> SimpleNamespace:
            def analyze(*args, **kwargs):
                call_order.append(name)
                if fail:
                    raise RuntimeError("429 quota")
                return ok_result

            return SimpleNamespace(
                name=name,
                model=f"m-{name}",
                provider_type="gemini",
                analyze=analyze,
            )

        p_late = make_provider("late", fail=False)
        p_early = make_provider("early", fail=True)
        now = time.time()
        router.active_providers = [p_late, p_early]
        router._cooldown_until = {"late": now + 30.0, "early": now + 4.0}

        sleep_calls: list[float] = []
        with patch("src.llm.router.time.sleep", side_effect=sleep_calls.append):
            res = router.analyze_image(Path("dummy.jpg"))

        self.assertIs(res, ok_result)
        self.assertEqual(call_order, ["early", "late"])
        self.assertEqual(len(sleep_calls), 1)
        self.assertGreaterEqual(sleep_calls[0], 1.0)
        self.assertLessEqual(sleep_calls[0], 5.0)

    def test_stamp_registry_hygiene_and_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = ReportStorage(Path(tmp) / "storage")
            match_a = FacilityMatch(
                facility_id="FAC-0076",
                canonical_name="Олениненська гімназія",
                region="Камінь-Каширський район",
                score=0.90,
                match_reason="settlement:оленине",
            )
            match_b = FacilityMatch(
                facility_id="FAC-0192",
                canonical_name="Озерський ліцей",
                region="Сарненський р-н",
                score=0.90,
                match_reason="settlement:озера",
            )

            # Номер складу '201' та фото їжі не повинні потрапляти в реєстр штампів
            storage.register_stamp("201", match_a, document_type="waybill_external")
            self.assertIsNone(storage.lookup_stamp("201"))
            storage.register_stamp("77", match_a, document_type="food_photo")
            self.assertIsNone(storage.lookup_stamp("77"))

            # Валідний штамп працює, а при конфлікті двох закладів стає неоднозначним (без перезапису)
            storage.register_stamp("77", match_a, document_type="act")
            hit = storage.lookup_stamp("77")
            self.assertIsNotNone(hit)
            self.assertEqual(hit.facility_id, "FAC-0076")

            storage.register_stamp("77", match_b, document_type="waybill_external")
            self.assertIsNone(storage.lookup_stamp("77"))
            with sqlite3.connect(storage.db_path) as conn:
                row = conn.execute(
                    "SELECT facility_id, occurrences, is_ambiguous FROM stamp_registry WHERE stamp_number = '77'"
                ).fetchone()
            conn.close()
            self.assertEqual(row, ("FAC-0076", 2, 1))

    def test_upright_retry_runs_for_90_270_even_with_high_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = Settings(
                storage_root=tmp_path / "storage",
                viber_downloads_dir=tmp_path / "ViberDownloads",
                viber_pc_dir=tmp_path / "viber_pc",
            )
            pipeline = ReportPipeline(settings)
            img_path = _save_random_image(tmp_path / "ttn.jpg")
            item = IncomingMediaItem(
                media_id="ttn_1",
                file_path=img_path,
                file_sha256="abc",
                created_at=datetime(2026, 11, 20, 10, 0, 0),
            )

            pass_rotations: list[int] = []

            def fake_analyze(path, caption=None, *, rotation_degrees=0, reference_date=None):
                pass_rotations.append(rotation_degrees)
                if rotation_degrees == 0:
                    return VisionAnalysisResult(
                        is_report=True,
                        document_type="waybill_external",
                        matched_facility_id="FAC-0053",
                        facility_hint="Карасинський ліцей",
                        legibility="clear",
                        confidence=0.96,
                        rotation_degrees=270,
                    )
                return VisionAnalysisResult(
                    is_report=True,
                    document_type="waybill_external",
                    matched_facility_id="FAC-0053",
                    facility_hint="Карасинський ліцей",
                    legibility="clear",
                    confidence=0.96,
                    rotation_degrees=180,
                )

            pipeline.vision.analyze_image = fake_analyze
            vis, match = pipeline._analyze_with_upright_retry(item)
            self.assertEqual(pass_rotations, [0, 270])
            self.assertEqual(vis.rotation_degrees, 90)
            self.assertIsNotNone(match)
            self.assertEqual(match.facility_id, "FAC-0053")


if __name__ == "__main__":
    unittest.main()

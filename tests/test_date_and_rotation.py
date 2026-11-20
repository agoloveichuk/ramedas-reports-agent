from __future__ import annotations

import base64
from datetime import date, datetime
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.config import PROJECT_ROOT
from src.date_utils import (
    format_month_folder,
    shorten_facility_dirname,
    validate_document_date,
)
from src.image_utils import (
    encode_image_for_llm,
    has_non_default_exif_orientation,
    normalize_rotation_degrees,
    rotate_pil_image_clockwise,
    save_oriented_image,
)
from src.models import VisionAnalysisResult


class DateUtilsTests(unittest.TestCase):
    def test_format_month_folder(self) -> None:
        self.assertEqual(format_month_folder(date(2026, 11, 18)), "2026-11 Листопад")
        self.assertEqual(format_month_folder(date(2026, 1, 5)), "2026-01 Січень")

    def test_shorten_facility_dirname_unique_for_catalog(self) -> None:
        facs_path = PROJECT_ROOT / "data" / "facilities.json"
        facilities = json.loads(facs_path.read_text(encoding="utf-8-sig"))
        seen: dict[tuple[str, str], str] = {}
        for fac in facilities:
            key = (fac["region"], shorten_facility_dirname(fac["canonical_name"]).lower())
            self.assertNotIn(key, seen, f"Collision for {key}")
            seen[key] = fac["id"]

        self.assertEqual(
            shorten_facility_dirname(
                'Опорний заклад загальної середньої освіти "Карасинський ліцей" '
                "Сошичненської сільської ради Волинської області"
            ),
            "Карасинський ліцей",
        )

    def test_validate_document_date(self) -> None:
        recv = datetime(2026, 11, 18, 14, 0, 0)
        self.assertEqual(validate_document_date("2026-11-18", recv), ("2026-11-18", "document_date"))
        self.assertEqual(validate_document_date("2026-11-11", recv), ("2026-11-11", "document_date"))
        self.assertEqual(
            validate_document_date("2023-11-16", recv),
            ("2026-11-16", "year_fixed:2023-11-16"),
        )
        self.assertEqual(
            validate_document_date("2026-03-18", recv),
            ("2026-11-18", "fallback_received_invalid:2026-03-18"),
        )
        self.assertEqual(
            validate_document_date(None, recv),
            ("2026-11-18", "fallback_received_missing"),
        )

        jan_recv = datetime(2026, 1, 4, 10, 0, 0)
        self.assertEqual(
            validate_document_date("2025-12-29", jan_recv),
            ("2025-12-29", "document_date"),
        )
        self.assertEqual(
            validate_document_date("2023-12-29", jan_recv),
            ("2025-12-29", "year_fixed:2023-12-29"),
        )


class ImageUtilsTests(unittest.TestCase):
    def test_normalize_rotation_degrees(self) -> None:
        cases = [
            (0, 0),
            (90, 90),
            (180, 180),
            (270, 270),
            (-90, 270),
            (360, 0),
            ("270°", 270),
            ("left", 90),
            ("right", 270),
            ("bottom", 180),
            (None, 0),
            (True, 0),
        ]
        for raw, expected in cases:
            self.assertEqual(normalize_rotation_degrees(raw), expected)

        vis = VisionAnalysisResult(is_report=True, rotation_degrees="-90")
        self.assertEqual(vis.rotation_degrees, 270)

    def test_clockwise_geometry_exif_and_encoding(self) -> None:
        img = Image.new("RGB", (200, 100), color=(0, 0, 0))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((199, 0), (0, 255, 0))
        img.putpixel((0, 99), (0, 0, 255))

        r90 = rotate_pil_image_clockwise(img, 90)
        self.assertEqual(r90.size, (100, 200))
        self.assertEqual(r90.getpixel((0, 0)), (0, 0, 255))

        r180 = rotate_pil_image_clockwise(img, 180)
        self.assertEqual(r180.size, (200, 100))
        self.assertEqual(r180.getpixel((199, 99)), (255, 0, 0))

        r270 = rotate_pil_image_clockwise(img, 270)
        self.assertEqual(r270.size, (100, 200))
        self.assertEqual(r270.getpixel((0, 0)), (0, 255, 0))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_png = tmp_path / "src.png"
            img.save(src_png, format="PNG")

            dst0 = tmp_path / "dst0.png"
            save_oriented_image(src_png, dst0, rotation_degrees=0)
            self.assertEqual(dst0.read_bytes(), src_png.read_bytes())

            mime, b64 = encode_image_for_llm(src_png, rotation_degrees=90)
            self.assertEqual(mime, "image/png")
            decoded = Image.open(io.BytesIO(base64.b64decode(b64)))
            self.assertEqual(decoded.size, (100, 200))

            exif_jpg = tmp_path / "exif6.jpg"
            exif = Image.Exif()
            exif[0x0112] = 6
            img.save(exif_jpg, format="JPEG", exif=exif)
            self.assertTrue(has_non_default_exif_orientation(exif_jpg))
            dst_exif = tmp_path / "dst_exif.jpg"
            save_oriented_image(exif_jpg, dst_exif, rotation_degrees=0)
            with Image.open(dst_exif) as ex_img:
                self.assertEqual(ex_img.size, (100, 200))
                self.assertIn(ex_img.getexif().get(0x0112), (None, 1))


if __name__ == "__main__":
    unittest.main()

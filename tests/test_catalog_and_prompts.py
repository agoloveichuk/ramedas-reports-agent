from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from src.catalog import FacilityCatalog
from src.config import PROJECT_ROOT
from src.llm.prompts import build_user_prompt, extract_json_object
from src.models import VisionAnalysisResult


class CatalogAndPromptsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = FacilityCatalog(PROJECT_ROOT / "data" / "facilities.json")

    def test_village_based_waybill_matches(self) -> None:
        vis_verhy = VisionAnalysisResult(
            is_report=True,
            document_type="waybill_external",
            facility_hint="Гімназія с. Верхи",
            settlement_or_address_hint="с. Верхи",
            legibility="clear",
            confidence=0.85,
        )
        m_verhy = self.catalog.match(vis_verhy)
        self.assertIsNotNone(m_verhy)
        self.assertEqual(m_verhy.facility_id, "FAC-0059")
        self.assertFalse(m_verhy.needs_review)
        self.assertGreaterEqual(m_verhy.score, 0.78)

        vis_gorodok = VisionAnalysisResult(
            is_report=True,
            document_type="waybill_external",
            facility_hint="ЗОШ ДНЗ с.Городок",
            settlement_or_address_hint="с. Городок",
            legibility="clear",
            confidence=0.90,
        )
        m_gorodok = self.catalog.match(vis_gorodok)
        self.assertIsNotNone(m_gorodok)
        self.assertEqual(m_gorodok.facility_id, "FAC-0104")
        self.assertFalse(m_gorodok.needs_review)

        # Чіткий акт лише за назвою села без прізвища кухаря все ще вимагає підтвердження
        vis_act_only_village = VisionAnalysisResult(
            is_report=True,
            document_type="act",
            facility_hint="Гімназія",
            settlement_or_address_hint="с. Городок",
            legibility="clear",
            confidence=0.90,
        )
        m_act = self.catalog.match(vis_act_only_village)
        self.assertIsNotNone(m_act)
        self.assertTrue(m_act.needs_review)

    def test_surname_matching_uses_whole_tokens(self) -> None:
        sample = [
            {
                "id": "FAC-TEST-1",
                "canonical_name": "Тестова гімназія №1",
                "region": "Тестова громада",
                "settlement": "с. Тестове",
                "aliases": [],
                "contacts": [{"name": "Нікітін Олександр", "phone": ""}],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "facilities.json"
            p.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
            cat = FacilityCatalog(p)

        vis_partial = VisionAnalysisResult(
            is_report=True,
            document_type="act",
            person_hints=["Кіт Марія"],
            confidence=0.90,
        )
        self.assertIsNone(cat.match(vis_partial))

        vis_exact = VisionAnalysisResult(
            is_report=True,
            document_type="act",
            person_hints=["Нікітін О.П."],
            confidence=0.90,
        )
        hit = cat.match(vis_exact)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.facility_id, "FAC-TEST-1")
        self.assertIn("contact:нікітін", hit.match_reason)

    def test_prompts_and_json_extraction(self) -> None:
        prompt = build_user_prompt("примітка", reference_date=datetime(2026, 11, 20, 10, 0))
        self.assertIn("2026-11-20", prompt)
        self.assertIn("примітка", prompt)

        wrapped = '```json\n[{"is_report": true, "document_type": "act"}]\n```'
        parsed = extract_json_object(wrapped)
        self.assertEqual(parsed["document_type"], "act")


if __name__ == "__main__":
    unittest.main()

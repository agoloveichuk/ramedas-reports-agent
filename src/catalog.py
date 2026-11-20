from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional
from rapidfuzz import fuzz

from src.config import IGNORED_LEGAL_ENTITIES
from src.models import FacilityMatch, VisionAnalysisResult

STOP_WORDS: frozenset[str] = frozenset(
    {
        "ззсо",
        "зош",
        "здо",
        "днз",
        "нвк",
        "заклад",
        "загальної",
        "середньої",
        "дошкільної",
        "освіти",
        "ліцей",
        "гімназія",
        "школа",
        "садок",
        "дитячий",
        "ясла",
        "філія",
        "опорний",
        "сільської",
        "селищної",
        "міської",
        "районної",
        "ради",
        "рада",
        "області",
        "область",
        "району",
        "район",
        "волинської",
        "волинська",
        "рівненської",
        "рівненська",
        "хмельницької",
        "хмельницька",
        "київської",
        "київська",
        "відділ",
        "молоді",
        "спорту",
        "управління",
        "імені",
        "героя",
        "україни",
        "село",
        "місто",
        "смт",
        "вулиця",
        "вул",
        "буд",
    }
)


def _normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D+", "", phone or "")
    if len(digits) == 12 and digits.startswith("380"):
        return digits[2:]
    if len(digits) == 10 and digits.startswith("0"):
        return digits
    return digits


def _normalize_text(text: str) -> str:
    text = (
        (text or "")
        .lower()
        .replace("’", "'")
        .replace("`", "'")
        .replace("«", " ")
        .replace("»", " ")
        .replace("-", " ")
    )
    for ent in IGNORED_LEGAL_ENTITIES:
        text = text.replace(ent, " ")
    text = re.sub(r"№\s*(\d+)", r"№\1", text)
    text = re.sub(r"[^\w\s№']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _distinctive_tokens(norm_text: str) -> list[str]:
    tokens = []
    for tok in norm_text.split():
        if tok in STOP_WORDS:
            continue
        if tok.startswith("№") or tok.isdigit() or len(tok) >= 3:
            tokens.append(tok)
    return tokens


def _extract_surname(full_name: str) -> str:
    cleaned = re.sub(r"\(.*?\)", "", full_name or "").strip().lower()
    parts = [p for p in re.split(r"\s+", cleaned) if p and len(p) > 2]
    return parts[0] if parts else ""


class FacilityCatalog:
    """Довідник закладів (data/facilities.json) із багатофакторним пошуком та оцінкою впевненості."""

    def __init__(self, json_path: Path, review_threshold: float = 0.78) -> None:
        self.json_path = json_path
        self.review_threshold = review_threshold
        self.facilities: list[dict[str, Any]] = json.loads(
            json_path.read_text(encoding="utf-8-sig")
        )
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_phone: dict[str, list[dict[str, Any]]] = {}
        self._by_surname: dict[str, list[dict[str, Any]]] = {}
        self._learned_by_facility: dict[str, list[str]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        self._by_id.clear()
        self._by_phone.clear()
        self._by_surname.clear()
        for fac in self.facilities:
            fid = fac["id"]
            self._by_id[fid] = fac
            for contact in fac.get("contacts", []):
                norm_phone = _normalize_phone(contact.get("phone", ""))
                if len(norm_phone) == 10:
                    self._by_phone.setdefault(norm_phone, []).append(fac)
                surname = _extract_surname(contact.get("name", ""))
                if surname:
                    self._by_surname.setdefault(surname, []).append(fac)
            for learned_surname in self._learned_by_facility.get(fid, []):
                if learned_surname:
                    self._by_surname.setdefault(learned_surname, []).append(fac)

    def update_learned_contacts(self, learned_map: dict[str, list[str]]) -> None:
        """Доповнює індекс прізвищ контактами, автоматично вивченими з чітких Актів/Меню."""
        self._learned_by_facility = learned_map
        self._build_indexes()

    def match(
        self,
        vision: VisionAnalysisResult,
        *,
        sender_phone: Optional[str] = None,
        sender_name: Optional[str] = None,
        chat_title: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> Optional[FacilityMatch]:
        # 1. Прямий збіг за номером телефону відправника (якщо відомий і унікальний)
        if sender_phone:
            norm_p = _normalize_phone(sender_phone)
            phone_hits = self._by_phone.get(norm_p, [])
            if len(phone_hits) == 1:
                fac = phone_hits[0]
                return FacilityMatch(
                    facility_id=fac["id"],
                    canonical_name=fac["canonical_name"],
                    region=fac["region"],
                    score=0.98,
                    legibility=vision.legibility,
                    needs_review=False,
                    match_reason=f"sender_phone:{norm_p}",
                )

        norm_facility_only = _normalize_text(vision.facility_hint or "")
        norm_settlement = _normalize_text(vision.settlement_or_address_hint or "")
        norm_caption = _normalize_text(caption or "")
        norm_chat = _normalize_text(chat_title or "")

        query_distinct = _distinctive_tokens(
            " ".join(filter(None, [norm_facility_only, norm_settlement, norm_caption]))
        )
        fac_distinct = _distinctive_tokens(norm_facility_only)
        settle_distinct = _distinctive_tokens(norm_settlement)

        person_surnames: set[str] = set()
        for person in list(vision.person_hints) + ([sender_name] if sender_name else []):
            surname = _extract_surname(person)
            if surname:
                person_surnames.add(surname)

        best_fac: Optional[dict[str, Any]] = None
        best_score = 0.0
        best_reason = ""
        best_has_independent_confirm = False

        for fac in self.facilities:
            score = 0.0
            reasons: list[str] = []
            independent_confirm = False

            if vision.matched_facility_id and fac.get("id") == vision.matched_facility_id:
                score += 0.45
                reasons.append(f"ai_id:{fac['id']}")

            canon_norm = _normalize_text(fac.get("canonical_name", ""))
            raw_norm = _normalize_text(fac.get("raw_name", ""))
            addr_norm = _normalize_text(fac.get("address", ""))
            region_norm = _normalize_text(fac.get("region", ""))
            aliases_norm = [_normalize_text(a) for a in fac.get("short_aliases", []) if a]

            target_distinct = _distinctive_tokens(
                f"{canon_norm} {raw_norm} {addr_norm} {' '.join(aliases_norm)}"
            )
            target_distinct_str = " ".join(target_distinct)

            # А) Збіг за характерними словами назви закладу
            name_added = False
            if fac_distinct and target_distinct:
                fac_distinct_str = " ".join(fac_distinct)
                sim = max(
                    fuzz.token_set_ratio(fac_distinct_str, target_distinct_str) / 100.0,
                    fuzz.ratio(fac_distinct_str, " ".join(_distinctive_tokens(canon_norm))) / 100.0,
                )
                for al in aliases_norm:
                    al_dist = " ".join(_distinctive_tokens(al))
                    if al_dist:
                        sim = max(sim, fuzz.ratio(fac_distinct_str, al_dist) / 100.0)

                q_nums = {t for t in fac_distinct if t.startswith("№") or t.isdigit()}
                t_nums = {t for t in target_distinct if t.startswith("№") or t.isdigit()}
                if q_nums and t_nums and not (q_nums & t_nums):
                    sim *= 0.35

                if sim >= 0.72:
                    weight = 0.50 if vision.matched_facility_id else 0.75
                    score += sim * weight
                    name_added = True
                    reasons.append(f"name:{sim:.2f}")

            # Б) Збіг населеного пункту / громади / адреси
            if settle_distinct and target_distinct:
                matched_specific = []
                matched_region = []
                reg_dist = _distinctive_tokens(region_norm)
                for st in settle_distinct:
                    if len(st) < 4:
                        continue
                    stem = st[: max(4, len(st) - 1)]
                    is_region_tok = any(
                        fuzz.ratio(st, rt) >= 82 or rt.startswith(stem) for rt in reg_dist
                    )
                    if is_region_tok:
                        matched_region.append(st)
                    elif any(
                        fuzz.ratio(st, tt) >= 82 or tt.startswith(stem) for tt in target_distinct
                    ):
                        matched_specific.append(st)

                if matched_specific:
                    only_settle_tokens = not fac_distinct or set(fac_distinct) <= set(matched_specific)
                    is_clear_waybill = (
                        vision.document_type in ("waybill_external", "waybill_internal")
                        and vision.legibility == "clear"
                    )
                    base_settle_score = 0.85 if is_clear_waybill else 0.65
                    score += base_settle_score if (not name_added and only_settle_tokens) else 0.30
                    reasons.append(f"settlement:{','.join(matched_specific[:2])}")
                    if vision.document_type in ("waybill_external", "waybill_internal"):
                        independent_confirm = True
                if matched_region and (matched_specific or fac_distinct):
                    score += 0.15
                    reasons.append(f"region:{','.join(matched_region[:2])}")

            # В) Збіг прізвища кухаря з довідником контактів або вивченими з Актів прізвищами
            if person_surnames:
                static_contacts = " ".join(
                    _normalize_text(c.get("name", "")) for c in fac.get("contacts", [])
                )
                learned_contacts = " ".join(self._learned_by_facility.get(fac["id"], []))
                static_tokens = set(static_contacts.split())
                learned_tokens = set(learned_contacts.split())
                for surname in person_surnames:
                    norm_surname = surname.replace("’", "'").replace("`", "'").strip(".,;:\"'")
                    in_static = bool(norm_surname and norm_surname in static_tokens)
                    in_learned = bool(norm_surname and norm_surname in learned_tokens)
                    if in_static or in_learned:
                        is_unique_surname = len(self._by_surname.get(surname, [])) == 1
                        if score > 0 or (not query_distinct and is_unique_surname):
                            bonus = 0.65 if (not query_distinct and is_unique_surname) else 0.25
                            score += bonus
                            source_tag = "contact" if in_static else "learned_contact"
                            reasons.append(f"{source_tag}:{surname}")
                            independent_confirm = True
                        break

            # Г) Підказка регіону з назви групового чату
            if norm_chat and region_norm:
                reg_tokens = _distinctive_tokens(region_norm)
                if any(t[:5] in norm_chat for t in reg_tokens if len(t) >= 4):
                    score += 0.10
                    reasons.append("chat_region")

            if score > best_score:
                best_score = score
                best_fac = fac
                best_reason = "+".join(reasons)
                best_has_independent_confirm = independent_confirm

        if not best_fac or best_score < 0.55:
            return None

        # Калібруємо фінальний score з урахуванням впевненості самої ШІ-моделі та чіткості почерку
        ai_conf = vision.confidence if vision.confidence > 0 else 0.85
        legibility_penalty = 0.0
        if vision.legibility == "poor" and not best_has_independent_confirm:
            legibility_penalty = 0.18
        elif vision.legibility == "medium" and not best_has_independent_confirm:
            legibility_penalty = 0.06

        raw_capped = min(best_score, 1.0)
        final_score = round(max(0.0, (raw_capped * 0.65 + ai_conf * 0.35) - legibility_penalty), 3)

        needs_review = (
            final_score < self.review_threshold
            or (vision.legibility == "poor" and not best_has_independent_confirm)
            or (ai_conf < 0.75 and not best_has_independent_confirm)
        )

        return FacilityMatch(
            facility_id=best_fac["id"],
            canonical_name=best_fac["canonical_name"],
            region=best_fac["region"],
            score=final_score,
            legibility=vision.legibility,
            needs_review=needs_review,
            match_reason=best_reason,
        )

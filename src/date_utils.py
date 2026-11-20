"""
Утиліти валідації та форматування дат документів і назв папок для бухгалтерії/архіву.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import re
from typing import Optional

UKRAINIAN_MONTHS: dict[int, str] = {
    1: "Січень",
    2: "Лютий",
    3: "Березень",
    4: "Квітень",
    5: "Травень",
    6: "Червень",
    7: "Липень",
    8: "Серпень",
    9: "Вересень",
    10: "Жовтень",
    11: "Листопад",
    12: "Грудень",
}

DOCUMENT_TYPE_LABELS: dict[str, str] = {
    "act": "Акт",
    "daily_report": "Щоденний_звіт",
    "menu": "Меню",
    "application": "Заявка",
    "raw_materials_calc": "Розрахунок_сировини",
    "waybill_internal": "Накладна",
    "waybill_external": "ТТН",
    "receipt": "Чек",
    "food_photo": "Контрольна_порція",
    "other": "Документ",
}


def format_month_folder(dt: date) -> str:
    """Повертає назву папки місяця виду '2026-11 Листопад'."""
    month_name = UKRAINIAN_MONTHS.get(dt.month, f"{dt.month:02d}")
    return f"{dt.year:04d}-{dt.month:02d} {month_name}"


FACILITY_BOILERPLATE_RE = re.compile(
    r"Опорн(?:ий|ого)\s+заклад(?:у)?\s+загальної\s+середньої\s+освіти|"
    r"Опорн(?:ий|ого)\s+заклад(?:у)?|"
    r"Заклад(?:у)?\s+загальної\s+середньої\s+освіти|"
    r"ОЗЗСО|ЗЗСО",
    flags=re.IGNORECASE,
)

COUNCIL_SUFFIX_RE = re.compile(
    r"(?:Камінь\s*[-\u2013]\s*Каширської|Любешівської|Маневицької|Прилісненської|Сошичненської|"
    r"Красилівської|Теофіпольської|Богуславської|Іванківської|Тараканівської|Підгайцівської|"
    r"Колківської|Гостомельської|Ратнівської|Цуманської|Ківерцівської|Сарненської)"
    r"\s+(?:міської|селищної|сільської)\s+ради(?:\s+\w+\s+області)?",
    flags=re.IGNORECASE,
)

OBLAST_SUFFIX_RE = re.compile(
    r"\s+(?:Волинської|Рівненської|Хмельницької|Київської)\s+області",
    flags=re.IGNORECASE,
)


def strip_facility_boilerplate(canonical_name: str) -> str:
    """Прибирає типові канцелярські префікси та назви рад/областей із назви закладу."""
    name = (canonical_name or "").strip()
    name = FACILITY_BOILERPLATE_RE.sub("", name)
    name = COUNCIL_SUFFIX_RE.sub("", name)
    name = OBLAST_SUFFIX_RE.sub("", name)
    return name


def shorten_facility_dirname(canonical_name: str, max_len: int = 85) -> str:
    """
    Робить читабельну, коротку й безпечну для Windows назву папки закладу
    без технічних префіксів на кшталт 'Опорний заклад загальної середньої освіти'.
    """
    name = strip_facility_boilerplate(canonical_name)
    name = re.sub(r'[<>:"/\\|?*«»\'`]+', " ", name)
    name = re.sub(r"\s+", " ", name).strip(" ,.-_")
    if not name:
        name = re.sub(r'[<>:"/\\|?*]+', "_", (canonical_name or "").strip())
    return name[:max_len].strip(" .") or "Заклад"


def _parse_iso_or_dmy_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    text = raw.strip()
    m_iso = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
    if m_iso:
        try:
            return date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            return None
    m_dmy = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$", text)
    if m_dmy:
        year = int(m_dmy.group(3))
        if year < 100:
            year += 2000
        try:
            return date(year, int(m_dmy.group(2)), int(m_dmy.group(1)))
        except ValueError:
            return None
    return None


def validate_document_date(
    raw_date: Optional[str],
    received_at: datetime,
    *,
    max_past_days: int = 14,
    max_future_days: int = 1,
) -> tuple[str, str]:
    """
    Перевіряє розпізнану дату документа на адекватність щодо дати надходження фото.
    - Допускає документи за поточний/минулий тиждень (до max_past_days днів назад).
    - Якщо помилковий лише рік (напр., 2023/2024 замість 2026) — виправляє рік.
    - Якщо дата відсутня, нерозпізнана або помилковий місяць/майбутнє — підставляє дату надходження.
    Повертає кортеж (YYYY-MM-DD, reason).
    """
    ref_date = received_at.date()
    min_valid = ref_date - timedelta(days=max(1, max_past_days))
    max_valid = ref_date + timedelta(days=max(0, max_future_days))

    if not raw_date or not raw_date.strip():
        return ref_date.isoformat(), "fallback_received_missing"

    parsed = _parse_iso_or_dmy_date(raw_date)
    if parsed is None:
        return ref_date.isoformat(), f"fallback_received_unparsed:{raw_date}"

    if min_valid <= parsed <= max_valid:
        return parsed.isoformat(), "document_date"

    # Спроба виправити лише рік (коли ШІ поставив 2023/2024 або на бланку опечатка в році)
    for candidate_year in (ref_date.year, ref_date.year - 1):
        try:
            adjusted = parsed.replace(year=candidate_year)
        except ValueError:
            continue
        if min_valid <= adjusted <= max_valid:
            return adjusted.isoformat(), f"year_fixed:{parsed.isoformat()}"

    return ref_date.isoformat(), f"fallback_received_invalid:{parsed.isoformat()}"

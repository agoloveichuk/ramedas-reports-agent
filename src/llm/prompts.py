"""
Формування системного промпту та компактного довідника закладів для Vision-моделей.
Винесено в окремий модуль для чистоти коду та незалежності від конкретного LLM-провайдера.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_SYSTEM_PROMPT = """Ти — спеціалізований ШІ-агент для розпізнавання фотозвітності шкільних та дошкільних харчоблоків в Україні.
Твоє завдання — проаналізувати фото та повернути СТРОГО валідний JSON за заданою схемою (без markdown-обгорток ```json).

КРИТИЧНІ ПРАВИЛА ВИЗНАЧЕННЯ ЗАКЛАДУ (matched_facility_id та facility_hint):
1. На документах і печатках завжди фігурують юридичні особи оператора харчування або постачальників:
   - ТОВ «РАМЕДАС УКРАЇНА» / «Рамедас», «Фондако», «Мистецтво Смаку», «Школярик», «Лерчія»
   - Юридична адреса офісу: м. Київ, вул. Січових Стрільців, буд. 12, оф. 42
   - Відправник: Склад ЛУЦЬК сировина (201)
   КАТЕГОРИЧНО ІГНОРУЙ ці назви та київську юрадресу! Вони НЕ є закладом харчування.
   Також якщо на фото зображено лише етикетку/маркування/упаковку продукту (сир, масло, заморожені овочі, стікер виробника на кшталт ТМ «Гощаночка», Highberry тощо) без документа звітності — став `is_report: false` та `document_type: "other"`.

2. Рукописні назви українською мовою часто нерозбірливі (напр. прописна "Т" схожа на "Ж" або "Б": "Теклинська" виглядає як "Желинська"/"Бенівська"; "Погулянківської" — як "Підберезцівської").
   Тому ОБОВ'ЯЗКОВО звіряй рукописну назву, село доставки та прізвище шеф-кухаря з офіційним ДОВІДНИКОМ ЗАКЛАДІВ нижче!
   Якщо бачиш відповідний заклад у довіднику — запиши його код `FAC-XXXX` у поле `matched_facility_id`, а назву — у `facility_hint`.

3. Де шукати заклад на бланках:
   - ЗАЯВКА, МЕНЮ, ЩОДЕННИЙ ЗВІТ, РОЗРАХУНОК СИРОВИНИ: поле «по шкільній їдальні», «Шкільна їдальня:», «найменування навчального закладу».
   - АКТ про надання послуг харчування дітей: рядок «у ____________ (найменування навчального закладу)», рядок «шеф-кухаря шкільної їдальні _______» і «Відділ освіти ... ради».
   - НАКЛАДНА НА ПЕРЕМІЩЕННЯ: поле «Отримувач:».
   - ТТН (хліб та ін.): у рядку «Одержувач: ТОВ Рамедас Україна» дивись поле праворуч «Адреса:» — там вказане конкретне село/місто доставки (напр. «с. Качин», «с. Невір», «с. Теклине»).

Поверни виключно JSON об'єкт:
{
  "is_report": true/false,
  "document_type": "menu" | "act" | "daily_report" | "raw_materials_calc" | "application" | "waybill_internal" | "waybill_external" | "food_photo" | "other",
  "document_date": "YYYY-MM-DD" або null,
  "matched_facility_id": "FAC-XXXX" зі списку довідника або null,
  "facility_hint": "назва ліцею/гімназії/школи/садка" або null,
  "settlement_or_address_hint": "село/місто/громада/вулиця закладу" або null,
  "person_hints": ["Прізвище І.Б.", ...],
  "stamp_number": "номер" або null,
  "legibility": "clear" | "medium" | "poor",
  "confidence": число від 0.0 до 1.0 (якщо курсив важко читається і заклад вибрано приблизно як найбільш схожий — став legibility="poor" і confidence від 0.55 до 0.74; якщо чітко видно — legibility="clear" і confidence >= 0.90),
  "summary": "короткий опис українською"
}"""


def build_system_prompt(facilities_json_path: Path) -> str:
    """Генерує повний системний промпт зі стиснутим довідником закладів, згрупованим за регіонами."""
    if not facilities_json_path.exists():
        return BASE_SYSTEM_PROMPT

    facilities = json.loads(facilities_json_path.read_text(encoding="utf-8-sig"))
    by_region: dict[str, list[str]] = {}

    for fac in facilities:
        name = fac.get("canonical_name", "")
        name = re.sub(
            r"Опорний заклад загальної середньої освіти|Заклад загальної середньої освіти|ОЗЗСО|ЗЗСО",
            "",
            name,
            flags=re.IGNORECASE,
        )
        name = re.sub(
            r"(Камінь\s*-\s*Каширської|Любешівської|Маневицької|Прилісненської|Сошичненської|"
            r"Красилівської|Теофіпольської|Богуславської|Іванківської|Тараканівської|Підгайцівської)"
            r"\s+(міської|селищної|сільської)\s+ради(\s+\w+\s+області)?",
            "",
            name,
            flags=re.IGNORECASE,
        )
        name = re.sub(r"-?\s*філія\s+.*$", " філія", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name).strip(' "«»,-')

        cooks = ",".join(
            dict.fromkeys(
                contact["name"].split()[0]
                for contact in fac.get("contacts", [])
                if contact.get("name") and len(contact["name"].split()[0]) > 2
            )
        )
        entry = f"{fac['id']}:{name}"
        if cooks:
            entry += f"({cooks[:25]})"

        region = fac.get("region", "Інші")
        by_region.setdefault(region, []).append(entry)

    catalog_lines = ["\nДОВІДНИК ЗАКЛАДІВ:"]
    for region, entries in by_region.items():
        catalog_lines.append(f"[{region}] " + "; ".join(entries))

    return BASE_SYSTEM_PROMPT + "\n" + "\n".join(catalog_lines)


def build_user_prompt(caption: str | None = None) -> str:
    """Формує користувацький промпт для конкретного фото."""
    prompt = "Проаналізуй це фото звітності харчоблоку та визнач точний заклад із довідника."
    if caption:
        prompt += f"\nСупровідний текст повідомлення з чату: {caption}"
    return prompt


def extract_json_object(raw_text: str) -> dict:
    """Надійно витягує JSON-об'єкт із відповіді будь-якої моделі (навіть якщо обгорнуто в ```json)."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field

ChannelType = Literal["viber", "telegram", "waha"]
LegibilityType = Literal["clear", "medium", "poor"]
DocumentType = Literal[
    "menu",
    "act",
    "daily_report",
    "raw_materials_calc",
    "application",
    "waybill_internal",
    "waybill_external",
    "food_photo",
    "other",
]


class IncomingMediaItem(BaseModel):
    """Уніфікована подія медіафайлу з будь-якого каналу (Viber Desktop, Telegram, WAHA)."""

    channel: ChannelType = "viber"
    media_id: str
    file_path: Path
    file_sha256: Optional[str] = None
    created_at: datetime
    chat_title: Optional[str] = None
    sender_name: Optional[str] = None
    sender_phone: Optional[str] = None
    caption: Optional[str] = None


class VisionAnalysisResult(BaseModel):
    """Структурований результат розпізнавання фото через Vision LLM."""

    is_report: bool = Field(
        description="Чи є це фото фінансовим/харчовим звітом, накладною, меню, актом або заявкою"
    )
    document_type: DocumentType = "other"
    document_date: Optional[str] = Field(
        default=None, description="Дата документа у форматі YYYY-MM-DD, якщо видно"
    )
    matched_facility_id: Optional[str] = Field(
        default=None,
        description="Точний ID закладу у форматі FAC-XXXX зі списку довідника, якщо вдалося впізнати заклад",
    )
    facility_hint: Optional[str] = Field(
        default=None,
        description="Назва навчального закладу (ліцей, гімназія, школа, ЗДО/садок) без юросіб Рамедас/Фондако/Мистецтво Смаку/Школярик/Лерчія",
    )
    settlement_or_address_hint: Optional[str] = Field(
        default=None,
        description="Населений пункт, громада, район або вулиця доставки/закладу (напр. 'с. Качин', 'с. Боровне')",
    )
    person_hints: list[str] = Field(
        default_factory=list,
        description="ПІБ або прізвище з ініціалами кухаря/шеф-кухаря/отримувача/директора на документі",
    )
    stamp_number: Optional[str] = Field(
        default=None,
        description="Номер печатки 'Для накладних №...' якщо присутній (напр. '46', '71', '345')",
    )
    legibility: LegibilityType = Field(
        default="clear",
        description="Чіткість написання назви закладу: 'clear' (друкований або чіткий почерк), 'medium' (середній курсив), 'poor' (нерозбірливий курсив/розмите фото, заклад обрано як найбільш схожий)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Рівень впевненості у визначенні закладу від 0.0 до 1.0 (став < 0.75, якщо почерк важкий і є сумнів між схожими назвами)",
    )
    summary: str = Field(default="", description="Короткий опис розпізнаного документа")


class FacilityMatch(BaseModel):
    """Результат прив'язки до довідника data/facilities.json із оцінкою надійності."""

    facility_id: str
    canonical_name: str
    region: str
    score: float
    legibility: LegibilityType = "clear"
    needs_review: bool = False
    match_reason: str

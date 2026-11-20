"""
Утиліти для нормалізації EXIF-орієнтації та автоматичного повороту фото (0° / 90° / 180° / 270°)
через Pillow при відправці у Vision LLM та збереженні у storage/.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
from pathlib import Path
import re
import shutil
from typing import IO, Literal, cast

from PIL import Image, ImageOps

from src.logger import get_logger

log = get_logger("image_utils")

RotationDegrees = Literal[0, 90, 180, 270]

_EXIF_ORIENTATION_TAG = 0x0112
_SIDE_TO_ROTATION: dict[str, RotationDegrees] = {
    "top": 0,
    "up": 0,
    "normal": 0,
    "left": 90,
    "bottom": 180,
    "down": 180,
    "upside_down": 180,
    "right": 270,
}
_CANONICAL_ANGLES: frozenset[int] = frozenset((0, 90, 180, 270))


def normalize_rotation_degrees(value: object) -> RotationDegrees:
    """Нормалізує кут повороту за годинниковою стрілкою до 0, 90, 180 або 270."""
    if value is None or isinstance(value, bool):
        return 0

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return 0
        if cleaned in _SIDE_TO_ROTATION:
            return _SIDE_TO_ROTATION[cleaned]
        match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
        if not match:
            return 0
        num = float(match.group(0))
    elif isinstance(value, (int, float)):
        num = float(value)
    else:
        return 0

    snapped = (int(round(num / 90.0)) * 90) % 360
    return cast(RotationDegrees, snapped if snapped in _CANONICAL_ANGLES else 0)


def has_non_default_exif_orientation(image_path: Path) -> bool:
    """Перевіряє, чи має зображення тег EXIF Orientation, відмінний від 1 (нормальна орієнтація)."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            orientation = exif.get(_EXIF_ORIENTATION_TAG)
            return bool(orientation and orientation != 1)
    except Exception:
        return False


def rotate_pil_image_clockwise(img: Image.Image, rotation_degrees: int = 0) -> Image.Image:
    """
    Застосовує EXIF-орієнтацію (якщо є) та повертає зображення ЗА ГОДИННИКОВОЮ СТРІЛКОЮ
    на заданий кут (0, 90, 180 або 270 градусів) без втрати пропорцій.
    """
    oriented = ImageOps.exif_transpose(img)
    angle = normalize_rotation_degrees(rotation_degrees)

    # У Pillow операції ROTATE_90 / ROTATE_270 визначені ПРОТИ годинникової стрілки:
    # - 90° за годинниковою (шапка була зліва -> стає зверху) = ROTATE_270
    # - 180° (шапка була знизу -> стає зверху) = ROTATE_180
    # - 270° за годинниковою (шапка була справа -> стає зверху) = ROTATE_90
    if angle == 90:
        return oriented.transpose(Image.Transpose.ROTATE_270)
    if angle == 180:
        return oriented.transpose(Image.Transpose.ROTATE_180)
    if angle == 270:
        return oriented.transpose(Image.Transpose.ROTATE_90)
    return oriented


def _save_image_encoded(
    img: Image.Image,
    target: Path | IO[bytes],
    *,
    source_format: str,
    target_suffix: str = "",
    jpeg_subsampling: int | None = None,
) -> str:
    """Зберігає зображення у потрібний формат із конвертацією режиму для JPEG і повертає MIME-тип."""
    fmt = (source_format or "JPEG").upper()
    suf = target_suffix.lower()

    if fmt == "PNG" or suf == ".png":
        img.save(target, format="PNG", optimize=True)
        return "image/png"
    if fmt == "WEBP" or suf == ".webp":
        img.save(target, format="WEBP", quality=95)
        return "image/webp"

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    save_kwargs: dict[str, object] = {"format": "JPEG", "quality": 95, "optimize": True}
    if jpeg_subsampling is not None:
        save_kwargs["subsampling"] = jpeg_subsampling
    img.save(target, **save_kwargs)
    return "image/jpeg"


def encode_image_for_llm(
    image_path: Path,
    *,
    rotation_degrees: int = 0,
) -> tuple[str, str]:
    """
    Готує (mime_type, base64_data) для Vision LLM.
    Читає байти напряму, якщо не потрібен поворот або EXIF-транспозиція.
    """
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/jpeg"
    angle = normalize_rotation_degrees(rotation_degrees)

    if angle == 0 and not has_non_default_exif_orientation(image_path):
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return mime_type, encoded

    try:
        with Image.open(image_path) as img:
            rotated = rotate_pil_image_clockwise(img, angle)
            buf = io.BytesIO()
            mime_type = _save_image_encoded(
                rotated,
                buf,
                source_format=img.format or "JPEG",
                target_suffix=image_path.suffix,
            )
            return mime_type, base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        log.warning(
            "Не вдалося повернути %s у пам'яті (%s), відправляємо оригінал",
            image_path.name,
            exc,
        )
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return mime_type, encoded


def save_oriented_image(
    source_path: Path,
    target_path: Path,
    *,
    rotation_degrees: int = 0,
) -> RotationDegrees:
    """
    Копіює фото у target_path з виправленням EXIF-орієнтації та поворотом на rotation_degrees.
    Без потреби повороту виконує побайтове копіювання shutil.copy2.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    angle = normalize_rotation_degrees(rotation_degrees)
    needs_exif = has_non_default_exif_orientation(source_path)
    same_file = source_path.resolve() == target_path.resolve()

    if angle == 0 and not needs_exif:
        if not same_file:
            shutil.copy2(source_path, target_path)
        return 0

    try:
        src_stat = source_path.stat()
        with Image.open(source_path) as img:
            fmt = img.format or "JPEG"
            rotated = rotate_pil_image_clockwise(img, angle)
            rotated.load()

        _save_image_encoded(
            rotated,
            target_path,
            source_format=fmt,
            target_suffix=target_path.suffix,
            jpeg_subsampling=0,
        )
        if not same_file:
            shutil.copystat(source_path, target_path)
        else:
            os.utime(target_path, (src_stat.st_atime, src_stat.st_mtime))
        return angle
    except Exception as exc:
        log.warning(
            "Не вдалося повернути фото %s на %d° (%s), копіюємо без змін",
            source_path.name,
            angle,
            exc,
        )
        if not same_file:
            shutil.copy2(source_path, target_path)
        return 0

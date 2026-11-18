"""
Зворотна сумісність із модулем маршрутизації Vision-моделей (src.llm.router).
Уся логіка промптів, провайдерів (Gemini, OpenAI, Anthropic, OpenRouter) та hot-reload
рознесена у пакеті `src/llm/`.
"""

from __future__ import annotations

from src.llm.router import VisionModelRouter

GeminiVisionClient = VisionModelRouter

__all__ = ["GeminiVisionClient", "VisionModelRouter"]

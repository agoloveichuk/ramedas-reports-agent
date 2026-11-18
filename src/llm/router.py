"""
Маршрутизатор моделей (VisionModelRouter) із підтримкою:
1. Гарячого перезавантаження (Hot-Reload) config/llm_routing.json та .env прямо в рантаймі.
2. Автоматичного перемикання на резервні моделі/провайдери (Fallback Chain).
3. Тимчасового блокування нестабільних моделей (Circuit Breaker Cooldown), щоб не гальмувати чергу.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from dotenv import dotenv_values

from src.config import PROJECT_ROOT, Settings
from src.llm.prompts import build_system_prompt
from src.llm.providers import BaseVisionProvider, create_provider
from src.logger import get_logger
from src.models import VisionAnalysisResult

logger = get_logger("llm")


class VisionModelRouter:
    """Мультипровайдерний маршрутизатор із hot-reload та резервними моделями."""

    def __init__(self, settings: Settings, routing_config_path: Path | None = None) -> None:
        self.settings = settings
        self.routing_path = routing_config_path or (PROJECT_ROOT / "config" / "llm_routing.json")
        self.env_path = PROJECT_ROOT / ".env"

        self.system_prompt = build_system_prompt(settings.facilities_json_path)
        self._last_routing_mtime: float = 0.0
        self._last_env_mtime: float = 0.0

        self.cooldown_seconds: float = 60.0
        self.max_attempts_per_model: int = 2
        self.active_providers: list[BaseVisionProvider] = []
        self._cooldown_until: dict[str, float] = {}
        self.last_used_route: str = ""

        self.reload_if_changed(force=True)

    def reload_if_changed(self, *, force: bool = False) -> bool:
        """
        Перевіряє час модифікації config/llm_routing.json та .env.
        Якщо файли змінилися в рантаймі — миттєво оновлює список провайдерів і ключів без рестарту.
        """
        routing_mtime = self.routing_path.stat().st_mtime if self.routing_path.exists() else 0.0
        env_mtime = self.env_path.stat().st_mtime if self.env_path.exists() else 0.0

        if not force and routing_mtime == self._last_routing_mtime and env_mtime == self._last_env_mtime:
            return False

        self._last_routing_mtime = routing_mtime
        self._last_env_mtime = env_mtime

        env_vars = {**os.environ}
        if self.env_path.exists():
            for k, v in dotenv_values(self.env_path).items():
                if v is not None:
                    env_vars[k] = v.strip()

        providers: list[BaseVisionProvider] = []
        if self.routing_path.exists():
            cfg = json.loads(self.routing_path.read_text(encoding="utf-8-sig"))
            self.cooldown_seconds = float(cfg.get("cooldown_seconds_on_failure", 60.0))
            self.max_attempts_per_model = int(cfg.get("max_attempts_per_model", 2))

            for route in cfg.get("routes", []):
                if not route.get("enabled", True):
                    continue
                key_env = route.get("api_key_env", "GEMINI_API_KEY")
                api_key = env_vars.get(key_env, "").strip()
                # Локальні сервери (Ollama/vLLM) можуть працювати без ключа
                if not api_key and route.get("provider") not in ("ollama", "vllm"):
                    continue
                providers.append(create_provider(route, api_key or "local"))

        # Якщо конфіг відсутній — використовуємо дефолтну модель із Settings
        if not providers and self.settings.gemini_api_key:
            providers.append(
                create_provider(
                    {
                        "name": "default_gemini",
                        "provider": "gemini",
                        "model": self.settings.gemini_model,
                    },
                    self.settings.gemini_api_key,
                )
            )

        self.active_providers = providers
        return True

    def analyze_image(self, image_path: Path, caption: str | None = None) -> VisionAnalysisResult:
        """
        Виконує аналіз фото з автоматичним перебором ланцюжка моделей (Primary -> Backup 1 -> Backup 2).
        Якщо модель повертає 429/503/помилку — переводить її в cooldown і миттєво бере наступну.
        """
        if self.reload_if_changed():
            routes_str = " -> ".join(f"{p.name}[{p.model}]" for p in self.active_providers)
            logger.info("[Hot-Reload] Оновлено ланцюжок моделей: %s", routes_str)

        if not self.active_providers:
            raise RuntimeError(
                "Не знайдено жодного активного LLM-маршруту з ключем API. "
                "Перевірте .env та config/llm_routing.json"
            )

        now = time.time()
        # Спочатку пробуємо провайдери, які не знаходяться в cooldown
        available = [p for p in self.active_providers if self._cooldown_until.get(p.name, 0.0) <= now]
        if not available:
            # Якщо всі моделі потрапили в cooldown — скидаємо таймер для першої та робимо паузу
            available = list(self.active_providers)

        errors: list[str] = []
        for provider in available:
            for attempt in range(1, self.max_attempts_per_model + 1):
                try:
                    result = provider.analyze(image_path, self.system_prompt, caption=caption)
                    self.last_used_route = f"{provider.provider_type}:{provider.model}"
                    return result
                except Exception as exc:
                    err_msg = f"{provider.name} ({provider.model}) спроба #{attempt}: {exc}"
                    errors.append(err_msg)
                    if attempt < self.max_attempts_per_model:
                        time.sleep(1.5 * attempt)
                    else:
                        # Переводимо проблемний маршрут у тимчасовий cooldown і перемикаємось на бекап
                        self._cooldown_until[provider.name] = time.time() + self.cooldown_seconds
                        logger.warning(
                            "[Fallback] Маршрут %s недоступний (%s), cooldown %.0fс -> перемикання на резервну модель...",
                            provider.name,
                            exc.__class__.__name__,
                            self.cooldown_seconds,
                        )

        raise RuntimeError(
            "Усі основні та резервні моделі не змогли обробити фото:\n  - " + "\n  - ".join(errors)
        )

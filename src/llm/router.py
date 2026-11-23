"""
Маршрутизатор моделей (VisionModelRouter) із підтримкою:
1. Гарячого перезавантаження (Hot-Reload) config/llm_routing.json та .env прямо в рантаймі.
2. Автоматичного перемикання на резервні моделі/провайдери (Fallback Chain).
3. Тимчасового блокування нестабільних моделей (Circuit Breaker Cooldown), щоб не гальмувати чергу.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import time
from dotenv import dotenv_values
import httpx

from src.config import PROJECT_ROOT, Settings
from src.llm.prompts import build_system_prompt
from src.llm.providers import BaseVisionProvider, create_provider
from src.logger import get_logger
from src.models import VisionAnalysisResult

logger = get_logger("llm")


def _sanitize_error_text(text: str) -> str:
    """Маскує API-ключі в URL винятків перед записом у логи, БД або алерти."""
    return re.sub(r"([?&](?:key|api_key|token)=)[^&\s'\"]+", r"\1***", text)


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
        self.escalation_providers: list[BaseVisionProvider] = []
        self._cooldown_until: dict[str, float] = {}
        self.last_used_route: str = ""
        self.last_used_model: str = ""

        self.reload_if_changed(force=True)

    @property
    def has_escalation(self) -> bool:
        self.reload_if_changed()
        return bool(self.escalation_providers)

    def has_escalation_for(self, exclude_models: set[str] | None = None) -> bool:
        self.reload_if_changed()
        excluded = exclude_models or set()
        return any(p.model not in excluded for p in self.escalation_providers)

    @staticmethod
    def _build_providers(
        routes: list[dict],
        env_vars: dict[str, str],
    ) -> list[BaseVisionProvider]:
        providers: list[BaseVisionProvider] = []
        for route in routes:
            if not route.get("enabled", True):
                continue
            key_env = route.get("api_key_env", "GEMINI_API_KEY")
            api_key = env_vars.get(key_env, "").strip()
            # Локальні сервери (Ollama/vLLM) можуть працювати без ключа
            if not api_key and route.get("provider") not in ("ollama", "vllm"):
                continue
            providers.append(create_provider(route, api_key or "local"))
        return providers

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
        escalation: list[BaseVisionProvider] = []
        if self.routing_path.exists():
            cfg = json.loads(self.routing_path.read_text(encoding="utf-8-sig"))
            self.cooldown_seconds = float(cfg.get("cooldown_seconds_on_failure", 60.0))
            self.max_attempts_per_model = int(cfg.get("max_attempts_per_model", 2))
            providers = self._build_providers(cfg.get("routes", []), env_vars)
            escalation = self._build_providers(cfg.get("escalation_routes", []), env_vars)

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
        self.escalation_providers = escalation
        return True

    def analyze_image(
        self,
        image_path: Path,
        caption: str | None = None,
        *,
        rotation_degrees: int = 0,
        reference_date: date | datetime | None = None,
        escalate: bool = False,
        exclude_models: set[str] | None = None,
    ) -> VisionAnalysisResult:
        """
        Виконує аналіз фото з автоматичним перебором ланцюжка моделей (Primary -> Backup 1 -> Backup 2).
        Якщо escalate=True — використовує сильніші моделі з escalation_routes (лише якщо їхні API-ключі задані).
        """
        if self.reload_if_changed():
            routes_str = " -> ".join(f"{p.name}[{p.model}]" for p in self.active_providers)
            logger.info("[Hot-Reload] Оновлено ланцюжок моделей: %s", routes_str)

        target_providers = self.escalation_providers if escalate else self.active_providers
        if exclude_models:
            target_providers = [p for p in target_providers if p.model not in exclude_models]
        if not target_providers:
            if escalate:
                raise RuntimeError(
                    "Ескалація на сильнішу модель недоступна: не задано жодного API-ключа для escalation_routes"
                )
            raise RuntimeError(
                "Не знайдено жодного активного LLM-маршруту з ключем API. "
                "Перевірте .env та config/llm_routing.json"
            )

        now = time.time()
        ready = [p for p in target_providers if self._cooldown_until.get(p.name, 0.0) <= now]
        cooling = sorted(
            [p for p in target_providers if p not in ready],
            key=lambda p: self._cooldown_until.get(p.name, 0.0),
        )

        errors: list[str] = []

        def _try_providers(providers: list[BaseVisionProvider]) -> VisionAnalysisResult | None:
            for provider in providers:
                for attempt in range(1, self.max_attempts_per_model + 1):
                    try:
                        result = provider.analyze(
                            image_path,
                            self.system_prompt,
                            caption=caption,
                            rotation_degrees=rotation_degrees,
                            reference_date=reference_date,
                        )
                        self.last_used_route = f"{provider.provider_type}:{provider.model}"
                        self.last_used_model = provider.model
                        return result
                    except Exception as exc:
                        safe_exc = _sanitize_error_text(str(exc))
                        errors.append(f"{provider.name} ({provider.model}) спроба #{attempt}: {safe_exc}")
                        is_permanent = (
                            isinstance(exc, httpx.HTTPStatusError)
                            and exc.response.status_code in (401, 403, 404)
                        )
                        if is_permanent:
                            self._cooldown_until[provider.name] = time.time() + 3600.0
                            logger.warning(
                                "[Fallback] Маршрут %s повернув HTTP %d (постійна помилка), вимкнено на 60 хв -> перемикання...",
                                provider.name,
                                exc.response.status_code,
                            )
                            break
                        if attempt < self.max_attempts_per_model:
                            time.sleep(1.5 * attempt)
                        else:
                            self._cooldown_until[provider.name] = time.time() + self.cooldown_seconds
                            logger.warning(
                                "[Fallback] Маршрут %s недоступний (%s), cooldown %.0fс -> перемикання на резервну модель...",
                                provider.name,
                                exc.__class__.__name__,
                                self.cooldown_seconds,
                            )
            return None

        if ready:
            res = _try_providers(ready)
            if res is not None:
                return res

        if cooling:
            # Чекаємо лише на тимчасовий cooldown (<= 120с); маршрути з 404/401/403 (3600с) не гальмують чергу
            cur_now = time.time()
            transient_cooling = [
                p for p in cooling if (self._cooldown_until.get(p.name, cur_now) - cur_now) <= 120.0
            ]
            if transient_cooling:
                soonest = min(self._cooldown_until.get(p.name, cur_now) for p in transient_cooling)
                wait_s = max(1.0, min(soonest - cur_now, 15.0))
                logger.warning("Усі LLM-провайдери в cooldown — пауза %.1fс до найближчої спроби", wait_s)
                time.sleep(wait_s)
                res = _try_providers(transient_cooling)
                if res is not None:
                    return res

        raise RuntimeError(
            "Усі основні та резервні моделі не змогли обробити фото:\n  - " + "\n  - ".join(errors)
        )

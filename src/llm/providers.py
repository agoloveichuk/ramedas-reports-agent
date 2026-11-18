from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import mimetypes
from pathlib import Path
import httpx

from src.llm.prompts import build_user_prompt, extract_json_object
from src.models import VisionAnalysisResult


def _encode_image(image_path: Path) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return mime_type, encoded


class BaseVisionProvider(ABC):
    def __init__(
        self,
        *,
        name: str,
        provider_type: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 35.0,
    ) -> None:
        self.name = name
        self.provider_type = provider_type
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def analyze(
        self,
        image_path: Path,
        system_prompt: str,
        caption: str | None = None,
    ) -> VisionAnalysisResult:
        raise NotImplementedError


class GeminiVisionProvider(BaseVisionProvider):
    def analyze(
        self,
        image_path: Path,
        system_prompt: str,
        caption: str | None = None,
    ) -> VisionAnalysisResult:
        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        base = (self.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        endpoint = f"{base}/{model_path}:generateContent"

        mime_type, b64_data = _encode_image(image_path)
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                        {"text": build_user_prompt(caption)},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(endpoint, params={"key": self.api_key}, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )
        return VisionAnalysisResult.model_validate(extract_json_object(raw_text))


class OpenAICompatibleVisionProvider(BaseVisionProvider):
    """Підходить для OpenAI, OpenRouter, Groq, локальних Ollama / vLLM (/v1/chat/completions)."""

    def analyze(
        self,
        image_path: Path,
        system_prompt: str,
        caption: str | None = None,
    ) -> VisionAnalysisResult:
        base = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{base}/chat/completions"

        mime_type, b64_data = _encode_image(image_path)
        data_uri = f"data:{mime_type};base64,{b64_data}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_prompt(caption)},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return VisionAnalysisResult.model_validate(extract_json_object(raw_text))


class AnthropicVisionProvider(BaseVisionProvider):
    def analyze(
        self,
        image_path: Path,
        system_prompt: str,
        caption: str | None = None,
    ) -> VisionAnalysisResult:
        base = (self.base_url or "https://api.anthropic.com/v1").rstrip("/")
        endpoint = f"{base}/messages"

        mime_type, b64_data = _encode_image(image_path)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": build_user_prompt(caption)},
                    ],
                }
            ],
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_text = data.get("content", [{}])[0].get("text", "{}")
        return VisionAnalysisResult.model_validate(extract_json_object(raw_text))


def create_provider(route_cfg: dict, api_key: str) -> BaseVisionProvider:
    provider_type = route_cfg.get("provider", "gemini").lower()
    kwargs = {
        "name": route_cfg.get("name", f"{provider_type}:{route_cfg.get('model')}"),
        "provider_type": provider_type,
        "model": route_cfg["model"],
        "api_key": api_key,
        "base_url": route_cfg.get("base_url"),
        "timeout_seconds": float(route_cfg.get("timeout_seconds", 35.0)),
    }

    if provider_type == "gemini":
        return GeminiVisionProvider(**kwargs)
    if provider_type in ("openai", "openrouter", "openai_compatible", "ollama", "vllm"):
        return OpenAICompatibleVisionProvider(**kwargs)
    if provider_type == "anthropic":
        return AnthropicVisionProvider(**kwargs)

    raise ValueError(f"Unsupported provider type: {provider_type}")

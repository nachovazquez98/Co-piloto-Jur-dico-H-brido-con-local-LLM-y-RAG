"""Clientes de LLM intercambiables.

- `ollama`: modelos locales servidos por Ollama (por defecto `llama3.1:8b`).
- `openai`: cualquier API compatible con OpenAI (`/v1/chat/completions`): OpenAI, Groq,
  OpenRouter, LM Studio, vLLM, llama.cpp server, etc. Basta con cambiar `base_url`,
  `model` y `api_key`.

Para añadir otro proveedor, crea una subclase de `LLMClient` y regístrala en `PROVIDERS`.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

from .config import LLMSettings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings

    def generate(self, prompt: str) -> str:
        """Devuelve la respuesta completa (sin alterar mayúsculas ni formato Markdown)."""
        return "".join(self.stream(prompt)).strip()

    def stream(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []

    def health(self) -> tuple[bool, str]:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def _url(self, path: str) -> str:
        return self.settings.base_url.rstrip("/") + path

    def stream(self, prompt: str) -> Iterator[str]:
        s = self.settings
        payload = {
            "model": s.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": s.temperature, "num_predict": s.max_tokens, "num_ctx": s.num_ctx},
        }
        try:
            with requests.post(self._url("/api/chat"), json=payload, stream=True, timeout=s.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMError(chunk["error"])
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except requests.RequestException as e:
            raise LLMError(f"No se pudo conectar con Ollama en {s.base_url}: {e}") from e

    def list_models(self) -> list[str]:
        try:
            r = requests.get(self._url("/api/tags"), timeout=5)
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))
        except requests.RequestException:
            return []

    def health(self) -> tuple[bool, str]:
        models = self.list_models()
        if not models:
            return False, f"Ollama no responde en {self.settings.base_url} (¿ejecutaste `ollama serve`?)"
        if self.settings.model not in models:
            return False, f"El modelo '{self.settings.model}' no está descargado. Ejecuta `ollama pull {self.settings.model}`."
        return True, f"Ollama activo · {self.settings.model}"


class OpenAICompatibleClient(LLMClient):
    def _url(self, path: str) -> str:
        base = self.settings.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return base + path

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.settings.api_key:
            h["Authorization"] = f"Bearer {self.settings.api_key}"
        return h

    def stream(self, prompt: str) -> Iterator[str]:
        s = self.settings
        payload = {
            "model": s.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": s.temperature,
            "max_tokens": s.max_tokens,
            "stream": True,
        }
        try:
            with requests.post(self._url("/chat/completions"), json=payload, headers=self._headers(),
                               stream=True, timeout=s.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or [{}]
                    token = choices[0].get("delta", {}).get("content")
                    if token:
                        yield token
        except requests.RequestException as e:
            raise LLMError(f"Error llamando a la API en {s.base_url}: {e}") from e

    def list_models(self) -> list[str]:
        try:
            r = requests.get(self._url("/models"), headers=self._headers(), timeout=10)
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []))
        except (requests.RequestException, ValueError, KeyError):
            return []

    def health(self) -> tuple[bool, str]:
        models = self.list_models()
        if models:
            return True, f"API activa · {self.settings.model}"
        return False, f"No se pudo listar modelos en {self.settings.base_url} (revisa URL y API key)"


PROVIDERS: dict[str, tuple[str, type[LLMClient]]] = {
    "ollama": ("Ollama (local)", OllamaClient),
    "openai": ("API compatible con OpenAI (remota o local)", OpenAICompatibleClient),
}

PROVIDER_DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
}


def get_client(settings: LLMSettings) -> LLMClient:
    try:
        return PROVIDERS[settings.provider][1](settings)
    except KeyError as e:
        raise LLMError(f"Proveedor de LLM desconocido: {settings.provider}") from e

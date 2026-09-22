"""
Клиент языковой модели.
тонкая обёртка над HTTP API без внешних зависимостей
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Optional

from ..data.tags import load_config

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_ENV_FILE = ".env"


def load_api_key(env_name: str = "GEMINI_API_KEY") -> Optional[str]:
    """
    Ключ из переменной окружения, иначе из .env в корне проекта.
    """
    key = os.environ.get(env_name)
    if key:
        return key.strip()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, _ENV_FILE)
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == env_name:
                return value.strip().strip('"').strip("'") or None
    return None


def _ssl_context() -> Optional[ssl.SSLContext]:
    """
    Корневые сертификаты из certifi.

    Сборки Python с python.org на macOS не используют системное хранилище
    сертификатов, и запрос падает с CERTIFICATE_VERIFY_FAILED, хотя curl
    к тому же адресу работает. Явный cafile снимает вопрос на любой машине.
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


class GeminiClient:
    """Вызов Gemini через REST. Единственный метод - complete()."""
    def __init__(self, api_key: str, cfg: Optional[dict] = None):
        cfg = cfg or load_config("llm")
        self.api_key = api_key
        self.model = cfg.get("model", "gemini-3.6-flash")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_output_tokens = int(cfg.get("max_output_tokens", 2500))
        self.thinking_level = cfg.get("thinking_level")
        self.timeout = float(cfg.get("timeout_s", 30))

    # ------------------------------------------------------------------
    def complete(self, system: str, user: str) -> Optional[str]:
        """Текст ответа модели либо None, если ответа получить не удалось."""
        generation: dict = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
        }
        if self.thinking_level:
            # Лимит токенов общий для размышления и ответа. Без ограничения
            # размышление съедает бюджет, и ответ обрывается на полуслове.
            generation["thinkingConfig"] = {"thinkingLevel": self.thinking_level}

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        request = urllib.request.Request(
            _ENDPOINT.format(model=self.model),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=_ssl_context()
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

        return _extract_text(body)


def _extract_text(body: dict) -> Optional[str]:
    """
    Из ответа Gemini берём только части с текстом.
    """
    candidates = body.get("candidates") or []
    if not candidates:
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    chunks = [p["text"] for p in parts if isinstance(p.get("text"), str)]
    text = "\n".join(chunks).strip()
    return text or None


def load_client(cfg: Optional[dict] = None):
    """
    Готовый клиент либо None, если ключа нет.

    None - нормальный режим работы, а не ошибка: демо и тесты
    обязаны проходить на машине без ключа и без сети.
    """
    cfg = cfg or load_config("llm")
    key = load_api_key(cfg.get("api_key_env", "GEMINI_API_KEY"))
    if not key:
        return None
    return GeminiClient(key, cfg)

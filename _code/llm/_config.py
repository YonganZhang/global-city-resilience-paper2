"""LLM gateway configuration. All endpoints and credentials come from environment variables."""
from __future__ import annotations

import os
from typing import Dict, List, Optional


def _env_list(name: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


BASE_URL = os.getenv("LLM_BASE_URL", "")  # OpenAI-compatible gateway serving the council models
API_KEY = os.getenv("LLM_API_KEY", "")
API_KEY_POOL: List[str] = _env_list("LLM_API_KEY_POOL") or ([API_KEY] if API_KEY else [])
MODEL = os.getenv("LLM_DEFAULT_MODEL", "deepseek-chat")
TavilyAPI = os.getenv("TAVILY_API_KEY", "")  # optional; web retrieval is not used in the full run
TAVILY_KEYS: List[str] = _env_list("TAVILY_API_KEYS") or ([TavilyAPI] if TavilyAPI else [])
FALLBACK_MODELS: List[str] = _env_list("LLM_FALLBACK_MODELS")
TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))
DEBUG: bool = os.getenv("LLM_DEBUG", "0") == "1"
LOG_DIR: str = os.getenv("LLM_LOG_DIR", "llm_logs")


def _proxy_dict() -> Optional[Dict[str, str]]:
    http_proxy, https_proxy = os.getenv("HTTP_PROXY", "").strip(), os.getenv("HTTPS_PROXY", "").strip()
    return {"http": http_proxy or https_proxy, "https": https_proxy or http_proxy} if (http_proxy or https_proxy) else {}


PROXIES = _proxy_dict()
WEB_MODE: str = os.getenv("LLM_WEB_MODE", "never").lower()
WEB_ALLOWED_DOMAINS: List[str] = _env_list("LLM_WEB_ALLOWED_DOMAINS")

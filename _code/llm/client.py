# llm_client.py
# A simple, robust client:
# - Always uses /v1/chat/completions for generation
# - Optional true web search via Tavily (no /v1/responses at all)

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# V4 (P3.5.k 2026-05-17): _config.py 在同目录 _code/llm/, 无需反向 sys.path

import requests
from requests.exceptions import SSLError, ConnectionError, ReadTimeout

from _config import (
    BASE_URL, API_KEY, MODEL, FALLBACK_MODELS, PROXIES, TIMEOUT, DEBUG, LOG_DIR,
    WEB_MODE, WEB_ALLOWED_DOMAINS  # WEB_ALLOWED_DOMAINS kept for compatibility; Tavily searches don't use it
)

# 2026-06-02 用户拍板: pipeline 限流基础设施 — adaptive throttle.
# 所有 LLM call 走全局 throttle, 撞 429 自动 backoff + 降速 + 熔断.
# 区分 速率限 vs 额度耗尽 (AccountExhaustedError).
from adaptive_throttle import GLOBAL_THROTTLE, AccountExhaustedError

ALLOW_PAID_ENV = "GCIRI_ALLOW_PAID_LLM"

# ── 多 API Key 轮询（分散限流） ──
import itertools as _itertools
try:
    from _config import API_KEY_POOL
    if API_KEY_POOL and len(API_KEY_POOL) > 1:
        _key_cycle = _itertools.cycle(API_KEY_POOL)
        _KEY_POOL_SIZE = len(API_KEY_POOL)
    else:
        _key_cycle = None
        _KEY_POOL_SIZE = 1
except ImportError:
    _key_cycle = None
    _KEY_POOL_SIZE = 1

import threading as _threading
_key_lock = _threading.Lock()

def _next_api_key() -> str:
    """Thread-safe round-robin key selection from pool."""
    if _key_cycle is None:
        return API_KEY
    with _key_lock:
        return next(_key_cycle)

# ── 智能代理选择：gateway API 直连，其他走系统代理 ──
def _smart_proxies(url: str) -> dict:
    """gateway API 直连（绕过 Clash），其他请求走系统代理。"""
    if any(host and host in url for host in os.getenv("LLM_DIRECT_HOSTS", "").split(",")):
        return {}  # 直连，不走代理
    return PROXIES

# ------------------------------
# utils
# ------------------------------
def _mask(s: str, keep: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep)

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _completion_token_budget(model_name: str, requested: int) -> int:
    """Model-specific safety caps for hidden reasoning-token spend.

    Pipeline stages pass a generous 16k ceiling because Gemini/other models used
    to truncate. GPT-5-nano bills hidden reasoning tokens inside that same
    ceiling, so keep a lower default unless explicitly overridden.
    """
    if model_name.startswith("gpt-5-nano"):
        cap = _env_int("LLM_GPT5_NANO_MAX_COMPLETION_TOKENS", 6000)
        return min(requested, cap)
    return requested

def _reasoning_effort_for(model_name: str) -> Optional[str]:
    if model_name.startswith("gpt-5-nano"):
        value = os.getenv("LLM_GPT5_NANO_REASONING_EFFORT", "minimal").strip().lower()
        if value in ("", "off", "false", "none"):
            return None
        return value
    return None

def _dump_text(filename: str, text: str) -> str:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    p = Path(LOG_DIR) / filename
    p.write_text(text, encoding="utf-8")
    return str(p)

def _dump_json(filename: str, obj: Any) -> str:
    return _dump_text(filename, json.dumps(obj, ensure_ascii=False, indent=2))

def _make_trace_id(tag: str = "") -> str:
    base = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    return f"{tag}_{base}" if tag else base

def _require_config() -> None:
    if not BASE_URL:
        raise RuntimeError("LLM_BASE_URL 为空。")
    if not API_KEY:
        raise RuntimeError("LLM_API_KEY 为空。")

def _join(base: str, path: str) -> str:
    base = base.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path

def _infer_endpoints() -> Dict[str, str]:
    b = BASE_URL.rstrip("/")
    return {
        "models_url": _join(b, "/models"),
        "chat_url": _join(b, "/chat/completions"),
    }

def _post_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    trace_id: str,
    retries: int = 3,
) -> requests.Response:
    last = None
    for i in range(retries + 1):
        try:
            return requests.post(url, headers=headers, json=payload, timeout=TIMEOUT, proxies=_smart_proxies(url))
        except (SSLError, ConnectionError, ReadTimeout, OSError) as e:
            last = e
            if i == retries:
                raise
            sleep = (2 ** i) + random.uniform(0, 0.3)
            if DEBUG:
                print(f"[LLM][RETRY] trace_id={trace_id} i={i+1}/{retries} err={repr(e)} sleep={sleep:.2f}s")
            time.sleep(sleep)
    raise last  # pragma: no cover

def _get_with_retry(url: str, headers: Dict[str, str], trace_id: str, retries: int = 3) -> requests.Response:
    last = None
    for i in range(retries + 1):
        try:
            return requests.get(url, headers=headers, timeout=TIMEOUT, proxies=_smart_proxies(url))
        except (SSLError, ConnectionError, ReadTimeout, OSError) as e:
            last = e
            if i == retries:
                raise
            sleep = (2 ** i) + random.uniform(0, 0.3)
            if DEBUG:
                print(f"[LLM][RETRY][GET] trace_id={trace_id} i={i+1}/{retries} err={repr(e)} sleep={sleep:.2f}s")
            time.sleep(sleep)
    raise last  # pragma: no cover

def _print_full_or_hint(label: str, obj: Any, max_chars: int = 12000) -> None:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = repr(obj)

    if len(s) <= max_chars:
        print(f"[{label}][FULL]\n{s}")
    else:
        print(f"[{label}] response too large ({len(s)} chars). Full content is saved in llm_logs raw.json.")
        print(f"[{label}][HEAD]\n{s[:4000]}")
        print(f"[{label}][TAIL]\n{s[-4000:]}")

def _dump_debug_artifacts(trace_id: str, data: Any) -> None:
    """报错/解析失败时尽量把返回结构+全部内容打印 & 落盘"""
    try:
        if isinstance(data, dict):
            top_keys = list(data.keys())
            print(f"[LLM][DEBUG] top_keys={top_keys}")
            if isinstance(data.get("choices"), list) and data["choices"]:
                _dump_json(f"{trace_id}_choices0.json", data["choices"][0])
                c0 = data["choices"][0]
                if isinstance(c0, dict) and isinstance(c0.get("message"), dict):
                    _dump_json(f"{trace_id}_message_content.json", c0["message"].get("content"))
    except Exception:
        pass
    _print_full_or_hint("LLM_RAW_RESPONSE", data)

# ------------------------------
# extractors
# ------------------------------
def _extract_text_from_content(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content["text"]
        return None
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            t = _extract_text_from_content(p)
            if isinstance(t, str) and t.strip():
                parts.append(t)
        return "\n".join(parts).strip() if parts else None
    return None

def _extract_text_from_chat_completions(data: Dict[str, Any]) -> Optional[str]:
    choices = data.get("choices")
    if not (isinstance(choices, list) and choices):
        return None
    c0 = choices[0]
    if not isinstance(c0, dict):
        return None
    msg = c0.get("message")
    if not isinstance(msg, dict):
        return None
    return _extract_text_from_content(msg.get("content"))

def _is_empty_length_completion(data: Dict[str, Any]) -> bool:
    """
    关键模式：
    - choices[0].message.content == ""
    - finish_reason == "length"
    - completion_tokens_details.reasoning_tokens > 0 且 text_tokens == 0（部分网关有）
    """
    try:
        choices = data.get("choices")
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
            return False
        c0 = choices[0]
        msg = c0.get("message", {})
        content = msg.get("content")
        finish = c0.get("finish_reason")

        if not (isinstance(content, str) and content == "" and finish == "length"):
            return False

        u = data.get("usage", {})
        d = u.get("completion_tokens_details", {}) if isinstance(u, dict) else {}
        r = d.get("reasoning_tokens", None)
        t = d.get("text_tokens", None)

        if r is None or t is None:
            return True
        return (isinstance(r, int) and r > 0) and (isinstance(t, int) and t == 0)
    except Exception:
        return False

# ------------------------------
# public: list models
# ------------------------------
def load_model_list(trace_tag: str = "") -> Optional[Dict[str, Any]]:
    _require_config()
    eps = _infer_endpoints()
    url = eps["models_url"]
    trace_id = _make_trace_id(trace_tag or "list_models")
    _cur_key = _next_api_key()
    headers = {"Authorization": f"Bearer {_cur_key}"}

    if DEBUG:
        print("\n" + "=" * 80)
        print(f"[LLM][MODELS][REQ] trace_id={trace_id} url={url} api_key={_mask(_cur_key)}")

    try:
        t0 = time.time()
        resp = _get_with_retry(url, headers, trace_id, retries=3)
        dt = time.time() - t0
        if DEBUG:
            print(f"[LLM][MODELS][RESP] trace_id={trace_id} status={resp.status_code} time={dt:.2f}s")

        if not resp.ok:
            _dump_text(f"{trace_id}_models_http_{resp.status_code}.txt", resp.text or "")
            return None

        data = resp.json()
        _dump_json(f"{trace_id}_models.json", data)
        if DEBUG and isinstance(data, dict):
            print(f"[LLM][MODELS] top_keys={list(data.keys())}")
        return data
    except Exception as e:
        if DEBUG:
            print(f"[LLM][MODELS][EXC] trace_id={trace_id} err={repr(e)}")
        return None

# ------------------------------
# Tavily helpers (optional)
# ------------------------------
def _infer_search_query_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Infer a compact search query from the last user message.
    Special-case for your city scoring prompt (国家/城市).
    """
    if not messages:
        return ""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = (m.get("content") or "")
            break
    if not last_user.strip():
        return ""

    country = ""
    city = ""
    m_country = re.search(r"国家：\s*([^\n(]+)", last_user)
    if m_country:
        country = (m_country.group(1) or "").strip()
    m_city = re.search(r"城市：\s*([^\n]+)", last_user)
    if m_city:
        city = (m_city.group(1) or "").strip()

    if city:
        base = f"{city} {country}".strip()
        return f"{base} 人口 GDP 工业 高校 科研 机场 港口 铁路枢纽 地铁 高铁 基础设施 公共设施 洪涝 地震 台风 应急 预案 统计 报告"
    return re.sub(r"\s+", " ", last_user.strip())[:120]

def _inject_web_context(messages: List[Dict[str, str]], context: str) -> List[Dict[str, str]]:
    """
    Inject Tavily web context into the last user message.
    If prompt contains marker '现在开始输出JSON：', insert before it to preserve strict JSON instructions.
    """
    if not (isinstance(context, str) and context.strip()):
        return messages
    new_msgs = [dict(m) for m in messages]
    for i in range(len(new_msgs) - 1, -1, -1):
        if new_msgs[i].get("role") != "user":
            continue
        c = new_msgs[i].get("content", "") or ""
        marker = "现在开始输出JSON："
        block = f"\n\n=== 多源联网搜索结果（可引用链接，优先官方/最新）===\n{context}\n=== 搜索结果结束 ===\n\n"
        if marker in c:
            new_msgs[i]["content"] = c.replace(marker, block + marker, 1)
        else:
            new_msgs[i]["content"] = c + block
        break
    return new_msgs

def _tavily_context_for(messages: List[Dict[str, str]], trace_id: str) -> str:
    """
    Performs multi-source search via Tavily and returns formatted context text.
    Requires tavily_websearch.py in the same project.
    """
    # V27.3.20 P4 修 (subagent 审): V4 P3.5.k 重命名后真函数叫 run_quick_web_context (动词白名单). 老 quick_web_context 不存在.
    from tavily import run_quick_web_context  # local module
    q = _infer_search_query_from_messages(messages)
    ctx = run_quick_web_context(q)
    if DEBUG:
        _dump_text(f"{trace_id}_tavily_context.txt", ctx)
        print(f"[LLM][TAVILY] injected_context_len={len(ctx)} query={q!r}")
    return ctx

# ------------------------------
# core: llm_chat (chat.completions + optional Tavily)
# ------------------------------
def run_llm_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 1200,
    trace_tag: str = "",
    retry_on_empty: bool = True,
    web_search: Optional[bool] = None,
    allowed_domains: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> str:
    """
    ✅ 你要的行为：
    - 默认：不联网，只走 /v1/chat/completions
    - web_search=True：只用 Tavily 联网搜索（不再尝试 /v1/responses），把结果注入 prompt 再生成
    - web_search=False：不联网

    WEB_MODE 兼容：
    - WEB_MODE="never" -> 默认不联网
    - WEB_MODE in ("auto","always","tavily","tavily_only") -> 默认联网（Tavily）
      * 注意：本实现里 "always" 仅表示“默认开启联网”，并不再尝试 /v1/responses
    """
    if os.getenv(ALLOW_PAID_ENV) != "1":
        raise RuntimeError(
            f"Paid LLM calls are disabled. Set {ALLOW_PAID_ENV}=1 only when you intentionally want to spend API balance."
        )
    _require_config()
    eps = _infer_endpoints()
    chat_url = eps["chat_url"]

    _cur_key = _next_api_key()
    headers = {
        "Authorization": f"Bearer {_cur_key}",
        "Content-Type": "application/json",
    }

    model_to_use = model or MODEL
    trace_id0 = _make_trace_id(trace_tag or "chat")

    # Decide whether to use Tavily
    if web_search is None:
        if WEB_MODE == "never":
            web_search = False
        elif WEB_MODE in ("auto", "always", "tavily", "tavily_only"):
            web_search = True
        else:
            web_search = False

    injected_messages = messages
    did_web_search = False

    if web_search:
        # We ignore allowed_domains here because Tavily uses its own multi-source strategy in tavily_websearch.py
        trace_id_ws = _make_trace_id((trace_tag or "chat") + "_tavily")
        try:
            ctx = _tavily_context_for(messages, trace_id_ws)
            if ctx.strip():
                injected_messages = _inject_web_context(messages, ctx)
                did_web_search = True
        except Exception as e:
            # Since user explicitly requested "联网就是用Tavily"，失败就明确报错（更符合预期）
            if DEBUG:
                print(f"[LLM][TAVILY_FAIL] trace_id={trace_id_ws} err={repr(e)}")
            raise RuntimeError(
                "Tavily 联网搜索失败：请检查是否安装 tavily-python、Tavily API Key 是否配置正确，或网络是否可达。"
            ) from e

    def _do_chat_post(model_name: str, max_tokens_local: int, trace_id: str, msgs: List[Dict[str, str]]) -> Dict[str, Any]:
        # V27.3.15 (2026-05-12) 兼容性:
        #   - gpt-5/o3/o4 系列智增增需 max_completion_tokens 不是 max_tokens
        #   - gpt-5/o3/o4 系列强制 temperature=1.0 (OpenAI API 限制, 不能改)
        _is_new_openai = any(model_name.startswith(p) for p in ("gpt-5", "o3", "o4"))
        token_param = "max_completion_tokens" if _is_new_openai else "max_tokens"
        effective_temp = 1.0 if _is_new_openai else temperature
        max_tokens_effective = _completion_token_budget(model_name, max_tokens_local)
        reasoning_effort = _reasoning_effort_for(model_name)
        payload = {
            "model": model_name,
            "messages": msgs,
            "temperature": effective_temp,
            token_param: max_tokens_effective,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        if DEBUG:
            print("\n" + "=" * 80)
            print(f"[LLM][REQ] trace_id={trace_id}")
            print(f"[LLM][REQ] url={chat_url}")
            print(
                f"[LLM][REQ] model={model_name} temp={temperature} "
                f"max_tokens={max_tokens_effective}"
                + (f" reasoning_effort={reasoning_effort}" if reasoning_effort else "")
            )
            print(f"[LLM][REQ] api_key={_mask(_cur_key)} proxies={'ON' if PROXIES else 'OFF'} timeout={TIMEOUT}s")
            if msgs:
                sys_len = len(msgs[0].get("content", ""))
                user_len = len(msgs[-1].get("content", ""))
                print(f"[LLM][REQ] messages_preview(system_len={sys_len}, last_len={user_len})")

        # 2026-06-02: 走全局 adaptive throttle (token bucket + circuit breaker).
        # 阻塞直到能发 (RPS 限速 + 熔断不放行).
        GLOBAL_THROTTLE.acquire()

        t0 = time.time()
        resp = _post_with_retry(chat_url, headers, payload, trace_id, retries=3)
        if (
            resp.status_code == 400
            and "reasoning_effort" in payload
            and "reasoning_effort" in (resp.text or "").lower()
        ):
            if DEBUG:
                print(f"[LLM][RETRY_NO_REASONING_EFFORT] trace_id={trace_id} gateway rejected reasoning_effort")
            _dump_text(f"{trace_id}_reasoning_effort_rejected_400.txt", resp.text or "")
            payload.pop("reasoning_effort", None)
            resp = _post_with_retry(chat_url, headers, payload, trace_id, retries=1)
        dt = time.time() - t0
        if DEBUG:
            print(f"[LLM][RESP] trace_id={trace_id} status={resp.status_code} time={dt:.2f}s request_id={resp.headers.get('x-request-id','')}")
        if resp.status_code == 429:
            # Rate limited — global throttle 接管: 区分速率/额度, 自适应降速, 熔断
            for retry_i in range(5):  # 多给几次, throttle 已限速 retry 不会爆炸
                try:
                    retry_after = float(resp.headers.get("Retry-After", "0")) or None
                    wait = GLOBAL_THROTTLE.on_429(retry_after=retry_after, body=resp.text or "")
                except AccountExhaustedError:
                    # 额度耗尽 — 不再 retry, 直接抛 (上层应停 pipeline + 提示充值)
                    if DEBUG:
                        print(f"[LLM][ACCOUNT_EXHAUSTED] trace_id={trace_id} 余额耗尽, 停止")
                    raise
                if DEBUG:
                    print(f"[LLM][429_RETRY] trace_id={trace_id} retry={retry_i+1}/5 wait={wait:.1f}s (throttle stats: {GLOBAL_THROTTLE.stats()})")
                time.sleep(wait)
                GLOBAL_THROTTLE.acquire()  # 重新拿 token 才能 retry
                resp = _post_with_retry(chat_url, headers, payload, trace_id, retries=1)
                dt = time.time() - t0
                if resp.ok:
                    GLOBAL_THROTTLE.on_success()
                    break
                if resp.status_code != 429:
                    break  # 非 429 错误, 不再 retry throttle, 让下面错误处理接管
            if DEBUG and not resp.ok:
                print(f"[LLM][429_EXHAUSTED] trace_id={trace_id} giving up after 5 retries; throttle={GLOBAL_THROTTLE.stats()}")
        elif resp.status_code == 200:
            GLOBAL_THROTTLE.on_success()
        if not resp.ok:
            _dump_text(f"{trace_id}_http_error_{resp.status_code}.txt", resp.text or "")
            resp.raise_for_status()

        data = resp.json()

        # 网关可能返回 200 但内容是 {"error": {...}}（你遇到过 /responses，chat 也可能这样）
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            _dump_json(f"{trace_id}_raw.json", data)
            raise RuntimeError(f"GatewayError: {data['error'].get('message','unknown')} (code={data['error'].get('code','')})")

        _dump_json(f"{trace_id}_raw.json", data)
        return data

    # ---------- Sanitize helper for 402 content filter retry ----------
    def _sanitize_messages_for_retry(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Aggressively remove sensitive content from messages to bypass API content filter."""
        from tavily import _clean_sensitive_content
        sanitized = []
        for m in msgs:
            new_m = dict(m)
            c = new_m.get("content", "")
            if c:
                new_m["content"] = _clean_sensitive_content(c)
            sanitized.append(new_m)
        return sanitized

    # ---------- Normal chat.completions stage ----------
    try:
        data = _do_chat_post(model_to_use, max_tokens, trace_id0, injected_messages)
    except Exception as e:
        err_msg = str(e).lower()
        # Detect HTTP 402 "sensitive words" content filter
        if "402" in err_msg or "敏感词" in err_msg or "sensitive" in err_msg:
            if DEBUG:
                print(f"[LLM][402_RETRY] Content filter triggered, sanitizing and retrying...")
            trace_id_sanitized = _make_trace_id((trace_tag or "chat") + "_sanitized_retry")
            sanitized_msgs = _sanitize_messages_for_retry(injected_messages)
            data = _do_chat_post(model_to_use, max_tokens, trace_id_sanitized, sanitized_msgs)
        else:
            raise

    # 特判：空 content + length（你已经遇到过）
    if retry_on_empty and isinstance(data, dict) and _is_empty_length_completion(data):
        if DEBUG:
            print("[LLM][EMPTY_CONTENT] Detected empty content with finish_reason=length (likely reasoning-only).")
            _dump_debug_artifacts(trace_id0, data)

        trace_id1 = _make_trace_id((trace_tag or "chat") + "_retry_more_tokens")
        bigger = min(max_tokens * 4, 8192)
        data = _do_chat_post(model_to_use, bigger, trace_id1, injected_messages)

        if isinstance(data, dict) and _is_empty_length_completion(data):
            if DEBUG:
                print("[LLM][EMPTY_CONTENT] Still empty after increasing max_tokens. Try fallback models.")
                _dump_debug_artifacts(trace_id1, data)

            for fb in FALLBACK_MODELS:
                trace_id_fb = _make_trace_id((trace_tag or "chat") + f"_fallback_{fb}")
                data_fb = _do_chat_post(fb, bigger, trace_id_fb, injected_messages)
                if isinstance(data_fb, dict) and not _is_empty_length_completion(data_fb):
                    content_fb = _extract_text_from_chat_completions(data_fb)
                    if isinstance(content_fb, str) and content_fb.strip():
                        _dump_text(f"{trace_id_fb}_content.txt", content_fb)
                        if DEBUG:
                            print(f"[LLM][FALLBACK_OK] model={fb} content_len={len(content_fb)}")
                        return content_fb

            raise RuntimeError("Empty-content issue persists after retry+fallback. Check llm_logs for *_raw.json")

    content = _extract_text_from_chat_completions(data)
    if not (isinstance(content, str) and content.strip()):
        if DEBUG:
            print("[LLM][PARSE_FAIL] cannot extract non-empty text.")
            _dump_debug_artifacts(trace_id0, data)
        raise RuntimeError(f"LLM response has no extractable text. raw saved: {Path(LOG_DIR) / (trace_id0 + '_raw.json')}")

    _dump_text(f"{trace_id0}_content.txt", content)
    if DEBUG:
        hint = " (with Tavily web context)" if did_web_search else ""
        print(f"[LLM][RESP]{hint} content_len={len(content)} head:\n{content[:600]}")
    return content

# tavily_websearch.py
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Any

try:
    from tavily import TavilyClient  # pip install tavily-python
except Exception:  # pragma: no cover
    TavilyClient = None  # type: ignore


def _get_tavily_keys() -> List[str]:
    keys: List[str] = []
    try:
        import sys
        from pathlib import Path
        _llm_dir = str(Path(__file__).resolve().parents[2] / "LLM")
        if _llm_dir not in sys.path:
            sys.path.insert(0, _llm_dir)
        import llm_config  # type: ignore
        ks = getattr(llm_config, "TAVILY_KEYS", None)
        if isinstance(ks, list):
            keys = [k.strip() for k in ks if isinstance(k, str) and k.strip()]
        if not keys:
            k = getattr(llm_config, "TavilyAPI", "")
            if isinstance(k, str) and k.strip():
                keys = [k.strip()]
    except Exception:
        pass
    env_k = os.getenv("TAVILY_API_KEY", "").strip()
    if env_k and env_k not in keys:
        keys.append(env_k)
    return keys


def _get_tavily_key() -> str:
    ks = _get_tavily_keys()
    return ks[0] if ks else ""


def _clear_proxy_for_tavily():
    """临时清除代理环境变量，让 Tavily SDK 直连（宁波机 Clash 会拦截 Tavily）。"""
    _saved = {}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if k in os.environ:
            _saved[k] = os.environ.pop(k)
    return _saved

def _restore_proxy(saved):
    """恢复之前保存的代理环境变量。"""
    for k, v in saved.items():
        os.environ[k] = v

_KEY_IDX = 0

def _client(offset: int = 0) -> "TavilyClient":
    global _KEY_IDX
    if TavilyClient is None:
        raise RuntimeError("tavily-python not installed. Run: pip install tavily-python")
    keys = _get_tavily_keys()
    if not keys:
        raise RuntimeError("Tavily API key not set. Set llm_config.TAVILY_KEYS / TavilyAPI or env TAVILY_API_KEY.")
    idx = (_KEY_IDX + offset) % len(keys)
    return TavilyClient(api_key=keys[idx])


def _rotate_key():
    global _KEY_IDX
    keys = _get_tavily_keys()
    if keys:
        _KEY_IDX = (_KEY_IDX + 1) % len(keys)


def _safe_search(c_ref, **kwargs):
    """轮询 Tavily key：遇到限流/429 自动切换到下一个 key 重试。"""
    keys = _get_tavily_keys()
    last_exc = None
    for _ in range(max(1, len(keys))):
        try:
            return c_ref[0].search(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "blocked" in msg or "excessive" in msg or "429" in msg or "rate" in msg:
                _rotate_key()
                c_ref[0] = _client()
                last_exc = e
                continue
            raise
    raise last_exc if last_exc else RuntimeError("Tavily: all keys exhausted")


def run_tavily_search_category(
    query: str,
    category_type: str,
    *,
    sleep_sec: float = 0.2,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Multi-strategy search with English-oriented categories:
      - encyclopedia: Wikipedia, Britannica, World Bank
      - news: topic="news" (English news sources)
      - government: UN, World Bank, government statistics
      - academic: Google Scholar, ResearchGate, PDF reports
    Returns: list of {title,url,content}
    """
    _c_ref = [_client()]
    _saved_proxy = _clear_proxy_for_tavily()  # 清除代理，让 Tavily 直连

    if sleep_sec > 0:
        time.sleep(sleep_sec)

    if category_type == "encyclopedia":
        resp = _safe_search(_c_ref, 
            query=query,
            search_depth="basic",
            max_results=min(max_results, 5),
            include_domains=[
                "en.wikipedia.org",
                "britannica.com",
                "worldbank.org",
                "data.worldbank.org",
                "cia.gov",
            ],
        )
    elif category_type == "news":
        resp = _safe_search(_c_ref, 
            query=query,
            search_depth="basic",
            topic="news",
            max_results=max_results,
        )
    elif category_type == "government":
        resp = _safe_search(_c_ref, 
            query=query + " (site:worldbank.org OR site:un.org OR site:undrr.org OR site:unhabitat.org OR site:gov)",
            search_depth="basic",
            max_results=max_results,
        )
    elif category_type == "academic":
        resp = _safe_search(_c_ref, 
            query=query + " (site:scholar.google.com OR site:researchgate.net OR site:sciencedirect.com OR filetype:pdf OR report)",
            search_depth="basic",
            max_results=max_results,
        )
    else:
        resp = _safe_search(_c_ref, query=query, search_depth="basic", max_results=max_results)

    _restore_proxy(_saved_proxy)  # 恢复代理（不影响其他程序）

    results = resp.get("results", []) if isinstance(resp, dict) else []
    out: List[Dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        out.append({
            "title": r.get("title", "") or "",
            "url": r.get("url", "") or "",
            "content": r.get("content", "") or "",
        })
    return out


def run_multi_source_search(
    user_question: str,
    *,
    wiki_max: int = 5,
    news_max: int = 5,
    gov_max: int = 5,
    acad_max: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns dict with keys: wiki/news/gov/acad
    """
    return {
        "wiki": search_category(user_question, "encyclopedia", max_results=wiki_max),
        "news": search_category(user_question, "news", max_results=news_max),
        "gov": search_category(user_question, "government", max_results=gov_max),
        "acad": search_category(user_question, "academic", max_results=acad_max),
    }


def _clean_sensitive_content(text: str) -> str:
    """
    Remove keywords that may trigger Chinese LLM API content moderation.

    保守策略：大部分敏感词保留过滤（防止豆包/GLM/混元拒答），
    但 "political" 和 "corruption" 这两个 GIRI 指标核心英文词不过滤，
    因为它们是 World Bank / UN 标准术语，不会触发英文 API，
    且对 Political_stability 和 Control_of_corruption 两个指标的评分质量至关重要。
    """
    # 过滤：可能触发国内 API 审查的词（中英文都保留）
    sensitive_keywords = [
        "政治", "腐败", "政变", "暴动", "新疆", "西藏", "台湾",
        "人权", "民主", "自由", "游行", "示威", "镇压",
        "coup", "riot", "xinjiang", "tibet", "taiwan",
        "human rights", "democracy", "protest", "suppression",
    ]
    # 不过滤："political" 和 "corruption"
    # 理由：这两个是 GIRI 指标名（Political_stability, Control_of_corruption），
    # 属于 World Bank 标准术语，国内 API 对纯英文治理术语容忍度较高。

    cleaned = text
    for kw in sensitive_keywords:
        cleaned = cleaned.replace(kw, "[filtered]")
        cleaned = cleaned.replace(kw.upper(), "[filtered]")
        cleaned = cleaned.replace(kw.title(), "[filtered]")

    return cleaned

def _filter_relevant_results(
    items: List[Dict[str, Any]],
    city_name: str,
    country_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Filter search results to keep only those mentioning the city or country.
    Falls back to returning all items if none match (better than empty).
    """
    if not city_name:
        return items
    city_lower = city_name.strip().lower()
    country_lower = (country_name or "").strip().lower()
    keywords = [city_lower]
    if country_lower:
        keywords.append(country_lower)

    relevant = []
    for it in items:
        text = (it.get("title", "") + " " + it.get("url", "") + " " + it.get("content", "")).lower()
        if any(kw in text for kw in keywords):
            relevant.append(it)
    return relevant if relevant else items


def build_search_context(
    bundle: Dict[str, List[Dict[str, Any]]],
    *,
    max_chars_per_item: int = 800,
    city_name: str = "",
    country_name: str = "",
) -> str:
    """
    Pretty text block for prompt injection. Filters irrelevant results first.
    """
    def _clip(s: str) -> str:
        s = (s or "").strip()
        s = _clean_sensitive_content(s)
        if len(s) <= max_chars_per_item:
            return s
        return s[:max_chars_per_item].rstrip() + "..."

    parts: List[str] = []
    mapping = [
        ("wiki", "Encyclopedia"),
        ("news", "News"),
        ("gov", "Government/Intl Org"),
        ("acad", "Academic/Reports")
    ]

    for key, label in mapping:
        items = bundle.get(key, [])
        if not items:
            continue
        if city_name:
            items = _filter_relevant_results(items, city_name, country_name)
        parts.append(f"\n--- [{label}] Sources (Top {len(items)}) ---")
        for i, it in enumerate(items, 1):
            title = _clip(it.get("title", "No title"))
            url = (it.get("url", "") or "").strip()
            content = _clip(it.get("content", "No content"))
            parts.append(f"[{label}#{i}] {title}\nURL: {url}\nSummary: {content}\n")
    return "\n".join(parts).strip()


def run_quick_web_context(user_question: str, city_name: str = "", country_name: str = "") -> str:
    """
    One-call helper: search + filter + format. Raises if Tavily isn't available/configured.
    """
    bundle = multi_source_search(user_question)
    text = format_search_context(bundle, city_name=city_name, country_name=country_name)
    return text or ""


def run_quick_web_context_v2(city_name: str, country_name: str, min_results: int = 20) -> str:
    """
    V2: 多轮 Tavily 搜索，保证每个城市至少 20 条结果。
    第 1 轮：4 方向精确搜索（wiki/news/gov/acad）。
    第 2 轮（补救）：放宽查询（去引号、加国家名、加通用关键词）。
    第 3 轮（兜底）：纯通用搜索（城市名 + 国家名）。
    仍不足 5 条 → 抛异常。5-19 条 → 标记 low quality 但继续。
    """
    # ── 第 1 轮：精确搜索 ──
    queries_r1 = {
        "wiki": f'"{city_name}" "{city_name} city" population economy infrastructure {country_name}',
        "news": f'"{city_name}" "{city_name}" urban development resilience {country_name}',
        "gov": f'"{city_name}" "{city_name}" HDI governance statistics {country_name}',
        "acad": f'"{city_name}" "{city_name}" urban resilience inequality environment {country_name}',
    }
    bundle = {
        "wiki": search_category(queries_r1["wiki"], "encyclopedia", max_results=10),
        "news": search_category(queries_r1["news"], "news", max_results=10),
        "gov": search_category(queries_r1["gov"], "government", max_results=10),
        "acad": search_category(queries_r1["acad"], "academic", max_results=10),
    }

    total = sum(len(v) for v in bundle.values())

    # ── 第 2 轮：放宽搜索（去引号，加更多关键词） ──
    if total < min_results:
        queries_r2 = {
            "wiki": f'{city_name} city {country_name} population demographics economy Wikipedia',
            "news": f'{city_name} {country_name} development infrastructure climate disaster',
            "gov": f'{city_name} {country_name} government statistics poverty HDI World Bank',
            "acad": f'{city_name} {country_name} urban resilience vulnerability risk assessment',
        }
        for key in queries_r2:
            if len(bundle[key]) < 5:
                extra = search_category(queries_r2[key], "general", max_results=10)
                # 去重：只添加 URL 不重复的结果
                existing_urls = {it.get("url", "") for it in bundle[key]}
                for item in extra:
                    if item.get("url", "") not in existing_urls:
                        bundle[key].append(item)
                        existing_urls.add(item.get("url", ""))
        total = sum(len(v) for v in bundle.values())

    # ── 第 3 轮：纯通用兜底 ──
    if total < min_results:
        fallback = search_category(
            f'{city_name} {country_name} city overview',
            "general", max_results=20
        )
        existing_urls = set()
        for v in bundle.values():
            for it in v:
                existing_urls.add(it.get("url", ""))
        for item in fallback:
            if item.get("url", "") not in existing_urls:
                bundle["news"].append(item)  # 归入 news 类
                existing_urls.add(item.get("url", ""))
        total = sum(len(v) for v in bundle.values())

    # ── 质量判断 ──
    if total < 5:
        raise RuntimeError(
            f"Tavily results insufficient for {city_name}: only {total}/5 minimum "
            f"(wiki={len(bundle['wiki'])}, news={len(bundle['news'])}, "
            f"gov={len(bundle['gov'])}, acad={len(bundle['acad'])})"
        )

    text = format_search_context(bundle, city_name=city_name, country_name=country_name)
    if not text or len(text) < 50:
        raise RuntimeError(f"Tavily search context too short ({len(text or '')} chars) for {city_name}")

    # 标记搜索质量（供 Step 3 融合参考）
    if total < min_results:
        text = f"[SEARCH_QUALITY: LOW ({total}/{min_results} results)]\n\n" + text

    return text

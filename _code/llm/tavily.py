# tavily_websearch.py
from __future__ import annotations

import time
from typing import Dict, List, Any

import requests

TAVILY_URL = "https://api.tavily.com/search"
_LIMIT_STATUS = {429, 432, 433}  # 限流或额度用尽 → 换下一个 key
_KEY_IDX = 0


def _get_tavily_keys() -> List[str]:
    """Tavily key 只从 _config.TAVILY_KEYS 取（环境变量 TAVILY_API_KEYS / TAVILY_API_KEY）。"""
    from _config import TAVILY_KEYS
    return [k.strip() for k in TAVILY_KEYS if isinstance(k, str) and k.strip()]


def _post_search(**payload) -> Dict[str, Any]:
    """直连 Tavily REST 搜索一次；遇限流换下一个 key，其他错误直接抛出。

    不用 tavily-python SDK：本模块名 tavily 会遮蔽 SDK 包名，SDK 也不在项目环境里。
    """
    global _KEY_IDX
    keys = _get_tavily_keys()
    if not keys:
        raise RuntimeError("Tavily API key not set. Set env TAVILY_API_KEYS or TAVILY_API_KEY.")
    session = requests.Session()
    session.trust_env = False  # 代理会拦截 Tavily，直连
    for _ in range(len(keys)):
        resp = session.post(TAVILY_URL, json={"api_key": keys[_KEY_IDX % len(keys)], **payload}, timeout=30)
        if resp.status_code in _LIMIT_STATUS:
            _KEY_IDX += 1
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Tavily: all {len(keys)} keys hit rate or plan limits")


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
    if sleep_sec > 0:
        time.sleep(sleep_sec)

    if category_type == "encyclopedia":
        resp = _post_search(
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
        resp = _post_search(
            query=query,
            search_depth="basic",
            topic="news",
            max_results=max_results,
        )
    elif category_type == "government":
        resp = _post_search(
            query=query + " (site:worldbank.org OR site:un.org OR site:undrr.org OR site:unhabitat.org OR site:gov)",
            search_depth="basic",
            max_results=max_results,
        )
    elif category_type == "academic":
        resp = _post_search(
            query=query + " (site:scholar.google.com OR site:researchgate.net OR site:sciencedirect.com OR filetype:pdf OR report)",
            search_depth="basic",
            max_results=max_results,
        )
    else:
        resp = _post_search(query=query, search_depth="basic", max_results=max_results)

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
        "wiki": run_tavily_search_category(user_question, "encyclopedia", max_results=wiki_max),
        "news": run_tavily_search_category(user_question, "news", max_results=news_max),
        "gov": run_tavily_search_category(user_question, "government", max_results=gov_max),
        "acad": run_tavily_search_category(user_question, "academic", max_results=acad_max),
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
    bundle = run_multi_source_search(user_question)
    text = build_search_context(bundle, city_name=city_name, country_name=country_name)
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
        "wiki": run_tavily_search_category(queries_r1["wiki"], "encyclopedia", max_results=10),
        "news": run_tavily_search_category(queries_r1["news"], "news", max_results=10),
        "gov": run_tavily_search_category(queries_r1["gov"], "government", max_results=10),
        "acad": run_tavily_search_category(queries_r1["acad"], "academic", max_results=10),
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
                extra = run_tavily_search_category(queries_r2[key], "general", max_results=10)
                # 去重：只添加 URL 不重复的结果
                existing_urls = {it.get("url", "") for it in bundle[key]}
                for item in extra:
                    if item.get("url", "") not in existing_urls:
                        bundle[key].append(item)
                        existing_urls.add(item.get("url", ""))
        total = sum(len(v) for v in bundle.values())

    # ── 第 3 轮：纯通用兜底 ──
    if total < min_results:
        fallback = run_tavily_search_category(
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

    text = build_search_context(bundle, city_name=city_name, country_name=country_name)
    if not text or len(text) < 50:
        raise RuntimeError(f"Tavily search context too short ({len(text or '')} chars) for {city_name}")

    # 标记搜索质量（供 Step 3 融合参考）
    if total < min_results:
        text = f"[SEARCH_QUALITY: LOW ({total}/{min_results} results)]\n\n" + text

    return text

import json
import os
from typing import Any

import requests


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web with Tavily and return model-friendly JSON."""
    query = query.strip()
    if not query:
        return _json_result(ok=False, error="搜索关键词不能为空。")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return _json_result(ok=False, error="缺少环境变量 TAVILY_API_KEY。")

    # Keep the range small so a model cannot accidentally create a costly request.
    max_results = max(1, min(max_results, 8))
    print(f"\n[Tool] 正在搜索：{query}")

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            timeout=20,
        )
        response.raise_for_status()
    except requests.Timeout:
        return _json_result(ok=False, error="Tavily 搜索超时。")
    except requests.RequestException as exc:
        status_code = getattr(exc.response, "status_code", None)
        error = f"Tavily 搜索失败（HTTP {status_code}）。" if status_code else "Tavily 搜索失败。"
        return _json_result(ok=False, error=error)

    try:
        payload = response.json()
    except requests.JSONDecodeError:
        return _json_result(ok=False, error="Tavily 返回了无法解析的响应。")

    results: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score"),
            }
        )

    return _json_result(
        ok=True,
        query=query,
        results=results,
        result_count=len(results),
    )


def _json_result(**data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)

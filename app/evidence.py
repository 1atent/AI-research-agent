import json
import math
from urllib.parse import urlsplit, urlunsplit

from app.models import Evidence, ResearchState, SearchOutcome


MAX_SNIPPETS_PER_SOURCE = 8
MAX_SNIPPET_CHARS = 4000


def parse_search_results(raw_result: str, sub_question: str) -> list[Evidence]:
    """Convert untrusted search-tool JSON into validated Evidence objects."""
    return parse_search_response(raw_result, sub_question).evidence


def parse_search_response(raw_result: str, sub_question: str) -> SearchOutcome:
    """Preserve search errors instead of turning every failure into an empty list."""
    try:
        payload = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError):
        return SearchOutcome(ok=False, error="搜索结果不是有效 JSON。")

    if not isinstance(payload, dict):
        return SearchOutcome(ok=False, error="搜索结果不是 JSON 对象。")
    if payload.get("ok") is not True:
        error = _clean_text(payload.get("error")) or "搜索失败。"
        return SearchOutcome(ok=False, error=error)

    raw_items = payload.get("results")
    if not isinstance(raw_items, list):
        return SearchOutcome(ok=False, error="搜索结果缺少有效 results 列表。")

    evidence: list[Evidence] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        title = _clean_text(item.get("title"))
        url = _clean_text(item.get("url"))
        content = _clean_text(item.get("content"))
        if not title or not _url_key(url) or not content:
            continue

        evidence.append(
            Evidence(
                title=title,
                url=url,
                content=content[:MAX_SNIPPET_CHARS],
                score=_parse_score(item.get("score")),
                supports=[sub_question],
                snippets=[content[:MAX_SNIPPET_CHARS]],
            )
        )

    return SearchOutcome(ok=True, evidence=evidence)


def add_evidence(state: ResearchState, new_evidence: list[Evidence]) -> int:
    """Add new evidence to state, merging duplicate URLs and their support."""
    evidence_by_url = {_url_key(item.url): item for item in state.evidence}
    added_count = 0

    for candidate in new_evidence:
        key = _url_key(candidate.url)
        if not key or not candidate.content.strip():
            continue

        existing = evidence_by_url.get(key)
        if existing is not None:
            _merge_evidence(existing, candidate)
            continue

        state.evidence.append(candidate)
        evidence_by_url[key] = candidate
        added_count += 1

    return added_count


def count_unique_sources(state: ResearchState) -> int:
    return len({_url_key(item.url) for item in state.evidence if _url_key(item.url)})


def _merge_evidence(existing: Evidence, candidate: Evidence) -> None:
    for sub_question in candidate.supports:
        if sub_question not in existing.supports:
            existing.supports.append(sub_question)

    snippets = existing.snippets or [existing.content]
    snippets = list(dict.fromkeys(
        text.strip()[:MAX_SNIPPET_CHARS] for text in snippets if text.strip()
    ))
    for text in candidate.snippets or [candidate.content]:
        cleaned = text.strip()[:MAX_SNIPPET_CHARS]
        if cleaned and cleaned not in snippets:
            snippets.append(cleaned)
    if len(snippets) > MAX_SNIPPETS_PER_SOURCE:
        # Keep the first observation and recent complementary snippets within a budget.
        snippets = [snippets[0], *snippets[-(MAX_SNIPPETS_PER_SOURCE - 1):]]
    existing.snippets = snippets
    existing.content = "\n\n".join(snippets)
    existing.last_seen_step = max(existing.last_seen_step, candidate.last_seen_step)

    if candidate.score is not None and (
        existing.score is None or candidate.score > existing.score
    ):
        existing.score = candidate.score


def _url_key(url: str) -> str:
    """Normalize harmless URL differences for source deduplication."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""

    scheme = parts.scheme.casefold() # 协议转小写（HTTP → http）
    if scheme not in {"http", "https"} or not parts.netloc:
        return ""

    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (
            scheme,
            parts.netloc.casefold(),
            path,
            parts.query,
            "",
        )
    )


def _clean_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parse_score(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
        return score if math.isfinite(score) else None
    except (TypeError, ValueError):
        return None


def select_evidence(state: ResearchState, max_sources: int) -> list[Evidence]:
    """Reserve topic coverage and recent search results before filling by relevance."""
    if max_sources <= 0:
        return []
    candidates = [item for item in state.evidence if item.content.strip()]
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.score if item.score is not None else -1.0,
            item.last_seen_step,
        ),
        reverse=True,
    )
    selected: list[Evidence] = []
    seen_urls: set[str] = set()

    def reserve(item: Evidence) -> None:
        key = _url_key(item.url)
        if key and key not in seen_urls and len(selected) < max_sources:
            selected.append(item)
            seen_urls.add(key)

    missing = (
        state.last_evaluation.missing_sub_questions
        if state.last_evaluation else []
    )
    questions = list(dict.fromkeys([*missing, *state.plan.sub_questions]))
    for question in questions:
        relevant = [item for item in ranked if question in item.supports]
        if relevant:
            # New observations for an unresolved topic must reach the evaluator.
            if question in missing:
                relevant.sort(key=lambda item: item.last_seen_step, reverse=True)
            reserve(relevant[0])

    # Reserve up to five sources from the most recent successful search batch.
    latest_step = max((item.last_seen_step for item in ranked), default=0)
    if latest_step > 0:
        latest = [item for item in ranked if item.last_seen_step == latest_step]
        for item in latest[:5]:
            reserve(item)

    for item in ranked:
        reserve(item)
        if len(selected) >= max_sources:
            break
    return selected


def evidence_excerpt(item: Evidence, max_chars: int = 1200) -> str:
    """Share the character budget across snippets so new fragments are not cut off."""
    if max_chars <= 0:
        return ""
    snippets = item.snippets or [item.content]
    snippets = [text for text in snippets if text.strip()]
    if not snippets:
        return ""
    per_snippet = max(1, (max_chars - 2 * (len(snippets) - 1)) // len(snippets))
    return "\n\n".join(text[:per_snippet] for text in snippets)[:max_chars]

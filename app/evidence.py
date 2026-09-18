import json
from urllib.parse import urlsplit, urlunsplit

from app.models import Evidence, ResearchState


def parse_search_results(raw_result: str, sub_question: str) -> list[Evidence]:
    """Convert untrusted search-tool JSON into validated Evidence objects."""
    try:
        payload = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return []

    raw_items = payload.get("results")
    if not isinstance(raw_items, list):
        return []

    evidence: list[Evidence] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        title = _clean_text(item.get("title"))
        url = _clean_text(item.get("url"))
        content = _clean_text(item.get("content"))
        if not title or not url:
            continue

        evidence.append(
            Evidence(
                title=title,
                url=url,
                content=content,
                score=_parse_score(item.get("score")),
                supports=[sub_question],
            )
        )

    return evidence


def add_evidence(state: ResearchState, new_evidence: list[Evidence]) -> int:
    """Add new evidence to state, merging duplicate URLs and their support."""
    evidence_by_url = {_url_key(item.url): item for item in state.evidence}
    added_count = 0

    for candidate in new_evidence:
        key = _url_key(candidate.url)
        if not key:
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

    if len(candidate.content) > len(existing.content):
        existing.content = candidate.content

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
        return float(value)
    except (TypeError, ValueError):
        return None

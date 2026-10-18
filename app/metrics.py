"""Offline operational metrics, not an automatic factual-quality judge.

The input is an exported run, not model output. Citation checks only verify
whether body links belong to the source mapping; they cannot establish that a
source is reachable or supports the surrounding claim.
"""

import math
import re
from collections import Counter

from app.models import ResearchStatus, StopReason


_REFERENCE_HEADING = re.compile(
    r"^ {0,3}#{1,6}[ \t]+(?:参考来源|参考文献|引用来源|references|sources|bibliography)"
    r"[:：]?[ \t]*(?:#+[ \t]*)?\r?$",
    re.MULTILINE | re.IGNORECASE,
)
_FENCE_OPEN = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)*?\1")
_INLINE_LINK = re.compile(
    r"(?<!!)\[(?:\\.|[^\]\\\n])+\]\(\s*"
    r"(?:<(?P<angle>[^>\n]+)>|(?P<plain>(?:\\.|[^()\s]|\([^()\n]*\))+))"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^\)\n]*\)))?\s*\)"
)
_UNKNOWN_SOURCE = re.compile(r"\[(?:S\d+|unknown\s+source)\]", re.IGNORECASE)


def analyze_run(payload: dict, report: str) -> dict:
    """Measure an exported schema-v1 run without calling a model or network."""
    payload = _mapping(payload, "payload")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    state = _mapping(payload.get("state"), "state")
    exported_metrics = _mapping(payload.get("metrics"), "metrics")
    if not isinstance(report, str):
        raise ValueError("report must be a string")

    status = _choice(state.get("status", "planning"), ResearchStatus, "status")
    stop_reason = state.get("stop_reason")
    if stop_reason is not None:
        stop_reason = _choice(stop_reason, StopReason, "stop_reason")
    generated = state.get("report_generated", False)
    if not isinstance(generated, bool):
        raise ValueError("report_generated must be a boolean")

    trace = state.get("trace", [])
    if not isinstance(trace, list):
        raise ValueError("state.trace must be a list")
    search_errors = 0
    evaluation_errors = 0
    evaluation_events = 0
    for event in trace:
        event = _mapping(event, "trace event")
        details = _mapping(event.get("details", {}), "trace event details")
        if event.get("stage") == "search":
            search_errors += bool(details.get("error") or details.get("ok") is False)
        elif event.get("stage") == "evaluation":
            evaluation_events += 1
            result = details.get("result")
            nested_error = result.get("error") if isinstance(result, dict) else None
            evaluation_errors += bool(details.get("error") or nested_error)

    sources = _mapping(state.get("report_source_urls", {}), "report_source_urls")
    if any(not isinstance(key, str) or not isinstance(url, str) or not url.strip()
           for key, url in sources.items()):
        raise ValueError("report_source_urls must map strings to non-empty URLs")

    return {
        "status": status,
        "stop_reason": stop_reason,
        "report_generated": generated,
        "search_count": _number(
            exported_metrics.get("search_count", state.get("step_count", 0)),
            "search_count", integer=True,
        ),
        "evaluation_count": _number(
            exported_metrics.get("evaluation_count", evaluation_events),
            "evaluation_count", integer=True,
        ),
        "source_count": _number(
            exported_metrics.get("source_count", _source_count(state)),
            "source_count", integer=True,
        ),
        "elapsed_seconds": _number(
            exported_metrics.get("elapsed_seconds", state.get("elapsed_seconds", 0)),
            "elapsed_seconds",
        ),
        "evaluator_coverage": _evaluator_coverage(state),
        "search_error_count": search_errors,
        "evaluation_error_count": evaluation_errors,
        "citation_check": _citation_check(report, set(sources.values())),
    }


def summarize_runs(records: list[dict]) -> dict:
    """Aggregate operational results, retaining failed cases in rate denominators."""
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    measured = []
    error_count = 0
    for index, record in enumerate(records):
        record = _mapping(record, f"records[{index}]")
        error_count += bool(record.get("error") or record.get("error_type"))
        metrics = record.get("metrics")
        if metrics is None:
            continue
        metrics = _mapping(metrics, f"records[{index}].metrics")
        _choice(metrics.get("status"), ResearchStatus, "status")
        if not isinstance(metrics.get("report_generated"), bool):
            raise ValueError("report_generated must be a boolean")
        if metrics.get("stop_reason") is not None:
            _choice(metrics["stop_reason"], StopReason, "stop_reason")
        for field in ("search_count", "search_error_count", "evaluation_error_count"):
            _number(metrics.get(field), field, integer=True)
        _number(metrics.get("elapsed_seconds"), "elapsed_seconds")
        measured.append(metrics)

    case_count = len(records)
    measured_count = len(measured)
    completed = sum(item["status"] == "completed" for item in measured)
    generated = sum(item["report_generated"] for item in measured)
    stops = Counter(item.get("stop_reason") or "unknown" for item in measured)
    return {
        "case_count": case_count,
        "measured_case_count": measured_count,
        "completed_count": completed,
        "program_completion_rate": completed / case_count if case_count else None,
        "error_count": error_count,
        "report_generated_count": generated,
        "report_generation_rate": generated / case_count if case_count else None,
        "mean_search_count": (
            sum(item["search_count"] for item in measured) / measured_count
            if measured_count else None
        ),
        "mean_elapsed_seconds": (
            sum(item["elapsed_seconds"] for item in measured) / measured_count
            if measured_count else None
        ),
        "total_search_errors": sum(item["search_error_count"] for item in measured),
        "total_evaluation_errors": sum(item["evaluation_error_count"] for item in measured),
        "stop_reasons": dict(sorted(stops.items())),
    }


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _choice(value: object, choices: type, name: str) -> str:
    if not isinstance(value, str) or value not in {item.value for item in choices}:
        raise ValueError(f"invalid {name}: {value!r}")
    return str(value.value if hasattr(value, "value") else value)


def _number(value: object, name: str, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0 or (integer and not isinstance(value, int)):
        raise ValueError(f"{name} must be a non-negative {'integer' if integer else 'finite number'}")
    return value


def _source_count(state: dict) -> int:
    evidence = state.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("state.evidence must be a list")
    urls = set()
    for item in evidence:
        item = _mapping(item, "evidence item")
        url = item.get("url")
        if url is not None and not isinstance(url, str):
            raise ValueError("evidence URL must be a string")
        if url:
            urls.add(url.rstrip("/"))
    return len(urls)


def _question_set(value: object, name: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return {item.strip() for item in value if item.strip()}


def _evaluator_coverage(state: dict) -> float | None:
    plan = state.get("plan")
    evaluation = state.get("last_evaluation")
    if plan is None:
        return None
    plan = _mapping(plan, "state.plan")
    questions = _question_set(plan.get("sub_questions", []), "sub_questions")
    if not questions or evaluation is None:
        return None
    evaluation = _mapping(evaluation, "last_evaluation")
    if "answered_sub_questions" not in evaluation:
        return None
    answered = _question_set(evaluation["answered_sub_questions"], "answered_sub_questions")
    return len(questions & answered) / len(questions)


def _citation_check(report: str, known_urls: set[str]) -> dict:
    # The writer appends all sources, including unused ones, in this section.
    # That list must never masquerade as evidence that the body has citations.
    body = _without_fenced_code(report)
    body = _REFERENCE_HEADING.split(body, maxsplit=1)[0]
    body = _INLINE_CODE.sub("", body)
    links = list(_INLINE_LINK.finditer(body))
    known_count = sum((link.group("angle") or link.group("plain")) in known_urls for link in links)
    link_count = len(links)
    return {
        "inline_link_count": link_count,
        "known_inline_link_count": known_count,
        "known_inline_link_rate": known_count / link_count if link_count else None,
        "unknown_source_count": len(_UNKNOWN_SOURCE.findall(body)),
        "unmapped_link_count": link_count - known_count,
        "has_inline_citations": bool(known_count),
    }


def _without_fenced_code(report: str) -> str:
    """Ignore Markdown fences, including longer closers and unclosed blocks."""
    body_lines = []
    closing_fence = None
    for line in report.splitlines(keepends=True):
        line_text = line.rstrip("\r\n")
        if closing_fence is not None:
            if closing_fence.fullmatch(line_text):
                closing_fence = None
            continue
        opening = _FENCE_OPEN.fullmatch(line_text)
        if opening:
            fence = opening.group("fence")
            # Backtick info strings cannot themselves contain a backtick.
            if fence[0] != "`" or "`" not in opening.group("info"):
                closing_fence = re.compile(
                    rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*"
                )
                continue
        body_lines.append(line)
    return "".join(body_lines)

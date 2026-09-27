from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter

from openai import OpenAI

from app.client import create_client
from app.evidence import (
    add_evidence,
    count_unique_sources,
    parse_search_response,
    select_evidence,
)
from app.evaluator import evaluate_research
from app.models import (
    EvaluationResult,
    EvaluationStatus,
    ResearchEvent,
    ResearchState,
    ResearchStatus,
    StopReason,
)
from app.planner import create_research_plan
from app.tools import search_web


MAX_RESEARCH_STEPS = 8 # 最大搜索步数


def run_research(
    question: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> ResearchState:
    """Plan, search, evaluate, and continue until completion or budget limits."""
    question = question.strip()
    if not question:
        raise ValueError("研究问题不能为空。")
    if client is None and model is None:
        client, model = create_client()
    elif client is None or model is None:
        raise ValueError("client 和 model 必须同时提供，或者都不提供。")

    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    plan = create_research_plan(question, client, model)
    state = ResearchState(
        plan=plan, status=ResearchStatus.RESEARCHING, started_at=started_at
    )
    state.trace.append(ResearchEvent(
        stage="planning",
        step_count=0,
        elapsed_seconds=perf_counter() - started,
        details={
            "sub_questions": list(plan.sub_questions),
            "minimum_sources": plan.minimum_sources,
        },
    ))
    print(f"[Planner] 生成 {len(plan.sub_questions)} 个子问题，至少需要 {plan.minimum_sources} 个网页来源")
    for index, sub_question in enumerate(plan.sub_questions, start=1):
        print(f"  {index}. {sub_question}")

    for sub_question in plan.sub_questions:
        if state.step_count >= MAX_RESEARCH_STEPS:
            break

        normalized_query = _normalize_query(sub_question)
        if not normalized_query or _query_was_seen(state, normalized_query):
            continue

        _search_and_store(state, sub_question, sub_question)

    while True:
        evaluation_started = perf_counter()
        visible_urls = [item.url for item in select_evidence(state, 20)]
        evaluation = evaluate_research(state, client, model)
        state.last_evaluation = evaluation
        accepted = _can_complete(state, evaluation)
        next_search = (
            _choose_next_search(state, evaluation)
            if not accepted and state.step_count < MAX_RESEARCH_STEPS else None
        )
        source_count = count_unique_sources(state)
        state.trace.append(ResearchEvent(
            stage="evaluation",
            step_count=state.step_count,
            elapsed_seconds=perf_counter() - evaluation_started,
            details={
                "result": asdict(evaluation),
                "evidence_urls": visible_urls,
                "source_count": source_count,
                "source_shortfall": max(0, plan.minimum_sources - source_count),
                "completion_accepted": accepted,
                "chosen_query": next_search[0] if next_search else None,
                "error": evaluation.error,
            },
        ))
        print(
            f"[Evaluator] 已覆盖 {len(evaluation.answered_sub_questions)}/{len(plan.sub_questions)}，"
            f"缺失 {len(evaluation.missing_sub_questions)}，来源 {source_count}/{plan.minimum_sources}"
        )
        if evaluation.reason:
            print(f"[Evaluator] {evaluation.reason}")

        if accepted:
            return _finish(state, StopReason.COMPLETED, started)

        if state.step_count >= MAX_RESEARCH_STEPS:
            return _finish(state, StopReason.SEARCH_BUDGET_EXHAUSTED, started)

        if next_search is None:
            return _finish(state, StopReason.NO_NEW_QUERY, started)

        next_query, support_question = next_search
        print(f"[Agent] 补充查询：{next_query}")
        _search_and_store(state, next_query, support_question)


def _can_complete(state: ResearchState, evaluation: EvaluationResult) -> bool:
    if evaluation.status != EvaluationStatus.COMPLETE:
        return False
    if evaluation.missing_sub_questions:
        return False
    if not state.plan.sub_questions or not state.evidence:
        return False
    if not set(state.plan.sub_questions).issubset(evaluation.answered_sub_questions):
        return False
    enough_sources = count_unique_sources(state) >= state.plan.minimum_sources
    return enough_sources


def _search_and_store(
    state: ResearchState,
    query: str,
    support_question: str,
) -> None:
    started = perf_counter()
    state.searched_queries.add(query)
    state.step_count += 1  # Count attempted searches, including failures.
    print(f"[Agent] 搜索 {state.step_count}/{MAX_RESEARCH_STEPS}")
    raw_result = search_web(query)
    outcome = parse_search_response(raw_result, support_question)
    for item in outcome.evidence:
        item.last_seen_step = state.step_count
    added = add_evidence(state, outcome.evidence)
    state.trace.append(ResearchEvent(
        stage="search",
        step_count=state.step_count,
        elapsed_seconds=perf_counter() - started,
        details={
            "query": query,
            "sub_question": support_question,
            "ok": outcome.ok,
            "candidate_count": len(outcome.evidence),
            "added_source_count": added,
            "result_urls": [item.url for item in outcome.evidence],
            "source_count": count_unique_sources(state),
            "error": outcome.error,
        },
    ))
    print(f"[Evidence] 新增 {added} 个来源，累计 {count_unique_sources(state)} 个")
    if outcome.error:
        print(f"[Search error] {outcome.error}")


def _choose_next_search(
    state: ResearchState,
    evaluation: EvaluationResult,
) -> tuple[str, str] | None:
    candidates: list[tuple[str, str]] = []
    target = (
        evaluation.missing_sub_questions[0]
        if evaluation.missing_sub_questions else state.plan.question
    )
    if evaluation.next_query:
        candidates.append((evaluation.next_query, target))
    candidates.extend(
        (f"{question} reliable sources evidence", question)
        for question in evaluation.missing_sub_questions
    )

    if count_unique_sources(state) < state.plan.minimum_sources:
        # Semantic completeness does not imply enough corroborating webpages.
        candidates.append((
            f"{state.plan.question} additional sources evidence", state.plan.question
        ))
        candidates.extend(
            (f"{question} alternative sources evidence", question)
            for question in state.plan.sub_questions
        )

    for candidate, support_question in candidates:
        normalized = _normalize_query(candidate)
        if normalized and not _query_was_seen(state, normalized):
            return candidate.strip(), support_question
    return None


def _query_was_seen(state: ResearchState, normalized_query: str) -> bool:
    return any(
        _normalize_query(query) == normalized_query
        for query in state.searched_queries
    )


def _normalize_query(query: str) -> str:
    return " ".join(query.split()).casefold()


def _finish(
    state: ResearchState, trigger: StopReason, started: float
) -> ResearchState:
    if trigger == StopReason.COMPLETED:
        state.status = ResearchStatus.COMPLETED
    else:
        state.status = ResearchStatus.PARTIAL if state.evidence else ResearchStatus.FAILED
    state.stop_reason = trigger if state.evidence else StopReason.NO_EVIDENCE
    state.elapsed_seconds = perf_counter() - started
    state.trace.append(ResearchEvent(
        stage="finish",
        step_count=state.step_count,
        elapsed_seconds=0.0,
        details={
            "status": state.status.value,
            "stop_reason": state.stop_reason.value,
            "trigger": trigger.value,
        },
    ))
    print(f"[Agent] 研究结束：{state.status.value} / {state.stop_reason.value}")
    return state

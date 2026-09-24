from openai import OpenAI

from app.client import create_client
from app.evidence import add_evidence, count_unique_sources, parse_search_results
from app.evaluator import evaluate_research
from app.models import EvaluationResult, EvaluationStatus, ResearchState, ResearchStatus
from app.planner import create_research_plan
from app.tools import search_web


MAX_RESEARCH_STEPS = 8


def run_research(
    question: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> ResearchState:
    """Plan, search, evaluate, and continue until completion or budget limits."""
    if client is None and model is None:
        client, model = create_client()
    elif client is None or model is None:
        raise ValueError("client 和 model 必须同时提供，或者都不提供。")

    plan = create_research_plan(question, client, model)
    state = ResearchState(plan=plan, status=ResearchStatus.RESEARCHING)

    for sub_question in plan.sub_questions:
        if state.step_count >= MAX_RESEARCH_STEPS:
            break

        normalized_query = sub_question.strip().casefold()
        if not normalized_query or _query_was_seen(state, normalized_query):
            continue

        _search_and_store(state, sub_question, sub_question)

    while True:
        evaluation = evaluate_research(state, client, model)
        state.last_evaluation = evaluation

        if _can_complete(state, evaluation):
            state.status = ResearchStatus.COMPLETED
            return state

        if state.step_count >= MAX_RESEARCH_STEPS:
            break

        next_query = _choose_next_query(state, evaluation)
        if next_query is None:
            break

        support_question = (
            evaluation.missing_sub_questions[0]
            if evaluation.missing_sub_questions
            else state.plan.question
        )
        _search_and_store(state, next_query, support_question)

    state.status = ResearchStatus.PARTIAL if state.evidence else ResearchStatus.FAILED
    return state


def _can_complete(state: ResearchState, evaluation: EvaluationResult) -> bool:
    if evaluation.status != EvaluationStatus.COMPLETE:
        return False
    if evaluation.missing_sub_questions:
        return False
    enough_sources = count_unique_sources(state) >= state.plan.minimum_sources
    return enough_sources


def _search_and_store(
    state: ResearchState,
    query: str,
    support_question: str,
) -> None:
    raw_result = search_web(query)
    parsed_evidence = parse_search_results(raw_result, support_question)
    add_evidence(state, parsed_evidence)
    state.searched_queries.add(query)
    state.step_count += 1


def _choose_next_query(
    state: ResearchState,
    evaluation: EvaluationResult,
) -> str | None:
    candidates: list[str] = []
    if evaluation.next_query:
        candidates.append(evaluation.next_query)
    candidates.extend(
        f"{question} reliable sources evidence"
        for question in evaluation.missing_sub_questions
    )

    for candidate in candidates:
        normalized = candidate.strip().casefold()
        if normalized and not _query_was_seen(state, normalized):
            return candidate.strip()
    return None


def _query_was_seen(state: ResearchState, normalized_query: str) -> bool:
    return any(
        query.strip().casefold() == normalized_query
        for query in state.searched_queries
    )

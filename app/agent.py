from openai import OpenAI

from app.client import create_client
from app.evidence import add_evidence, count_unique_sources, parse_search_results
from app.models import ResearchState, ResearchStatus
from app.planner import create_research_plan
from app.tools import search_web


MAX_RESEARCH_STEPS = 6


def run_research(
    question: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> ResearchState:
    """Plan a question, search each sub-question, and collect evidence."""
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

        raw_result = search_web(sub_question)
        parsed_evidence = parse_search_results(raw_result, sub_question)
        add_evidence(state, parsed_evidence)

        state.searched_queries.add(sub_question)
        state.step_count += 1

    state.status = _determine_status(state)
    return state


def _determine_status(state: ResearchState) -> ResearchStatus:
    if not state.evidence:
        return ResearchStatus.FAILED

    supported_questions = {
        question
        for item in state.evidence
        for question in item.supports
    }
    all_questions_supported = all(
        question in supported_questions for question in state.plan.sub_questions
    )
    enough_sources = count_unique_sources(state) >= state.plan.minimum_sources

    if all_questions_supported and enough_sources:
        return ResearchStatus.COMPLETED
    return ResearchStatus.PARTIAL


def _query_was_seen(state: ResearchState, normalized_query: str) -> bool:
    return any(
        query.strip().casefold() == normalized_query
        for query in state.searched_queries
    )

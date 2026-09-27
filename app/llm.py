from time import perf_counter

from app.agent import run_research
from app.client import create_client
from app.models import ResearchEvent, ResearchState
from app.writer import write_report


def ask_agent(question: str) -> str:
    """Compatibility wrapper that returns only the final report."""
    _, report = run_agent(question)
    return report


def run_agent(question: str) -> tuple[ResearchState, str]:
    """Run the complete research workflow and return state plus report."""
    if not question.strip():
        raise ValueError("请输入一个具体的研究问题。")

    started = perf_counter()
    client, model = create_client()
    state = run_research(question, client, model)
    report_started = perf_counter()
    report = write_report(state, client, model)
    state.trace.append(ResearchEvent(
        stage="report",
        step_count=state.step_count,
        elapsed_seconds=perf_counter() - report_started,
        details={
            "model": model,
            "generated": state.report_generated,
            "source_urls": dict(state.report_source_urls),
        },
    ))
    state.elapsed_seconds = perf_counter() - started
    return state, report

from app.agent import run_research
from app.client import create_client
from app.models import ResearchState
from app.writer import write_report


def ask_agent(question: str) -> str:
    """Compatibility wrapper that returns only the final report."""
    _, report = run_agent(question)
    return report


def run_agent(question: str) -> tuple[ResearchState, str]:
    """Run the complete research workflow and return state plus report."""
    if not question.strip():
        raise ValueError("请输入一个具体的研究问题。")

    client, model = create_client()
    state = run_research(question, client, model)
    report = write_report(state, client, model)
    return state, report

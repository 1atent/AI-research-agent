import json
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from app.evidence import count_unique_sources
from app.models import ResearchState


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def state_to_dict(state: ResearchState) -> dict[str, object]:
    """Serialize only research data; no environment variables or client objects."""
    data = asdict(state)
    data["searched_queries"] = sorted(state.searched_queries)
    return {
        "schema_version": 1,
        "metrics": {
            "search_count": state.step_count,
            "evaluation_count": sum(event.stage == "evaluation" for event in state.trace),
            "source_count": count_unique_sources(state),
            "elapsed_seconds": state.elapsed_seconds,
        },
        "state": data,
    }


def save_research(
    state: ResearchState,
    report: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Create an exclusive run folder with a Markdown report and JSON trace."""
    # Validate the internally generated ID before using it as a path component.
    run_id = UUID(state.run_id).hex
    run_dir = Path(output_dir).resolve() / run_id
    serialized = json.dumps(
        state_to_dict(state), ensure_ascii=False, indent=2, allow_nan=False
    )
    stop_reason = state.stop_reason.value if state.stop_reason else "unknown"
    metadata = (
        f"> 任务状态：{state.status.value}；停止原因：{stop_reason}\n"
        f"> 搜索次数：{state.step_count}；网页来源：{count_unique_sources(state)}；"
        f"总耗时：{state.elapsed_seconds:.2f} 秒\n"
        f"> 报告文本已生成：{'是' if state.report_generated else '否'}。"
        "引用内容是否支持结论仍需核验。\n\n"
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "report.md"
    state_path = run_dir / "state.json"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(metadata + report)
    with state_path.open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
    return report_path, state_path

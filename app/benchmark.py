"""Local benchmark runner. Only the explicit `run` command calls paid APIs."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.export import DEFAULT_OUTPUT_DIR, save_research, state_to_dict
from app.metrics import analyze_run, summarize_runs
from app.review import review_template, summarize_reviews


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = PROJECT_ROOT / "evals" / "cases.json"
CASE_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cases(path: str | Path) -> tuple[list[dict], str]:
    """Validate a dataset before creating artifacts or making any API call."""
    raw = Path(path).read_bytes()
    cases = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("题集必须是非空 JSON 数组。")
    if len(cases) > 100:
        raise ValueError("起步评测题集最多支持 100 个案例。")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("案例必须是 JSON 对象。")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError("案例 id 只能使用小写字母、数字和分隔用短横线。")
        if case_id in seen:
            raise ValueError(f"重复的案例 id：{case_id}")
        seen.add(case_id)
        for field in ("question", "category"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"案例 {case_id} 缺少非空 {field}。")
        checklist = case.get("checklist")
        if not isinstance(checklist, list) or not checklist or any(
            not isinstance(item, str) or not item.strip() for item in checklist
        ):
            raise ValueError(f"案例 {case_id} 缺少有效 checklist。")
    return cases, hashlib.sha256(raw).hexdigest()


def select_cases(cases: list[dict], case_ids: list[str] | None, limit: int) -> list[dict]:
    if limit < 1:
        raise ValueError("limit 必须大于 0。")
    if case_ids:
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("--case-id 不能重复。")
        by_id = {case["id"]: case for case in cases}
        if any(case_id not in by_id for case_id in case_ids):
            raise ValueError("--case-id 包含题集中不存在的案例。")
        cases = [by_id[case_id] for case_id in case_ids]
    return cases[:limit]


def _code_version() -> dict:
    """Best-effort provenance; never export a remote URL or local environment."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=3, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, timeout=3, check=True,
        ).stdout.strip())
        return {"git_commit": revision, "working_tree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "working_tree_dirty": None}


def _write_json(path: Path, payload: dict, *, exclusive: bool = False) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        handle.write(text)


def run_batch(cases: list[dict], output_dir: Path, label: str, dataset_hash: str) -> Path:
    # Lazy import: listing cases and offline analysis never initialize a client.
    from app.llm import run_agent

    batch = {
        "schema_version": 1,
        "batch_id": uuid4().hex,
        "created_at": _now(),
        "label": label,
        "dataset_sha256": dataset_hash,
        "code_version": _code_version(),
        "status": "running",
        "results": [
            {
                "case_id": case["id"], "category": case["category"],
                "question": case["question"], "checklist": list(case["checklist"]),
                "run_id": None, "execution_status": "pending", "model": None,
                "error": None, "error_type": None,
                "report_path": None, "state_path": None, "metrics": None,
            }
            for case in cases
        ],
    }
    batch_dir = output_dir.resolve() / batch["batch_id"]
    batch_dir.mkdir(parents=True, exist_ok=False)
    _write_json(batch_dir / "batch.json", batch, exclusive=True)
    print(f"[Benchmark] 即将执行 {len(cases)} 个案例，会调用模型和 Tavily API。")
    print(f"[Benchmark] 批次目录：{batch_dir}")
    for index, result in enumerate(batch["results"], start=1):
        print(f"\n[Benchmark] {index}/{len(cases)}：{result['case_id']}")
        try:
            # Do not send the review checklist to the agent being evaluated.
            state, report = run_agent(result["question"])
            result["run_id"] = state.run_id
            report_path, state_path = save_research(
                state, report, batch_dir / "artifacts" / result["case_id"]
            )
            result["report_path"] = report_path.relative_to(batch_dir).as_posix()
            result["state_path"] = state_path.relative_to(batch_dir).as_posix()
            result["metrics"] = analyze_run(state_to_dict(state), report)
            result["model"] = next(
                (event.details.get("model") for event in reversed(state.trace)
                 if event.stage == "report"), None,
            )
            result["execution_status"] = "finished"
        except Exception as exc:
            # Provider exceptions can contain headers/URLs; persist the type, not str(exc).
            result["error_type"] = type(exc).__name__
            result["error"] = f"案例执行失败（{type(exc).__name__}）。"
            result["execution_status"] = "error"
            print(f"[Benchmark error] {result['error']} 后续案例将继续执行。")
        _write_json(batch_dir / "batch.json", batch)
    batch["status"] = "finished"
    batch["finished_at"] = _now()
    _write_json(batch_dir / "batch.json", batch)
    return batch_dir


def _artifact_path(batch_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("产物路径必须是批次目录内的相对路径。")
    path = (batch_dir / value).resolve()
    if not path.is_relative_to(batch_dir):
        raise ValueError("产物路径不能越过批次目录。")
    return path


def _load_batch(batch_dir: Path) -> dict:
    batch = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    if not isinstance(batch, dict) or type(batch.get("schema_version")) is not int:
        raise ValueError("batch.json 缺少有效 schema_version。")
    if batch["schema_version"] != 1:
        raise ValueError("不支持的批次文件版本。")
    if not isinstance(batch.get("batch_id"), str):
        raise ValueError("批次文件缺少有效 batch_id。")
    UUID(batch["batch_id"])
    results = batch.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("批次 results 必须是非空列表。")
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("批次条目必须是 JSON 对象。")
        case_id = result.get("case_id")
        if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError("批次条目 case_id 无效。")
        if case_id in seen:
            raise ValueError("批次中存在重复 case_id。")
        seen.add(case_id)
        if "run_id" not in result:
            raise ValueError("批次条目缺少 run_id。")
        if result.get("execution_status") not in {"pending", "finished", "error"}:
            raise ValueError("批次条目 execution_status 无效。")
        if result["run_id"] is not None:
            if not isinstance(result["run_id"], str):
                raise ValueError("案例 run_id 必须是 UUID 字符串或 null。")
            UUID(result["run_id"])
    return batch


def analyze_batch(batch_dir: str | Path) -> dict:
    """Recompute from saved artifacts; never rerun research or overwrite labels."""
    batch_dir = Path(batch_dir).resolve()
    batch = _load_batch(batch_dir)
    for result in batch["results"]:
        result["metrics"] = None  # Do not trust cached metrics when artifacts have changed.
        if result.get("error") or result.get("execution_status") == "pending":
            continue
        try:
            state_path = _artifact_path(batch_dir, result.get("state_path"))
            report_path = _artifact_path(batch_dir, result.get("report_path"))
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if payload.get("state", {}).get("run_id") != result["run_id"]:
                raise ValueError("产物 run_id 与案例记录不匹配。")
            if payload.get("state", {}).get("plan", {}).get("question") != result.get("question"):
                raise ValueError("产物研究问题与案例不匹配。")
            result["metrics"] = analyze_run(payload, report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            result["error_type"] = type(exc).__name__
            result["error"] = f"离线分析失败（{type(exc).__name__}），请检查该案例产物。"

    review_path = batch_dir / "reviews.json"
    if not review_path.exists():
        _write_json(review_path, review_template(batch), exclusive=True)
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "batch_id": batch["batch_id"],
        "label": batch.get("label", ""),
        "dataset_sha256": batch.get("dataset_sha256"),
        "code_version": batch.get("code_version"),
        "batch_status": batch.get("status"),
        "pending_case_count": sum(
            result.get("execution_status") == "pending" for result in batch["results"]
        ),
        "analyzed_at": _now(),
        "metrics": summarize_runs(batch["results"]),
        "manual_review": summarize_reviews(batch, reviews),
        "results": batch["results"],
    }
    _write_json(batch_dir / "summary.json", summary)
    (batch_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _display(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "暂无数据"
    if isinstance(value, (int, float)):
        return f"{value:.1%}" if percent else f"{value:.2f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _summary_markdown(summary: dict) -> str:
    metrics, manual = summary["metrics"], summary["manual_review"]
    lines = [
        "# Research Agent 评测结果", "",
        f"批次：`{summary['batch_id']}`；标签：{_display(summary['label'])}", "",
        f"批次状态：{_display(summary['batch_status'])}；尚未记录完整运行结果：{summary['pending_case_count']} 题。", "",
        "> 自动完成率是程序状态统计，不是答案正确率。链接匹配只验证来源映射，不验证可达性或事实支持。",
        "> 此题集是开发用小样本，不是外部基准；修改工作区或搜索结果变化都会影响可比性。", "",
        f"- 案例数：{metrics['case_count']}",
        f"- 程序完成率：{_display(metrics['program_completion_rate'], percent=True)}",
        f"- 报告生成率：{_display(metrics['report_generation_rate'], percent=True)}",
        f"- 运行或产物错误数：{metrics['error_count']}",
        f"- 有运行指标的案例数：{metrics['measured_case_count']}",
        f"- 平均搜索次数：{_display(metrics['mean_search_count'])}",
        f"- 平均耗时（秒）：{_display(metrics['mean_elapsed_seconds'])}",
        f"- 人工完整评分：{manual['reviewed_count']}；待评分：{manual['pending_count']}",
        f"- 人工通过率（仅完整评分案例）：{_display(manual['human_pass_rate'], percent=True)}", "",
        "人工通过率必须连同已评案例数报告，不能把未评案例视为通过。", "",
        "| 案例 | 状态 | 搜索次数 | 耗时（秒） | 报告 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        data = result.get("metrics") or {}
        status = "error" if result.get("error") else data.get("status", "pending")
        link = f"[查看]({result['report_path']})" if result.get("report_path") else "无"
        lines.append(
            f"| {result['case_id']} | {status} | {_display(data.get('search_count'))} | "
            f"{_display(data.get('elapsed_seconds'))} | {link} |"
        )
    lines.extend(["", "## 人工复核清单", "", "按以下清单检查报告并填写 reviews.json，再执行 analyze 更新汇总。", ""])
    for result in summary["results"]:
        lines.extend([f"### {result['case_id']}", "", result.get("question", ""), ""])
        lines.extend(f"- {item}" for item in result.get("checklist", []))
        lines.append("")
    return "\n".join(lines) + "\n"


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="研究 Agent 固定题集与离线评测")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="查看题集，不调用 API")
    listing.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    running = commands.add_parser("run", help="真实运行，会消耗模型和 Tavily 额度")
    running.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    running.add_argument("--limit", type=_positive_int, default=2, help="本次最多案例数，默认 2")
    running.add_argument("--case-id", nargs="+", help="只运行指定案例，顺序与参数一致")
    running.add_argument("--label", default="baseline", help="本次版本标签，不作为目录名")
    running.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "benchmarks")
    analyzing = commands.add_parser("analyze", help="重新分析已有批次，不调用 API")
    analyzing.add_argument("batch_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            cases, digest = load_cases(args.cases)
            print(f"题集 SHA256：{digest}\n共 {len(cases)} 个开发评测案例；仅列出，不调用 API。")
            for case in cases:
                print(f"\n{case['id']} [{case['category']}]\n{case['question']}")
            return 0
        if args.command == "run":
            cases, digest = load_cases(args.cases)
            cases = select_cases(cases, args.case_id, args.limit)
            batch_dir = run_batch(cases, args.output_dir, args.label, digest)
        else:
            batch_dir = args.batch_dir.resolve()
        summary = analyze_batch(batch_dir)
        print(f"\n[Benchmark] 评测汇总：{batch_dir / 'summary.md'}")
        print(f"[Benchmark] 人工评分：{batch_dir / 'reviews.json'}")
        print(f"[Benchmark] 程序完成率：{_display(summary['metrics']['program_completion_rate'], percent=True)}")
        print(f"[Benchmark] 人工完整评分数：{summary['manual_review']['reviewed_count']}")
        return 1 if summary["metrics"]["error_count"] else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"[Benchmark error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

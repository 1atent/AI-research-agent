import argparse

from app.evidence import count_unique_sources
from app.export import DEFAULT_OUTPUT_DIR, save_research
from app.llm import run_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="规划、检索并生成带引用的研究报告")
    parser.add_argument("--question", help="研究问题；省略时交互输入")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="结果目录")
    args = parser.parse_args(argv)
    question = (
        args.question if args.question is not None else input("请输入研究问题：")
    ).strip()

    try:
        state, answer = run_agent(question)
    except Exception as exc:
        print(f"\n[Error] 任务执行失败：{exc}")
        return 1

    print("\n==============================")
    print("AI Research Agent")
    print("==============================")
    print(f"任务状态：{state.status.value}")
    print(f"停止原因：{state.stop_reason.value if state.stop_reason else 'unknown'}")
    print(f"搜索步数：{state.step_count}")
    print(f"证据数量：{len(state.evidence)}")
    print(f"不同来源：{count_unique_sources(state)}")
    print(f"总耗时：{state.elapsed_seconds:.2f} 秒")
    print("==============================\n")

    print(answer)
    try:
        report_path, state_path = save_research(state, answer, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"\n[Export error] 报告已显示，但文件保存失败：{exc}")
        return 1
    print(f"\n报告文件：{report_path}")
    print(f"执行记录：{state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from app.evidence import count_unique_sources
from app.llm import run_agent


def main():
    question = input("请输入研究问题：").strip()

    try:
        state, answer = run_agent(question)
    except Exception as exc:
        print(f"\n[Error] 任务执行失败：{exc}")
        return

    print("\n==============================")
    print("AI Research Agent")
    print("==============================")
    print(f"任务状态：{state.status.value}")
    print(f"搜索步数：{state.step_count}")
    print(f"证据数量：{len(state.evidence)}")
    print(f"不同来源：{count_unique_sources(state)}")
    print("==============================\n")

    print(answer)


if __name__ == "__main__":
    main()

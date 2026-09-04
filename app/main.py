from app.llm import ask_agent


def main():
    question = input("请输入研究问题：").strip()

    try:
        answer = ask_agent(question)
    except Exception as exc:
        print(f"\n[Error] 任务执行失败：{exc}")
        return

    print("\n==============================")
    print("AI Research Agent")
    print("==============================")

    print(answer)


if __name__ == "__main__":
    main()

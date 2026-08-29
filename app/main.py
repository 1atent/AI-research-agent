from llm import ask_llm


def main():
    question = input("请输入问题：")

    answer = ask_llm(question)

    print("\nAI：")
    print(answer)


if __name__ == "__main__":
    main()
def search_web(query: str) -> str:
    print(f"[Tool] 正在搜索：{query}")

    # 第一版先模拟搜索结果
    return (
        f"这是关于“{query}”的模拟搜索结果。"
        "真实版本将在这里接入 Web Search API。"
    )
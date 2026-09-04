# AI Research Agent

一个基于大语言模型 Tool Calling 的轻量研究 Agent。它能根据用户问题自主调用 Tavily 搜索，并根据多个网页来源生成带引用的回答。

## 已实现

- OpenAI 兼容模型接入
- Tavily 真实网页搜索
- 多轮工具调用
- 结构化搜索结果
- 重复查询检测
- 最大步数限制和失败降级
- Markdown 来源引用

## 运行

```powershell
conda activate <your-environment>
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m app.main
```

如果已经有 `.env`，不要执行复制命令。确保其包含：

```dotenv
API_KEY=...
BASE_URL=...
MODEL=...
TAVILY_API_KEY=...
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

## Agent 执行流程

1. 模型分析用户问题。
2. 需要外部证据时调用 `search_web`。
3. 程序执行 Tavily 搜索并把结构化结果返回模型。
4. 模型可继续发起补充搜索，或生成带来源的最终答案。
5. 达到最大步数时，程序强制停止搜索并要求模型总结现有证据。

## 当前边界

当前版本使用 Tavily 返回的搜索摘要，还没有实现网页全文抓取、任务规划器、证据校验器和持久知识库。

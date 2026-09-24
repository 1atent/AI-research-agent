# AI Research Agent

一个基于大语言模型规划、证据评估和 Tavily 检索的轻量研究 Agent。它能拆解用户问题、执行多轮补充搜索，并根据多个网页来源生成带引用的回答。

## 已实现

- OpenAI 兼容模型接入
- Tavily 真实网页搜索
- 多轮检索与补充查询
- 结构化搜索结果
- 重复查询检测
- 最大步数限制和失败降级
- Markdown 来源引用
- 研究问题规划与子问题拆解
- 结构化证据存储、URL 去重与证据合并
- 完成度评估与动态补充检索
- 受证据约束的 Markdown 报告生成
- 可观察的研究状态、搜索步数和来源数量

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

1. Planner 将用户问题拆解为可检索的子问题。
2. Agent 使用 Tavily 逐项搜索，并将结果转换为结构化 Evidence。
3. Evidence Store 按 URL 去重，并合并来源对不同子问题的支持关系。
4. Evaluator 判断已回答项、缺失项和下一次查询。
5. 程序同时检查来源数量、重复查询和最大步数。
6. Writer 只根据已收集证据生成报告，程序将来源编号替换为可验证链接。

## 当前边界

当前版本使用 Tavily 返回的搜索摘要，还没有进行网页全文抓取与句子级引用校验。Evaluator 和 Writer 仍然可能受到模型判断误差影响，因此后续需要建立固定评估集并统计完成率、引用有效率、延迟和成本。

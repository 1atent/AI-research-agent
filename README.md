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
- 来源不足时继续补充搜索，而不是直接结束
- 保留同一 URL 的互补摘要，优先将缺失主题和近期证据送入评估
- 结构化执行轨迹、耗时和明确的停止原因
- 每次运行独立保存 Markdown 报告与 JSON 研究记录
- 8 题固定开发评测集、批量运行和离线结果分析
- 区分程序完成率、引用链接映射检查与人工质量评分

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

也可以直接传入问题，不需要等待交互输入：

```powershell
python -m app.main --question "比较 RAG 与微调的适用场景、成本和限制"
```

自定义结果目录：

```powershell
python -m app.main --question "你的研究问题" --output-dir ./outputs
```

每次成功走完研究和写作流程后，默认在项目 `outputs/<run_id>/` 下生成：

- `report.md`：带任务状态、停止原因、搜索次数、耗时和来源引用的报告。
- `state.json`：计划、证据片段、查询历史、最终评估、来源编号映射和执行轨迹。

同一次运行的输出不会覆盖已有结果，不同运行有独立目录。`outputs/` 已被 Git 忽略。导出逻辑不读取环境变量，也不保存模型客户端或 API 请求头；研究问题和搜索内容仍可能包含隐私信息，分享前请检查。

JSON 顶层包含 `schema_version`、`metrics`、`state`。其中 `state.trace` 记录 `planning`、`search`、`evaluation`、`finish` 和 `report` 事件。这是程序执行记录和评估结论，不是模型隐藏思维链。`elapsed_seconds` 为墙钟耗时，未统计 token 用量或费用。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试使用模拟的模型响应和搜索结果，不消耗 API 额度。包含端到端命令行测试，验证补充搜索、再次评估、引用替换与文件导出能连起来工作；真实模型的研究质量需要另行验收。

## 固定题集与评测

评测设计将程序运行指标与人工质量判断分开，参考 [OpenAI 官方评测最佳实践](https://developers.openai.com/api/docs/guides/evaluation-best-practices)。本项目是本地轻量实现，不接入托管 Evals 服务、不增加新的依赖，也不增加自动评分模型费用。

先查看题集，此操作不调用 API：

```powershell
python -m app.benchmark list
```

然后真实运行少量案例，检查模型配置和输出（会消耗模型和 Tavily 额度）：

```powershell
python -m app.benchmark run --limit 2 --label baseline
```

`run` 默认最多运行 2 题。准备好后可使用 `--limit 8` 跑完起步题集，或用 `--case-id rag-concepts fictional-evidence-shortfall` 选择案例。`--cases` 可指定自己的 JSON 题集，`--output-dir` 可指定批次输出根目录。

每个批次在 `outputs/benchmarks/<batch_id>/` 下保存：

- `batch.json`：题目及人工检查清单快照、题集 SHA256、Git commit/工作区是否有修改、模型名、运行记录。Git 信息不可用时标为 `null`；commit 加 dirty 标记不是未提交源码快照，做正式对照前请自行保存代码版本。
- `artifacts/<case_id>/<run_id>/report.md` 和 `state.json`：每题产物。
- `summary.json` 和 `summary.md`：自动指标、人工评分汇总和逐题检查清单。
- `reviews.json`：待填写的人工评分，不预填虚构成绩。

批次先保存全部待执行条目，再逐题保存进度；单题异常记录错误类型并继续后续案例，不保存可能包含敏感请求信息的原始异常文本。终端中断后可以离线分析已保存题目，但不支持自动恢复剩余任务或保存模型异常时的部分研究状态。

填写该批次的 `reviews.json` 后，重新分析已有产物：

```powershell
python -m app.benchmark analyze outputs/benchmarks/<batch_id>
```

将 `<batch_id>` 替换为实际目录名。`analyze` 不调用模型或搜索 API，也不会覆盖已有人工评分；只重新生成派生汇总。批次/运行编号不匹配或评分格式非法时会报错，避免误用历史标签。

### 指标的含义与边界

- `program_completion_rate`：最终状态为 `completed` 的案例数 / 本批次全部选定案例数。运行失败、缺失产物和尚未执行的案例都留在分母中，不是事实正确率。
- `report_generation_rate`：收到报告文本的案例数 / 全部选定案例数，不表示报告质量合格。
- 平均搜索次数与平均耗时：仅基于有有效运行指标的案例；必须结合 `measured_case_count` 解读。异常中止的耗时与已消耗调用可能没有保留，不能用这些均值估算全部成本。
- `evaluator_coverage`：模型自评的已回答子问题比例，不是独立评测。评测题目的 `checklist` 不发送给被评 Agent，避免直接暗示判分要点。
- `citation_check`：仅检查报告正文 Markdown 行内链接是否匹配已保存的来源映射，排除文末参考列表；没有正文引用时比例为 `null`，不能算 100%。映射内链接不保证网页可访问，更不保证其支持附近结论。
- 人工评分：`coverage`、`grounding`、`source_quality` 分别填 0/1/2，`passed` 独立填布尔值。四项全部填写才计为完整评审；未审为 `null`，不能当成通过。

8 题只是开发起步小样本，不是外部基准。真实引用支持情况需要人工打开来源核验，暂未实现自动语义核验、HTTP 可达性检查、token/费用统计或版本差异显著性分析。完整评分锚点和操作说明见 [evals/README.md](evals/README.md)。

## Agent 执行流程

1. Planner 将用户问题拆解为可检索的子问题。
2. Agent 使用 Tavily 逐项搜索，并将结果转换为结构化 Evidence。
3. Evidence Store 按 URL 去重，合并互补摘要并记录候选子问题关联；关联本身不代表已证实支持该子问题。
4. Evaluator 判断已回答项、缺失项和下一次查询。
5. 程序同时检查来源数量、重复查询和最大步数。
6. Writer 只根据已收集证据生成报告，程序将来源编号替换为可验证链接。
7. 命令行显示结果并保存报告与结构化研究记录，便于复盘和后续评测。

## 如何判断任务结束

只有 Evaluator 返回 `complete`、所有计划内子问题被列为已回答、没有缺失项、并且去重后的有效网页来源数达到计划要求时，程序才接受完成。否则继续尝试新的查询；来源数不足时，即使模型认为主题已覆盖，也会尝试补充来源。

`status` 描述结果，`stop_reason` 解释为什么停止：

| stop_reason | 含义 | 最终 status |
| --- | --- | --- |
| `completed` | 模型评估与程序检查都通过 | `completed` |
| `search_budget_exhausted` | 已达到 8 次搜索尝试上限，仍未达标 | `partial` |
| `no_new_query` | 当前策略无法产生未执行过的查询 | `partial` |
| `no_evidence` | 最终没有可用证据 | `failed` |

没有证据时，`stop_reason` 为 `no_evidence`；`finish` 事件中的 `trigger` 仍保留实际触发停止的原因。搜索失败也消耗搜索预算；大小写和多余空白不同的相同查询不会重复执行。即使最后一次搜索耗尽预算，新增证据仍会先被评估。

评估器最多接收 20 个来源，写作器最多接收 12 个来源；选择时优先保留子问题覆盖与近期搜索结果，再按相关度补齐。每个来源最多保留 8 段、每段最多 4000 字符的摘要，传给模型时每个来源摘要预算为 1200 字符。这些边界限制上下文规模，也意味着模型未必看到全部收集内容。

## 当前边界

当前版本使用 Tavily 返回的搜索摘要，还没有进行网页全文抓取与句子级引用校验。来源数指去重后的网页 URL 数，不代表相互独立的机构数量，也不保证来源权威。`report_generated` 仅表示模型返回了报告文本，不代表报告已经事实核验。

Evaluator 和 Writer 仍然可能受到模型判断误差影响，`completed` 不是绝对正确性保证。模型 API 异常目前会由命令行提示并中止，不支持保存这类异常发生时的部分状态、重试或断点恢复。

下一阶段应运行固定题集并人工核查覆盖度、引用是否支持结论、来源质量和延迟；选择真实失败案例改进提示词或检索策略，再在相同题集上对照。token 和费用统计、全文抓取及句子级验证可以随后迭代。

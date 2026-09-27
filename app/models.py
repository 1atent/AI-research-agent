from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class ResearchStatus(str, Enum):
    """整个研究任务的状态。"""

    PLANNING = "planning"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class EvaluationStatus(str, Enum):
    """Evaluator 对当前资料是否充分的判断。"""

    CONTINUE = "continue"
    COMPLETE = "complete"


class StopReason(str, Enum):
    """为什么结束研究；与完成/部分完成/失败状态分别记录。"""

    COMPLETED = "completed"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    NO_NEW_QUERY = "no_new_query"
    NO_EVIDENCE = "no_evidence"


@dataclass
class ResearchPlan:
    """Planner 生成的研究计划。"""

    question: str
    sub_questions: list[str]
    minimum_sources: int = 3  # 最少需要的资料数量


@dataclass
class Evidence:
    """一条从搜索结果中获得的证据。"""

    title: str
    url: str
    content: str
    score: float | None = None
    supports: list[str] = field(default_factory=list)  # 这条证据"候选"支撑哪些子问题
    snippets: list[str] = field(default_factory=list)  # 同一网页从不同搜索返回的互补摘要
    last_seen_step: int = 0


@dataclass
class SearchOutcome:
    """区分搜索失败、成功但为空，以及有候选证据的结果。"""

    ok: bool
    evidence: list[Evidence] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvaluationResult:
    """Evaluator 对当前研究进度的检查结果。"""

    status: EvaluationStatus
    answered_sub_questions: list[str] = field(default_factory=list)  # 已经回答的子问题
    missing_sub_questions: list[str] = field(default_factory=list)  # 缺失的子问题
    next_query: str | None = None  # 下一次搜索的关键词
    reason: str = ""
    error: str | None = None  # 解析失败等可追踪错误；不保存原始模型输出


@dataclass
class ResearchEvent:
    """一次规划、搜索、评估或报告生成的结构化记录。"""

    stage: str
    step_count: int
    elapsed_seconds: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class ResearchState:
    """Agent 当前的完整运行状态。"""

    plan: ResearchPlan
    evidence: list[Evidence] = field(default_factory=list)
    searched_queries: set[str] = field(default_factory=set)  # 已经搜索过的关键词
    step_count: int = 0  # Agent 已经执行的步数
    status: ResearchStatus = ResearchStatus.PLANNING
    last_evaluation: EvaluationResult | None = None
    stop_reason: StopReason | None = None
    trace: list[ResearchEvent] = field(default_factory=list)
    elapsed_seconds: float = 0.0  # run_agent 包括规划、研究和报告的总耗时
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    report_source_urls: dict[str, str] = field(default_factory=dict)
    report_generated: bool = False  # 仅表示收到报告文本，不代表事实已经核验

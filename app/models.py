from dataclasses import dataclass, field
from enum import Enum


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
    supports: list[str] = field(default_factory=list)  # 这条证据可以支持哪些子问题


@dataclass
class ResearchState:
    """Agent 当前的完整运行状态。"""

    plan: ResearchPlan
    evidence: list[Evidence] = field(default_factory=list)
    searched_queries: set[str] = field(default_factory=set)  # 已经搜索过的关键词
    step_count: int = 0  # Agent 已经执行的步数
    status: ResearchStatus = ResearchStatus.PLANNING


@dataclass
class EvaluationResult:
    """Evaluator 对当前研究进度的检查结果。"""

    status: EvaluationStatus
    answered_sub_questions: list[str] = field(default_factory=list)  # 已经回答的子问题
    missing_sub_questions: list[str] = field(default_factory=list)  # 缺失的子问题
    next_query: str | None = None  # 下一次搜索的关键词
    reason: str = ""

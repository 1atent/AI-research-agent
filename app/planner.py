import json

from openai import OpenAI

from app.models import ResearchPlan


PLANNER_SYSTEM_PROMPT = """
你是一个研究任务规划器。

请将用户的研究问题拆解为 2 到 5 个互不重复、可以通过检索资料回答的子问题。

要求：
1. 所有子问题合起来必须覆盖用户的原始问题。
2. 不要生成意思重复的子问题。
3. 子问题应该具体，并且适合使用搜索引擎检索。
4. 简单问题不需要强行拆解成很多部分。
5. minimum_sources 应在 1 到 10 之间。
6. 只返回 JSON，不要返回 Markdown 或其他解释。

输出格式：
{
  "sub_questions": ["子问题 1", "子问题 2"],
  "minimum_sources": 3
}
""".strip()


def create_research_plan(
    question: str,
    client: OpenAI,
    model: str,
) -> ResearchPlan:
    """Ask the LLM for a research plan and validate its response."""
    question = question.strip()
    if not question:
        raise ValueError("研究问题不能为空。")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        return create_fallback_plan(question)

    return parse_research_plan(content, question)


def parse_research_plan(content: str, original_question: str) -> ResearchPlan:
    """Convert untrusted model output into a safe ResearchPlan."""
    original_question = original_question.strip()
    try:
        data = json.loads(_remove_markdown_fence(content))
    except (json.JSONDecodeError, TypeError):
        return create_fallback_plan(original_question)

    if not isinstance(data, dict):
        return create_fallback_plan(original_question)

    raw_sub_questions = data.get("sub_questions")
    if not isinstance(raw_sub_questions, list):
        return create_fallback_plan(original_question)

    sub_questions: list[str] = []
    seen: set[str] = set()
    for item in raw_sub_questions:
        if not isinstance(item, str):
            continue

        cleaned = item.strip()
        normalized = cleaned.casefold()
        if not cleaned or normalized in seen:
            continue

        sub_questions.append(cleaned)
        seen.add(normalized)
        if len(sub_questions) == 5:
            break

    if not sub_questions:
        return create_fallback_plan(original_question)

    minimum_sources = _normalize_minimum_sources(data.get("minimum_sources", 3))
    return ResearchPlan(
        question=original_question,
        sub_questions=sub_questions,
        minimum_sources=minimum_sources,
    )


def create_fallback_plan(question: str) -> ResearchPlan:
    """Create a usable plan when the model output cannot be parsed."""
    return ResearchPlan(
        question=question.strip(),
        sub_questions=[question.strip()],
        minimum_sources=3,
    )


def _normalize_minimum_sources(value: object) -> int:
    try:
        minimum_sources = int(value)
    except (TypeError, ValueError):
        minimum_sources = 3
    return max(1, min(minimum_sources, 10))


def _remove_markdown_fence(content: str) -> str:
    """Remove an optional ```json wrapper while keeping parsing strict."""
    content = content.strip()
    if not content.startswith("```"):
        return content

    lines = content.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

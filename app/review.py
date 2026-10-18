"""Validate human labels without treating an agent's self-evaluation as truth."""

from statistics import mean


SCORE_FIELDS = ("coverage", "grounding", "source_quality")


def review_template(batch: dict) -> dict:
    return {
        "schema_version": 1,
        "batch_id": batch["batch_id"],
        "reviews": [
            {
                "case_id": result["case_id"],
                "run_id": result["run_id"],
                "passed": None,
                **{field: None for field in SCORE_FIELDS},
                "notes": "",
            }
            for result in batch["results"]
        ],
    }


def summarize_reviews(batch: dict, payload: dict) -> dict:
    """Only fully labelled, identity-matched rows contribute to quality scores."""
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError("人工评分文件缺少有效 schema_version。")
    if payload["schema_version"] != 1 or payload.get("batch_id") != batch["batch_id"]:
        raise ValueError("人工评分文件版本或 batch_id 与当前批次不匹配。")
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("reviews 必须是列表。")
    by_id = {result["case_id"]: result for result in batch["results"]}
    seen: set[str] = set()
    reviewed: list[dict] = []
    for row in reviews:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError("评分条目缺少有效 case_id。")
        case_id = row["case_id"]
        if case_id not in by_id or case_id in seen:
            raise ValueError("评分包含未知或重复 case_id。")
        seen.add(case_id)
        result = by_id[case_id]
        if row.get("run_id") != result["run_id"]:
            raise ValueError("评分 run_id 与本次运行不匹配，不能沿用旧运行的评分。")
        passed = row.get("passed")
        if passed is not None and type(passed) is not bool:
            raise ValueError("passed 必须是 true、false 或 null。")
        for field in SCORE_FIELDS:
            value = row.get(field)
            if value is not None and (type(value) is not int or value not in (0, 1, 2)):
                raise ValueError(f"{field} 必须是 0、1、2 或 null。")
        if not isinstance(row.get("notes", ""), str):
            raise ValueError("notes 必须是字符串。")
        if passed is True and (
            result.get("error") or not (result.get("metrics") or {}).get("report_generated")
        ):
            raise ValueError("运行失败或未生成报告的条目不能被标为人工通过。")
        if passed is not None and all(row.get(field) is not None for field in SCORE_FIELDS):
            reviewed.append(row)

    return {
        "reviewed_count": len(reviewed),
        "pending_count": len(by_id) - len(reviewed),
        "human_pass_count": sum(row["passed"] for row in reviewed),
        "human_pass_rate": mean(row["passed"] for row in reviewed) if reviewed else None,
        **{
            f"mean_{field}": mean(row[field] for row in reviewed) if reviewed else None
            for field in SCORE_FIELDS
        },
    }

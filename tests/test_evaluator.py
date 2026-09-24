import unittest
from unittest.mock import Mock

from app.evaluator import evaluate_research, parse_evaluation
from app.models import (
    Evidence,
    EvaluationStatus,
    ResearchPlan,
    ResearchState,
)


class ParseEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.state = _state_with_evidence()

    def test_complete_result_is_accepted_when_all_questions_are_answered(self):
        content = """
        {
          "status": "complete",
          "answered_sub_questions": ["q1", "q2"],
          "missing_sub_questions": [],
          "next_query": null,
          "reason": "all covered"
        }
        """

        result = parse_evaluation(content, self.state)

        self.assertEqual(result.status, EvaluationStatus.COMPLETE)
        self.assertEqual(result.missing_sub_questions, [])

    def test_omitted_question_forces_continue(self):
        content = """
        {
          "status": "complete",
          "answered_sub_questions": ["q1"],
          "missing_sub_questions": [],
          "next_query": null,
          "reason": "incorrectly declared complete"
        }
        """

        result = parse_evaluation(content, self.state)

        self.assertEqual(result.status, EvaluationStatus.CONTINUE)
        self.assertEqual(result.missing_sub_questions, ["q2"])
        self.assertEqual(result.next_query, "q2")

    def test_invalid_json_returns_safe_fallback(self):
        result = parse_evaluation("not json", self.state)

        self.assertEqual(result.status, EvaluationStatus.CONTINUE)
        self.assertEqual(result.missing_sub_questions, ["q1", "q2"])


class EvaluateResearchTests(unittest.TestCase):
    def test_no_evidence_does_not_call_model(self):
        state = ResearchState(
            plan=ResearchPlan(question="question", sub_questions=["q1"])
        )
        client = Mock()

        result = evaluate_research(state, client, "test-model")

        self.assertEqual(result.status, EvaluationStatus.CONTINUE)
        client.chat.completions.create.assert_not_called()


def _state_with_evidence() -> ResearchState:
    return ResearchState(
        plan=ResearchPlan(question="question", sub_questions=["q1", "q2"]),
        evidence=[
            Evidence(
                title="Source",
                url="https://example.com",
                content="evidence",
                supports=["q1", "q2"],
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()

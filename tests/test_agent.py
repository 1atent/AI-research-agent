import json
import unittest
from unittest.mock import Mock, patch

from app.agent import run_research
from app.models import (
    EvaluationResult,
    EvaluationStatus,
    ResearchPlan,
    ResearchStatus,
    StopReason,
)


class RunResearchTests(unittest.TestCase):
    def test_client_and_model_must_be_provided_together(self):
        with self.assertRaises(ValueError):
            run_research("question", client=Mock(), model=None)

    @patch("app.agent.evaluate_research")
    @patch("app.agent.search_web")
    @patch("app.agent.create_research_plan")
    def test_research_collects_evidence_and_completes(
        self, mock_plan, mock_search, mock_evaluate
    ):
        mock_plan.return_value = ResearchPlan(
            question="compare",
            sub_questions=["q1", "q2"],
            minimum_sources=2,
        )
        mock_search.side_effect = [
            _search_result("Source 1", "https://example.com/1"),
            _search_result("Source 2", "https://example.com/2"),
        ]
        mock_evaluate.return_value = EvaluationResult(
            status=EvaluationStatus.COMPLETE,
            answered_sub_questions=["q1", "q2"],
            missing_sub_questions=[],
            reason="complete",
        )

        state = run_research("compare", Mock(), "test-model")

        self.assertEqual(state.status, ResearchStatus.COMPLETED)
        self.assertEqual(state.step_count, 2)
        self.assertEqual(len(state.evidence), 2)
        self.assertEqual(state.searched_queries, {"q1", "q2"})

    @patch("app.agent.search_web")
    @patch("app.agent.create_research_plan")
    def test_no_evidence_marks_task_as_failed(self, mock_plan, mock_search):
        mock_plan.return_value = ResearchPlan(
            question="question",
            sub_questions=["q1"],
            minimum_sources=1,
        )
        mock_search.return_value = '{"ok": false, "error": "timeout"}'

        state = run_research("question", Mock(), "test-model")

        self.assertEqual(state.status, ResearchStatus.FAILED)
        self.assertEqual(state.stop_reason, StopReason.NO_EVIDENCE)
        self.assertLessEqual(state.step_count, 8)
        searches = [event for event in state.trace if event.stage == "search"]
        self.assertTrue(all(event.details["error"] == "timeout" for event in searches))

    @patch("app.agent.evaluate_research")
    @patch("app.agent.search_web")
    @patch("app.agent.create_research_plan")
    def test_insufficient_sources_marks_task_as_partial(
        self, mock_plan, mock_search, mock_evaluate
    ):
        mock_plan.return_value = ResearchPlan(
            question="question",
            sub_questions=["q1"],
            minimum_sources=2,
        )
        mock_search.return_value = _search_result(
            "Only source", "https://example.com/only"
        )
        mock_evaluate.return_value = EvaluationResult(
            status=EvaluationStatus.CONTINUE,
            answered_sub_questions=[],
            missing_sub_questions=["q1"],
            next_query=None,
            reason="insufficient evidence",
        )

        state = run_research("question", Mock(), "test-model")

        self.assertEqual(state.status, ResearchStatus.PARTIAL)


class ResearchLoopRegressionTests(unittest.TestCase):
    def setUp(self):
        self.plan_patch = patch("app.agent.create_research_plan")
        self.search_patch = patch("app.agent.search_web")
        self.evaluation_patch = patch("app.agent.evaluate_research")
        self.plan = self.plan_patch.start()
        self.search = self.search_patch.start()
        self.evaluate = self.evaluation_patch.start()
        self.addCleanup(self.plan_patch.stop)
        self.addCleanup(self.search_patch.stop)
        self.addCleanup(self.evaluation_patch.stop)
        self.plan.return_value = ResearchPlan("question", ["q1"], minimum_sources=2)
        self.complete = EvaluationResult(
            EvaluationStatus.COMPLETE, answered_sub_questions=["q1"], reason="covered"
        )

    def test_all_questions_answered_but_source_shortfall_triggers_search(self):
        self.search.side_effect = [
            _search_result("First", "https://example.com/first"),
            _search_result("Second", "https://example.com/second"),
        ]
        self.evaluate.return_value = self.complete
        state = run_research("question", Mock(), "test-model")
        self.assertEqual(state.status, ResearchStatus.COMPLETED)
        self.assertEqual(state.stop_reason, StopReason.COMPLETED)
        self.assertEqual(self.search.call_count, 2)
        evaluations = [event for event in state.trace if event.stage == "evaluation"]
        self.assertFalse(evaluations[0].details["completion_accepted"])
        self.assertEqual(evaluations[0].details["source_shortfall"], 1)
        self.assertIsNotNone(evaluations[0].details["chosen_query"])
        self.assertTrue(evaluations[-1].details["completion_accepted"])

    def test_last_budgeted_search_is_evaluated_before_stopping(self):
        self.search.side_effect = [
            _search_result("First", "https://example.com/first"),
            _search_result("Second", "https://example.com/second"),
        ]
        self.evaluate.return_value = self.complete
        with patch("app.agent.MAX_RESEARCH_STEPS", 2):
            state = run_research("question", Mock(), "test-model")
        self.assertEqual(state.status, ResearchStatus.COMPLETED)
        self.assertEqual(self.evaluate.call_count, 2)

    def test_budget_exhaustion_prevents_further_searches(self):
        self.search.return_value = _search_result("First", "https://example.com/first")
        self.evaluate.return_value = EvaluationResult(
            EvaluationStatus.CONTINUE, missing_sub_questions=["q1"],
            next_query="new query", reason="missing detail",
        )
        with patch("app.agent.MAX_RESEARCH_STEPS", 2):
            state = run_research("question", Mock(), "test-model")
        self.assertEqual(state.status, ResearchStatus.PARTIAL)
        self.assertEqual(state.stop_reason, StopReason.SEARCH_BUDGET_EXHAUSTED)
        self.assertEqual(self.search.call_count, 2)
        self.assertEqual(state.last_evaluation.reason, "missing detail")

    def test_repeated_queries_stop_without_consuming_search_budget(self):
        self.plan.return_value.minimum_sources = 1
        self.search.return_value = _search_result("First", "https://example.com/first")
        self.evaluate.return_value = EvaluationResult(
            EvaluationStatus.CONTINUE, missing_sub_questions=["q1"],
            next_query="   Q1   ", reason="need more",
        )
        state = run_research("question", Mock(), "test-model")
        self.assertEqual(state.stop_reason, StopReason.NO_NEW_QUERY)
        queries = [call.args[0].strip().casefold() for call in self.search.call_args_list]
        self.assertEqual(len(queries), len(set(queries)))

    def test_empty_result_has_no_error_but_task_fails_with_no_evidence(self):
        self.search.return_value = '{"ok": true, "results": []}'
        self.evaluate.return_value = EvaluationResult(
            EvaluationStatus.CONTINUE, missing_sub_questions=["q1"]
        )
        state = run_research("question", Mock(), "test-model")
        searches = [event for event in state.trace if event.stage == "search"]
        self.assertEqual(state.status, ResearchStatus.FAILED)
        self.assertEqual(state.stop_reason, StopReason.NO_EVIDENCE)
        self.assertTrue(all(event.details["ok"] for event in searches))
        self.assertTrue(all(event.details["error"] is None for event in searches))

    def test_model_complete_without_answered_questions_is_not_accepted(self):
        self.plan.return_value.minimum_sources = 1
        self.search.return_value = _search_result("First", "https://example.com/first")
        self.evaluate.return_value = EvaluationResult(EvaluationStatus.COMPLETE)
        state = run_research("question", Mock(), "test-model")
        self.assertNotEqual(state.status, ResearchStatus.COMPLETED)


def _search_result(title: str, url: str) -> str:
    return json.dumps(
        {
            "ok": True,
            "results": [
                {
                    "title": title,
                    "url": url,
                    "content": "evidence",
                    "score": 0.9,
                }
            ],
        }
    )


if __name__ == "__main__":
    unittest.main()

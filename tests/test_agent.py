import json
import unittest
from unittest.mock import Mock, patch

from app.agent import run_research
from app.models import ResearchPlan, ResearchStatus


class RunResearchTests(unittest.TestCase):
    def test_client_and_model_must_be_provided_together(self):
        with self.assertRaises(ValueError):
            run_research("question", client=Mock(), model=None)

    @patch("app.agent.search_web")
    @patch("app.agent.create_research_plan")
    def test_research_collects_evidence_and_completes(self, mock_plan, mock_search):
        mock_plan.return_value = ResearchPlan(
            question="compare",
            sub_questions=["q1", "q2"],
            minimum_sources=2,
        )
        mock_search.side_effect = [
            _search_result("Source 1", "https://example.com/1"),
            _search_result("Source 2", "https://example.com/2"),
        ]

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
        self.assertEqual(state.step_count, 1)

    @patch("app.agent.search_web")
    @patch("app.agent.create_research_plan")
    def test_insufficient_sources_marks_task_as_partial(self, mock_plan, mock_search):
        mock_plan.return_value = ResearchPlan(
            question="question",
            sub_questions=["q1"],
            minimum_sources=2,
        )
        mock_search.return_value = _search_result(
            "Only source", "https://example.com/only"
        )

        state = run_research("question", Mock(), "test-model")

        self.assertEqual(state.status, ResearchStatus.PARTIAL)


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

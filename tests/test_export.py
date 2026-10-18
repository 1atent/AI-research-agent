import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.export import save_research
from app.models import (
    Evidence, ResearchEvent, ResearchPlan, ResearchState, ResearchStatus, StopReason,
)


class ExportTests(unittest.TestCase):
    def test_report_and_trace_roundtrip_with_unicode_and_sets(self):
        state = ResearchState(
            plan=ResearchPlan("中文问题", ["子问题"]),
            evidence=[Evidence("来源", "https://example.com", "证据")],
            status=ResearchStatus.PARTIAL,
            stop_reason=StopReason.NO_NEW_QUERY,
            searched_queries={"query-b", "query-a"},
            step_count=2,
            elapsed_seconds=1.5,
            report_generated=True,
        )
        state.trace.append(ResearchEvent("search", 1, 0.5, {"query": "query-a"}))
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict("os.environ", {"API_KEY": "SECRET_SENTINEL"}):
                report_path, state_path = save_research(state, "研究报告", output_dir)
            raw_json = state_path.read_text(encoding="utf-8")
            payload = json.loads(raw_json)
            self.assertIn("研究报告", report_path.read_text(encoding="utf-8"))
            self.assertIn("停止原因：no_new_query", report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"]["searched_queries"], ["query-a", "query-b"])
            self.assertEqual(payload["state"]["status"], "partial")
            self.assertEqual(payload["metrics"]["source_count"], 1)
            self.assertEqual(payload["state"]["trace"][0]["stage"], "search")
            self.assertNotIn("SECRET_SENTINEL", raw_json)
            self.assertNotIn("API_KEY", raw_json)

    def test_same_run_cannot_overwrite_existing_results(self):
        state = ResearchState(ResearchPlan("question", ["q1"]))
        with tempfile.TemporaryDirectory() as output_dir:
            report_path, _ = save_research(state, "original", output_dir)
            with self.assertRaises(FileExistsError):
                save_research(state, "replacement", output_dir)
            self.assertIn("original", report_path.read_text(encoding="utf-8"))

    def test_separate_runs_have_distinct_output_folders(self):
        with tempfile.TemporaryDirectory() as output_dir:
            first, _ = save_research(ResearchState(ResearchPlan("q", ["q"])), "one", output_dir)
            second, _ = save_research(ResearchState(ResearchPlan("q", ["q"])), "two", output_dir)
            self.assertNotEqual(first.parent, second.parent)
            self.assertEqual(first.parent.parent, Path(output_dir).resolve())


if __name__ == "__main__":
    unittest.main()

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from app.main import main


def _response(content: dict | str) -> Mock:
    if isinstance(content, dict):
        content = json.dumps(content)
    return Mock(choices=[Mock(message=Mock(content=content))])


class WorkflowTests(unittest.TestCase):
    @patch("app.agent.search_web")
    @patch("app.llm.create_client")
    def test_cli_runs_real_orchestration_and_exports_trace(self, mock_client, mock_search):
        client = Mock()
        mock_client.return_value = client, "test-model"
        client.chat.completions.create.side_effect = [
            _response({"sub_questions": ["q1"], "minimum_sources": 2}),
            _response({"status": "complete", "answered_sub_questions": ["q1"],
                       "missing_sub_questions": [], "reason": "covered"}),
            _response({"status": "complete", "answered_sub_questions": ["q1"],
                       "missing_sub_questions": [], "reason": "covered"}),
            _response("结论 [S1]。补充信息 [S2]。"),
        ]
        mock_search.side_effect = [
            json.dumps({"ok": True, "results": [{
                "title": f"来源{i}", "url": f"https://example.com/{i}",
                "content": f"证据{i}", "score": 0.9,
            }]}) for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = main(["--question", "研究问题", "--output-dir", output_dir])
            state_path = next(Path(output_dir).glob("*/state.json"))
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            report = (state_path.parent / "report.md").read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["state"]["status"], "completed")
            self.assertEqual(payload["state"]["stop_reason"], "completed")
            self.assertEqual(payload["metrics"]["search_count"], 2)
            self.assertEqual(payload["metrics"]["evaluation_count"], 2)
            self.assertEqual(len(payload["state"]["report_source_urls"]), 2)
            self.assertTrue(payload["state"]["report_generated"])
            self.assertIn("https://example.com/0", report)
            self.assertIn("补充查询", stdout.getvalue())
            self.assertEqual(client.chat.completions.create.call_count, 4)


if __name__ == "__main__":
    unittest.main()

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.benchmark import (
    DEFAULT_CASES, analyze_batch, load_cases, main, run_batch, select_cases,
)
from app.models import (
    Evidence, EvaluationResult, EvaluationStatus, ResearchEvent,
    ResearchPlan, ResearchState, ResearchStatus, StopReason,
)


def cases():
    return [
        {"id": f"case-{index}", "category": "概念解释", "question": f"问题{index}",
         "checklist": ["检查是否有支持结论的引用"]}
        for index in range(2)
    ]


def fake_run(question):
    state = ResearchState(
        plan=ResearchPlan(question, [question], minimum_sources=1),
        status=ResearchStatus.COMPLETED, stop_reason=StopReason.COMPLETED,
        evidence=[Evidence("来源", "https://example.com", "证据")],
        searched_queries={question}, step_count=1, elapsed_seconds=0.5,
        last_evaluation=EvaluationResult(
            EvaluationStatus.COMPLETE, answered_sub_questions=[question]
        ),
        report_source_urls={"S1": "https://example.com"}, report_generated=True,
        trace=[ResearchEvent("report", 1, 0.1, {"model": "mock-model"})],
    )
    return state, "结论 [来源](https://example.com)。\n\n## 参考来源\n- 来源"


class CaseDatasetTests(unittest.TestCase):
    def test_default_dataset_contains_eight_distinct_valid_cases(self):
        dataset, digest = load_cases(DEFAULT_CASES)
        self.assertEqual(len(dataset), 8)
        self.assertEqual(len({case["id"] for case in dataset}), 8)
        self.assertEqual(len(digest), 64)

    def test_invalid_case_schemas_are_rejected(self):
        invalid = [[], {}, [None], cases() + [cases()[0]]]
        for field, value in (("id", "../outside"), ("id", "UPPER"),
                             ("question", " "), ("category", None),
                             ("checklist", []), ("checklist", [None])):
            item = cases()[0]
            item[field] = value
            invalid.append([item])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            for payload in invalid:
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    load_cases(path)

    def test_selection_respects_explicit_order_and_limit(self):
        selected = select_cases(cases(), ["case-1", "case-0"], 1)
        self.assertEqual([item["id"] for item in selected], ["case-1"])
        for ids, limit in ((["missing"], 2), (["case-0", "case-0"], 2), (None, 0)):
            with self.subTest(ids=ids, limit=limit), self.assertRaises(ValueError):
                select_cases(cases(), ids, limit)

    @patch("app.llm.run_agent", side_effect=AssertionError("must stay offline"))
    def test_listing_does_not_run_research(self, mock_run):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["list"]), 0)
        self.assertIn("rag-concepts", output.getvalue())
        mock_run.assert_not_called()


class BatchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name)
        self.runner = patch("app.llm.run_agent", side_effect=fake_run).start()
        self.addCleanup(patch.stopall)
        patch("app.benchmark._code_version", return_value={
            "git_commit": "test-commit", "working_tree_dirty": True,
        }).start()
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def make_batch(self):
        return run_batch(cases(), self.output_dir, "baseline", "dataset-hash")

    def test_batch_exports_all_artifacts_and_blank_reviews(self):
        folder = self.make_batch()
        summary = analyze_batch(folder)
        manifest = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["metrics"]["case_count"], 2)
        self.assertEqual(summary["metrics"]["program_completion_rate"], 1)
        self.assertIsNone(summary["manual_review"]["human_pass_rate"])
        self.assertEqual(summary["manual_review"]["pending_count"], 2)
        self.assertEqual(manifest["status"], "finished")
        self.assertEqual(manifest["dataset_sha256"], "dataset-hash")
        self.assertTrue((folder / "summary.md").is_file())
        for result in summary["results"]:
            self.assertTrue((folder / result["report_path"]).is_file())
            self.assertTrue((folder / result["state_path"]).is_file())
            self.assertEqual(result["model"], "mock-model")
            self.assertEqual(result["metrics"]["citation_check"]["known_inline_link_count"], 1)
        self.assertEqual([call.args for call in self.runner.call_args_list], [("问题0",), ("问题1",)])

    def test_failure_is_retained_and_next_case_runs_without_exporting_exception_secret(self):
        self.runner.side_effect = [RuntimeError("SECRET_SENTINEL"), fake_run("问题1")]
        folder = self.make_batch()
        summary = analyze_batch(folder)
        self.assertEqual(self.runner.call_count, 2)
        self.assertEqual(summary["metrics"]["error_count"], 1)
        self.assertEqual(summary["metrics"]["program_completion_rate"], 0.5)
        self.assertEqual(summary["metrics"]["measured_case_count"], 1)
        for path in (folder / "batch.json", folder / "summary.json", folder / "summary.md"):
            self.assertNotIn("SECRET_SENTINEL", path.read_text(encoding="utf-8"))

    def test_analysis_reuses_artifacts_preserves_labels_and_makes_no_calls(self):
        folder = self.make_batch()
        analyze_batch(folder)
        reviews_path = folder / "reviews.json"
        reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
        reviews["reviews"][0].update(passed=True, coverage=2, grounding=2, source_quality=1)
        reviews_path.write_text(json.dumps(reviews), encoding="utf-8")
        before = reviews_path.read_bytes()
        self.runner.reset_mock(side_effect=True)
        self.runner.side_effect = AssertionError("offline analysis must not run research")
        summary = analyze_batch(folder)
        self.runner.assert_not_called()
        self.assertEqual(reviews_path.read_bytes(), before)
        self.assertEqual(summary["manual_review"]["reviewed_count"], 1)
        self.assertEqual(summary["manual_review"]["pending_count"], 1)
        self.assertEqual(summary["manual_review"]["human_pass_rate"], 1)

    def test_distinct_batches_do_not_overwrite_results(self):
        first, second = self.make_batch(), self.make_batch()
        self.assertNotEqual(first, second)
        self.assertTrue((first / "batch.json").is_file())

    def test_missing_artifact_is_error_not_success(self):
        folder = self.make_batch()
        manifest = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
        manifest["results"][0]["state_path"] = "missing/state.json"
        (folder / "batch.json").write_text(json.dumps(manifest), encoding="utf-8")
        summary = analyze_batch(folder)
        self.assertEqual(summary["metrics"]["program_completion_rate"], 0.5)
        self.assertEqual(summary["metrics"]["error_count"], 1)

    def test_state_identity_must_match_case(self):
        folder = self.make_batch()
        manifest = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
        state_path = folder / manifest["results"][0]["state_path"]
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        payload["state"]["run_id"] = manifest["results"][1]["run_id"]
        state_path.write_text(json.dumps(payload), encoding="utf-8")
        summary = analyze_batch(folder)
        self.assertEqual(summary["metrics"]["error_count"], 1)

    def test_artifact_cannot_escape_batch_directory(self):
        folder = self.make_batch()
        manifest = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
        manifest["results"][0]["state_path"] = "../outside.json"
        (folder / "batch.json").write_text(json.dumps(manifest), encoding="utf-8")
        summary = analyze_batch(folder)
        self.assertEqual(summary["results"][0]["error_type"], "ValueError")

    def test_pending_cases_stay_in_denominator_after_interruption(self):
        self.runner.side_effect = [fake_run("问题0"), KeyboardInterrupt]
        with self.assertRaises(KeyboardInterrupt):
            self.make_batch()
        folder = next(self.output_dir.iterdir())
        summary = analyze_batch(folder)
        self.assertEqual(summary["batch_status"], "running")
        self.assertEqual(summary["pending_case_count"], 1)
        self.assertEqual(summary["metrics"]["case_count"], 2)
        self.assertEqual(summary["metrics"]["program_completion_rate"], 0.5)

    def test_cli_run_defaults_to_two_and_cli_analysis_is_offline(self):
        self.assertEqual(main(["run", "--output-dir", str(self.output_dir)]), 0)
        self.assertEqual(self.runner.call_count, 2)
        folder = next(self.output_dir.iterdir())
        self.runner.reset_mock()
        self.assertEqual(main(["analyze", str(folder)]), 0)
        self.runner.assert_not_called()

    def test_invalid_batch_file_returns_error_not_traceback(self):
        folder = self.output_dir / "bad-batch"
        folder.mkdir()
        for payload in ([], {}, {"schema_version": 1, "batch_id": None}):
            (folder / "batch.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(payload=payload):
                self.assertEqual(main(["analyze", str(folder)]), 1)


if __name__ == "__main__":
    unittest.main()

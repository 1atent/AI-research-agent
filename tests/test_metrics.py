import copy
import unittest

from app.metrics import analyze_run, summarize_runs


def exported_run():
    return {
        "schema_version": 1,
        "metrics": {"search_count": 3, "evaluation_count": 2, "source_count": 2, "elapsed_seconds": 4.5},
        "state": {
            "status": "completed", "stop_reason": "completed", "report_generated": True,
            "plan": {"sub_questions": ["q1", "q2"]},
            "last_evaluation": {"answered_sub_questions": ["q1", "q1", "unplanned"]},
            "report_source_urls": {"S1": "https://example.com/a", "S2": "https://example.com/b_(topic)"},
            "trace": [
                {"stage": "search", "details": {"ok": False, "error": "timeout"}},
                {"stage": "search", "details": {"ok": True}},
                {"stage": "evaluation", "details": {"error": "invalid_json", "result": {"error": "invalid_json"}}},
                {"stage": "evaluation", "details": {"error": None}},
            ],
        },
    }


class AnalyzeRunTests(unittest.TestCase):
    def test_reports_operational_counts_and_evaluator_self_assessment(self):
        result = analyze_run(exported_run(), "结论 [来源](https://example.com/a)")
        self.assertEqual(result["evaluator_coverage"], 0.5)
        self.assertEqual(result["search_count"], 3)
        self.assertEqual(result["evaluation_count"], 2)
        self.assertEqual(result["search_error_count"], 1)
        self.assertEqual(result["evaluation_error_count"], 1)

    def test_reference_list_is_not_body_citation(self):
        report = "没有引用的结论。\n\n## 参考来源\n- [来源](https://example.com/a)\n- [unknown source]"
        citations = analyze_run(exported_run(), report)["citation_check"]
        self.assertEqual(citations["inline_link_count"], 0)
        self.assertIsNone(citations["known_inline_link_rate"])
        self.assertFalse(citations["has_inline_citations"])
        self.assertEqual(citations["unknown_source_count"], 0)

    def test_english_and_chinese_reference_sections_are_not_body_citations(self):
        for heading in (
            "## References", "## REFERENCES", "## Sources", "## sources",
            "## Bibliography", "## 参考文献", "## 引用来源", "## 参考来源",
            "# References", "### Sources ###", "###### 参考文献：", "## References\r",
        ):
            report = (
                f"Uncited claim.\n\n{heading}\n- [Source](https://example.com/a)\n"
                "[S99]\n\n## 参考来源\n- [来源](https://example.com/a)"
            )
            with self.subTest(heading=heading):
                citations = analyze_run(exported_run(), report)["citation_check"]
                self.assertEqual(citations["inline_link_count"], 0)
                self.assertEqual(citations["unknown_source_count"], 0)
                self.assertFalse(citations["has_inline_citations"])

    def test_citations_only_prove_mapping_not_factual_support(self):
        report = "完全可能是错误结论 [来源](https://example.com/a)。"
        result = analyze_run(exported_run(), report)
        self.assertEqual(result["citation_check"]["known_inline_link_rate"], 1.0)
        self.assertNotIn("factual_accuracy", result)
        self.assertNotIn("citation_support_rate", result["citation_check"])

    def test_unknown_tokens_unmapped_links_and_parenthesized_urls(self):
        report = "[正常](https://example.com/a) [括号](https://example.com/b_(topic)) [站外](https://other.test) [S99] [S1] [unknown source]"
        result = analyze_run(exported_run(), report)["citation_check"]
        self.assertEqual(result["inline_link_count"], 3)
        self.assertEqual(result["known_inline_link_count"], 2)
        self.assertAlmostEqual(result["known_inline_link_rate"], 2 / 3)
        self.assertEqual(result["unknown_source_count"], 3)
        self.assertEqual(result["unmapped_link_count"], 1)

    def test_code_examples_and_images_are_not_citations(self):
        report = "```md\n[来源](https://example.com/a) [S99]\n```\n`[S99]` ![图片](https://example.com/a)"
        result = analyze_run(exported_run(), report)["citation_check"]
        self.assertEqual(result["inline_link_count"], 0)
        self.assertEqual(result["unknown_source_count"], 0)

    def test_reference_heading_inside_code_does_not_truncate_real_body(self):
        report = "```md\n## 参考来源\n```\n真实正文 [来源](https://example.com/a)"
        result = analyze_run(exported_run(), report)["citation_check"]
        self.assertEqual(result["known_inline_link_count"], 1)

    def test_both_fence_types_hide_reference_headings_and_allow_longer_closers(self):
        for fence in ("```", "~~~"):
            for heading in ("## References", "## 参考来源"):
                report = (
                    f"   {fence}markdown\n{heading}\n[示例](https://example.com/a) [S99]\n"
                    f"  {fence}{fence[0]}\n真实正文 [来源](https://example.com/a)"
                )
                with self.subTest(fence=fence, heading=heading):
                    citations = analyze_run(exported_run(), report)["citation_check"]
                    self.assertEqual(citations["inline_link_count"], 1)
                    self.assertEqual(citations["known_inline_link_count"], 1)
                    self.assertEqual(citations["unknown_source_count"], 0)

    def test_unclosed_fences_ignore_remaining_content(self):
        for fence in ("```", "~~~"):
            report = (
                "真实正文 [来源](https://example.com/a)\n"
                f"{fence}\n## References\n## 参考来源\n[示例](https://example.com/a) [S99]"
            )
            with self.subTest(fence=fence):
                citations = analyze_run(exported_run(), report)["citation_check"]
                self.assertEqual(citations["inline_link_count"], 1)
                self.assertEqual(citations["unknown_source_count"], 0)

    def test_shorter_or_other_character_fences_do_not_close_blocks(self):
        for opening, invalid_close in (("````", "```"), ("~~~~", "~~~"), ("```", "~~~"), ("~~~", "```")):
            report = (
                f"{opening}\n{invalid_close}\n[示例](https://example.com/a) [S99]\n"
                f"{opening}\n真实正文 [来源](https://example.com/a)"
            )
            with self.subTest(opening=opening, invalid_close=invalid_close):
                citations = analyze_run(exported_run(), report)["citation_check"]
                self.assertEqual(citations["inline_link_count"], 1)
                self.assertEqual(citations["unknown_source_count"], 0)

    def test_missing_historical_fields_get_safe_defaults(self):
        result = analyze_run({"schema_version": 1, "state": {}, "metrics": {}}, "")
        self.assertEqual(result["status"], "planning")
        self.assertEqual(result["search_count"], 0)
        self.assertFalse(result["report_generated"])
        self.assertIsNone(result["evaluator_coverage"])
        self.assertIsNone(result["stop_reason"])

    def test_unknown_or_empty_subquestions_have_no_coverage(self):
        for plan in (None, {}, {"sub_questions": []}):
            payload = exported_run()
            payload["state"]["plan"] = plan
            with self.subTest(plan=plan):
                self.assertIsNone(analyze_run(payload, "")["evaluator_coverage"])
        payload = exported_run()
        payload["state"]["last_evaluation"] = None
        self.assertIsNone(analyze_run(payload, "")["evaluator_coverage"])
        payload["state"]["last_evaluation"] = {}
        self.assertIsNone(analyze_run(payload, "")["evaluator_coverage"])

    def test_legacy_count_fallbacks_and_nested_errors(self):
        payload = exported_run()
        payload["metrics"] = {}
        payload["state"]["step_count"] = 4
        payload["state"]["elapsed_seconds"] = 7.0
        payload["state"]["evidence"] = [{"url": "https://example.com"}, {"url": "https://example.com/"}]
        del payload["state"]["trace"][2]["details"]["error"]
        result = analyze_run(payload, "")
        self.assertEqual(result["source_count"], 1)
        self.assertEqual(result["search_count"], 4)
        self.assertEqual(result["evaluation_count"], 2)
        self.assertEqual(result["evaluation_error_count"], 1)

    def test_invalid_numeric_metrics_raise_value_error(self):
        for field in ("search_count", "evaluation_count", "source_count", "elapsed_seconds"):
            for invalid in (-1, float("nan"), float("inf"), True, "3", None):
                payload = exported_run()
                payload["metrics"][field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    analyze_run(payload, "")
        payload = exported_run()
        payload["metrics"]["search_count"] = 1.5
        with self.assertRaises(ValueError):
            analyze_run(payload, "")

    def test_invalid_structures_statuses_and_versions_raise_value_error(self):
        changes = [
            ("schema_version", 2), ("schema_version", True), ("state", []), ("metrics", None),
        ]
        for field, invalid in changes:
            payload = exported_run()
            payload[field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_run(payload, "")
        state_changes = [
            ("status", "success"), ("stop_reason", "success"), ("report_generated", "yes"),
            ("report_source_urls", []), ("report_source_urls", {"S1": None}),
            ("trace", [None]), ("trace", [{"details": None}]),
            ("plan", {"sub_questions": "q1"}),
            ("last_evaluation", {"answered_sub_questions": [False]}),
        ]
        for field, invalid in state_changes:
            payload = exported_run()
            payload["state"][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_run(payload, "")
        with self.assertRaises(ValueError):
            analyze_run(None, "")
        with self.assertRaises(ValueError):
            analyze_run(exported_run(), None)

    def test_input_is_not_mutated(self):
        payload = exported_run()
        original = copy.deepcopy(payload)
        analyze_run(payload, "")
        self.assertEqual(payload, original)


class SummarizeRunsTests(unittest.TestCase):
    def test_failures_and_missing_metrics_stay_in_denominator(self):
        successful = analyze_run(exported_run(), "结论")
        partial = copy.deepcopy(successful)
        partial.update(status="partial", stop_reason="search_budget_exhausted", search_count=7, elapsed_seconds=9.5, report_generated=False)
        summary = summarize_runs([
            {"case_id": "a", "metrics": successful, "error": None},
            {"case_id": "b", "metrics": partial},
            {"case_id": "c", "metrics": None, "error": "network failed", "error_type": "RuntimeError"},
            {"case_id": "d", "metrics": None},
        ])
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["measured_case_count"], 2)
        self.assertEqual(summary["completed_count"], 1)
        self.assertEqual(summary["program_completion_rate"], 0.25)
        self.assertEqual(summary["report_generation_rate"], 0.25)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["mean_search_count"], 5)
        self.assertEqual(summary["mean_elapsed_seconds"], 7)
        self.assertEqual(summary["total_search_errors"], 2)
        self.assertEqual(summary["total_evaluation_errors"], 2)
        self.assertEqual(summary["stop_reasons"], {"completed": 1, "search_budget_exhausted": 1})

    def test_zero_records_have_no_rates_or_means(self):
        summary = summarize_runs([])
        self.assertEqual(summary["case_count"], 0)
        for field in ("program_completion_rate", "report_generation_rate", "mean_search_count", "mean_elapsed_seconds"):
            self.assertIsNone(summary[field])
        self.assertEqual(summary["stop_reasons"], {})

    def test_all_failed_cases_have_zero_rates_but_unknown_means(self):
        summary = summarize_runs([{"metrics": None, "error_type": "ValueError"}])
        self.assertEqual(summary["program_completion_rate"], 0)
        self.assertEqual(summary["report_generation_rate"], 0)
        self.assertIsNone(summary["mean_search_count"])
        self.assertEqual(summary["error_count"], 1)

    def test_invalid_records_raise_value_error(self):
        for invalid in (None, {}, [None], [{"metrics": []}], [{"metrics": {}}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summarize_runs(invalid)
        for field, invalid in (("search_count", True), ("elapsed_seconds", float("nan")), ("status", "bad"), ("stop_reason", "bad"), ("report_generated", 1)):
            metrics = analyze_run(exported_run(), "")
            metrics[field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_runs([{"metrics": metrics}])


if __name__ == "__main__":
    unittest.main()

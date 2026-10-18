import copy
import unittest

from app.review import review_template, summarize_reviews


def batch():
    return {
        "batch_id": "batch",
        "results": [
            {"case_id": "a", "run_id": "run-a", "error": None,
             "metrics": {"report_generated": True}},
            {"case_id": "b", "run_id": "run-b", "error": None,
             "metrics": {"report_generated": True}},
        ],
    }


def filled_reviews():
    payload = review_template(batch())
    payload["reviews"][0].update(passed=True, coverage=2, grounding=1, source_quality=2)
    payload["reviews"][1].update(passed=False, coverage=0, grounding=1, source_quality=0)
    return payload


class HumanReviewTests(unittest.TestCase):
    def test_unreviewed_template_has_no_quality_score(self):
        summary = summarize_reviews(batch(), review_template(batch()))
        self.assertEqual(summary["reviewed_count"], 0)
        self.assertEqual(summary["pending_count"], 2)
        self.assertIsNone(summary["human_pass_rate"])
        self.assertIsNone(summary["mean_coverage"])

    def test_partial_rows_are_pending_not_quality_evidence(self):
        payload = review_template(batch())
        payload["reviews"][0].update(passed=True, coverage=2, notes="部分填写")
        summary = summarize_reviews(batch(), payload)
        self.assertEqual(summary["reviewed_count"], 0)
        self.assertEqual(summary["human_pass_count"], 0)

    def test_only_fully_reviewed_rows_determine_quality_averages(self):
        payload = filled_reviews()
        summary = summarize_reviews(batch(), payload)
        self.assertEqual(summary["reviewed_count"], 2)
        self.assertEqual(summary["human_pass_rate"], 0.5)
        self.assertEqual(summary["mean_coverage"], 1)
        self.assertEqual(summary["mean_grounding"], 1)

    def test_missing_rows_remain_pending(self):
        payload = filled_reviews()
        payload["reviews"].pop()
        summary = summarize_reviews(batch(), payload)
        self.assertEqual(summary["reviewed_count"], 1)
        self.assertEqual(summary["pending_count"], 1)

    def test_other_batch_or_run_labels_cannot_be_reused(self):
        payload = filled_reviews()
        payload["batch_id"] = "other-batch"
        with self.assertRaises(ValueError):
            summarize_reviews(batch(), payload)
        payload = filled_reviews()
        payload["reviews"][0]["run_id"] = "other-run"
        with self.assertRaises(ValueError):
            summarize_reviews(batch(), payload)

    def test_duplicate_or_unknown_case_ids_are_rejected(self):
        for case_id in ("a", "unknown"):
            payload = filled_reviews()
            payload["reviews"][1]["case_id"] = case_id
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                summarize_reviews(batch(), payload)

    def test_boolean_scores_out_of_range_and_string_labels_are_rejected(self):
        for field, invalid in (
            ("coverage", True), ("grounding", -1), ("source_quality", 3),
            ("coverage", "2"), ("passed", "true"), ("passed", 1), ("notes", []),
        ):
            payload = filled_reviews()
            payload["reviews"][0][field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                summarize_reviews(batch(), payload)

    def test_failed_or_empty_report_cannot_receive_pass_label(self):
        for result_change in ({"error": "failed"}, {"metrics": {"report_generated": False}}):
            data = batch()
            data["results"][0].update(result_change)
            with self.subTest(change=result_change), self.assertRaises(ValueError):
                summarize_reviews(data, filled_reviews())

    def test_inputs_are_not_changed(self):
        data, reviews = batch(), filled_reviews()
        original = copy.deepcopy((data, reviews))
        summarize_reviews(data, reviews)
        self.assertEqual((data, reviews), original)

    def test_invalid_review_structure_is_rejected(self):
        for invalid in (None, [], {}, {"schema_version": True},
                        {"schema_version": 1, "batch_id": "batch", "reviews": [None]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summarize_reviews(batch(), invalid)


if __name__ == "__main__":
    unittest.main()

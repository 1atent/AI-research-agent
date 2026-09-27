import json
import unittest

from app.evidence import (
    MAX_SNIPPETS_PER_SOURCE,
    add_evidence,
    count_unique_sources,
    evidence_excerpt,
    parse_search_response,
    parse_search_results,
    select_evidence,
)
from app.models import (
    Evidence, EvaluationResult, EvaluationStatus, ResearchPlan, ResearchState,
)


class ParseSearchResultsTests(unittest.TestCase):
    def test_valid_results_are_converted_to_evidence(self):
        raw_result = json.dumps(
            {
                "ok": True,
                "results": [
                    {
                        "title": "RAG Introduction",
                        "url": "https://example.com/rag",
                        "content": "RAG combines retrieval and generation.",
                        "score": 0.9,
                    }
                ],
            }
        )

        evidence = parse_search_results(raw_result, "什么是 RAG？")

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].title, "RAG Introduction")
        self.assertEqual(evidence[0].supports, ["什么是 RAG？"])

    def test_invalid_or_failed_result_returns_empty_list(self):
        self.assertEqual(parse_search_results("非法 JSON", "question"), [])
        self.assertEqual(
            parse_search_results('{"ok": false, "error": "timeout"}', "question"),
            [],
        )

    def test_items_without_title_or_url_are_ignored(self):
        raw_result = json.dumps(
            {
                "ok": True,
                "results": [
                    {"title": "", "url": "https://example.com", "content": "A"},
                    {"title": "Title", "url": "", "content": "B"},
                ],
            }
        )

        self.assertEqual(parse_search_results(raw_result, "question"), [])

    def test_empty_content_and_invalid_urls_are_not_evidence(self):
        result = json.dumps({"ok": True, "results": [
            {"title": "Empty", "url": "https://example.com", "content": " "},
            {"title": "Invalid", "url": "file:///private", "content": "text"},
        ]})
        self.assertEqual(parse_search_results(result, "question"), [])

    def test_successful_empty_search_is_distinct_from_failure(self):
        empty = parse_search_response('{"ok": true, "results": []}', "q1")
        failure = parse_search_response('{"ok": false, "error": "timeout"}', "q1")
        self.assertTrue(empty.ok)
        self.assertIsNone(empty.error)
        self.assertFalse(failure.ok)
        self.assertEqual(failure.error, "timeout")

    def test_nonfinite_scores_are_discarded(self):
        result = json.dumps({"ok": True, "results": [{
            "title": "Title", "url": "https://example.com", "content": "text",
            "score": "NaN",
        }]})
        self.assertIsNone(parse_search_results(result, "q1")[0].score)


class EvidenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.state = ResearchState(
            plan=ResearchPlan(question="question", sub_questions=["q1", "q2"])
        )

    def test_duplicate_url_is_merged(self):
        first = Evidence(
            title="Source",
            url="https://EXAMPLE.com/article/",
            content="short",
            score=0.5,
            supports=["q1"],
        )
        duplicate = Evidence(
            title="Source updated",
            url="https://example.com/article#section",
            content="a longer description",
            score=0.8,
            supports=["q2"],
        )

        self.assertEqual(add_evidence(self.state, [first]), 1)
        self.assertEqual(add_evidence(self.state, [duplicate]), 0)

        self.assertEqual(count_unique_sources(self.state), 1)
        self.assertEqual(self.state.evidence[0].supports, ["q1", "q2"])
        self.assertEqual(self.state.evidence[0].snippets, ["short", "a longer description"])
        self.assertIn("short", self.state.evidence[0].content)
        self.assertIn("a longer description", self.state.evidence[0].content)
        self.assertEqual(self.state.evidence[0].score, 0.8)

    def test_repeated_snippet_is_not_duplicated_and_recent_step_is_preserved(self):
        first = Evidence("Source", "https://example.com", "original", last_seen_step=1)
        update = Evidence("Source", "https://example.com", "original", last_seen_step=4)
        add_evidence(self.state, [first, update, update])
        self.assertEqual(self.state.evidence[0].snippets, ["original"])
        self.assertEqual(self.state.evidence[0].last_seen_step, 4)

    def test_fragment_budget_keeps_original_and_newest_observations(self):
        for index in range(12):
            add_evidence(self.state, [Evidence(
                "Source", "https://example.com", f"fragment-{index}",
            )])
        snippets = self.state.evidence[0].snippets
        self.assertEqual(len(snippets), MAX_SNIPPETS_PER_SOURCE)
        self.assertEqual(snippets[0], "fragment-0")
        self.assertEqual(snippets[-1], "fragment-11")


class EvidenceSelectionTests(unittest.TestCase):
    def test_recent_missing_topic_is_visible_after_twenty_old_sources(self):
        state = ResearchState(ResearchPlan("question", ["q1", "q2"]))
        state.evidence = [
            Evidence(f"Old {i}", f"https://example.com/{i}", "old", 0.99, ["q1"],
                     last_seen_step=1)
            for i in range(25)
        ]
        recent = Evidence("New", "https://example.com/new", "important", 0.1, ["q2"],
                          last_seen_step=6)
        state.evidence.append(recent)
        state.last_evaluation = EvaluationResult(
            EvaluationStatus.CONTINUE, missing_sub_questions=["q2"]
        )
        selected = select_evidence(state, 20)
        self.assertIn(recent, selected)
        self.assertLessEqual(len(selected), 20)
        self.assertTrue(any("q1" in item.supports for item in selected))

    def test_recent_duplicate_source_excerpt_keeps_new_fragment(self):
        item = Evidence("Source", "https://example.com", "", snippets=[
            "ORIGINAL " + "a" * 2000,
            "NEW_INFORMATION " + "b" * 2000,
        ])
        excerpt = evidence_excerpt(item, 1200)
        self.assertIn("ORIGINAL", excerpt)
        self.assertIn("NEW_INFORMATION", excerpt)
        self.assertLessEqual(len(excerpt), 1200)


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from app.evidence import add_evidence, count_unique_sources, parse_search_results
from app.models import Evidence, ResearchPlan, ResearchState


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
        self.assertEqual(self.state.evidence[0].content, "a longer description")
        self.assertEqual(self.state.evidence[0].score, 0.8)


if __name__ == "__main__":
    unittest.main()

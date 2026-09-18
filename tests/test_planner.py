import unittest
from unittest.mock import Mock

from app.planner import create_research_plan, parse_research_plan


class ParseResearchPlanTests(unittest.TestCase):
    def test_parse_valid_plan(self):
        content = """
        {
          "sub_questions": ["什么是 RAG？", "什么是 LLM Wiki？"],
          "minimum_sources": 4
        }
        """

        plan = parse_research_plan(content, "RAG 和 LLM Wiki 有什么区别？")

        self.assertEqual(len(plan.sub_questions), 2)
        self.assertEqual(plan.minimum_sources, 4)

    def test_remove_empty_and_duplicate_questions(self):
        content = """
        {
          "sub_questions": ["什么是 RAG？", "", "  什么是 RAG？  ", 123],
          "minimum_sources": 3
        }
        """

        plan = parse_research_plan(content, "什么是 RAG？")

        self.assertEqual(plan.sub_questions, ["什么是 RAG？"])

    def test_invalid_json_uses_fallback(self):
        plan = parse_research_plan("这不是 JSON", "什么是 RAG？")

        self.assertEqual(plan.sub_questions, ["什么是 RAG？"])
        self.assertEqual(plan.minimum_sources, 3)

    def test_source_count_is_limited(self):
        content = '{"sub_questions": ["什么是 RAG？"], "minimum_sources": 999}'

        plan = parse_research_plan(content, "什么是 RAG？")

        self.assertEqual(plan.minimum_sources, 10)

    def test_markdown_json_fence_is_supported(self):
        content = '```json\n{"sub_questions": ["什么是 RAG？"]}\n```'

        plan = parse_research_plan(content, "什么是 RAG？")

        self.assertEqual(plan.sub_questions, ["什么是 RAG？"])


class CreateResearchPlanTests(unittest.TestCase):
    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            create_research_plan("   ", Mock(), "test-model")

    def test_client_response_is_converted_to_plan(self):
        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=(
                            '{"sub_questions": ["什么是 RAG？", '
                            '"什么是 LLM Wiki？"], "minimum_sources": 3}'
                        )
                    )
                )
            ]
        )

        plan = create_research_plan("比较 RAG 和 LLM Wiki", client, "test-model")

        self.assertEqual(len(plan.sub_questions), 2)
        client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()

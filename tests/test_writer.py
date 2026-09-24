import unittest
from unittest.mock import Mock

from app.models import Evidence, ResearchPlan, ResearchState, ResearchStatus
from app.writer import write_report


class WriteReportTests(unittest.TestCase):
    def test_no_evidence_returns_message_without_calling_model(self):
        state = ResearchState(
            plan=ResearchPlan(question="question", sub_questions=["q1"])
        )
        client = Mock()

        report = write_report(state, client, "test-model")

        self.assertIn("未获得可用证据", report)
        client.chat.completions.create.assert_not_called()

    def test_source_tokens_are_replaced_with_trusted_links(self):
        state = ResearchState(
            plan=ResearchPlan(question="question", sub_questions=["q1"]),
            evidence=[
                Evidence(
                    title="Trusted Source",
                    url="https://example.com/trusted",
                    content="evidence",
                    score=0.9,
                    supports=["q1"],
                )
            ],
            status=ResearchStatus.COMPLETED,
        )
        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=(
                            "关键结论 [S1]。"
                            "不可信链接：[Fake](https://evil.example/path)"
                        )
                    )
                )
            ]
        )

        report = write_report(state, client, "test-model")

        self.assertIn("[Trusted Source](https://example.com/trusted)", report)
        self.assertNotIn("https://evil.example", report)
        self.assertIn("## 参考来源", report)


if __name__ == "__main__":
    unittest.main()

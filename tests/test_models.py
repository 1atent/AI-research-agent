import unittest

from app.models import (
    Evidence,
    ResearchPlan,
    ResearchState,
    ResearchStatus,
)


class ResearchStateTests(unittest.TestCase):

    def test_create_research_state(self):
        plan = ResearchPlan(
            question="RAG 和 LLM Wiki 有什么区别？",
            sub_questions=[
                "什么是 RAG？",
                "什么是 LLM Wiki？",
            ],
            minimum_sources=2,
        )

        state = ResearchState(plan=plan)

        self.assertEqual(state.step_count, 0)
        self.assertEqual(state.status, ResearchStatus.PLANNING)
        self.assertEqual(len(state.evidence), 0)
        self.assertEqual(len(state.searched_queries), 0)
        self.assertEqual(len(state.plan.sub_questions), 2)

    def test_add_evidence(self):
        plan = ResearchPlan(
            question="什么是 RAG？",
            sub_questions=["RAG 的核心原理是什么？"],
        )

        state = ResearchState(plan=plan)

        evidence = Evidence(
            title="RAG Introduction",
            url="https://example.com/rag",
            content="RAG combines retrieval and generation.",
            score=0.9,
            supports=["RAG 的核心原理是什么？"],
        )

        state.evidence.append(evidence)
        state.searched_queries.add("RAG introduction")
        state.step_count += 1
        state.status = ResearchStatus.RESEARCHING

        self.assertEqual(len(state.evidence), 1)
        self.assertEqual(state.evidence[0].title, "RAG Introduction")
        self.assertIn("RAG introduction", state.searched_queries)
        self.assertEqual(state.step_count, 1)
        self.assertEqual(
            state.status,
            ResearchStatus.RESEARCHING,
        )


if __name__ == "__main__":
    unittest.main()
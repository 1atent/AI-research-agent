import json
import os
import unittest
from unittest.mock import Mock, patch

from app.tools import search_web


class SearchWebTests(unittest.TestCase):
    def test_missing_api_key_returns_structured_error(self):
        with patch.dict(os.environ, {}, clear=True):
            result = json.loads(search_web("AI Agent"))

        self.assertFalse(result["ok"])
        self.assertIn("TAVILY_API_KEY", result["error"])

    @patch("app.tools.requests.post")
    def test_success_keeps_only_required_fields(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "Evidence",
                    "score": 0.9,
                    "raw_content": "must not be returned",
                }
            ]
        }
        mock_post.return_value = response

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            result = json.loads(search_web("AI Agent", max_results=99))

        self.assertTrue(result["ok"])
        self.assertEqual(result["result_count"], 1)
        self.assertNotIn("raw_content", result["results"][0])
        self.assertEqual(mock_post.call_args.kwargs["json"]["max_results"], 8)


if __name__ == "__main__":
    unittest.main()

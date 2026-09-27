import json
import os
import unittest
from unittest.mock import Mock, patch

import requests

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

    @patch("app.tools.requests.post")
    def test_invalid_response_schema_returns_structured_error(self, mock_post):
        response = Mock()
        mock_post.return_value = response
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            for payload in [[], {"results": None}, {}]:
                with self.subTest(payload=payload):
                    response.json.return_value = payload
                    result = json.loads(search_web("query"))
                    self.assertFalse(result["ok"])
                    self.assertIn("结构", result["error"])

    @patch("app.tools.requests.post")
    def test_non_object_result_items_are_skipped(self, mock_post):
        response = Mock()
        response.json.return_value = {
            "results": [None, "invalid", {"title": "Example", "url": "https://example.com", "content": "Evidence"}]
        }
        mock_post.return_value = response
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            result = json.loads(search_web("query"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["result_count"], 1)

    @patch("app.tools.requests.post", side_effect=requests.Timeout)
    def test_timeout_returns_structured_error(self, mock_post):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            result = json.loads(search_web("query"))
        self.assertFalse(result["ok"])
        self.assertIn("超时", result["error"])


if __name__ == "__main__":
    unittest.main()

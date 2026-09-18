import os
import unittest
from unittest.mock import patch

from app.client import create_client


class CreateClientTests(unittest.TestCase):
    def test_missing_api_key_is_rejected(self):
        with patch.dict(os.environ, {"MODEL": "test-model"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "API_KEY"):
                create_client()

    def test_missing_model_is_rejected(self):
        with patch.dict(os.environ, {"API_KEY": "test-key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MODEL"):
                create_client()

    @patch("app.client.OpenAI")
    def test_configuration_creates_client(self, mock_openai):
        with patch.dict(
            os.environ,
            {
                "API_KEY": "test-key",
                "BASE_URL": "https://example.com/v1",
                "MODEL": "test-model",
            },
            clear=True,
        ):
            client, model = create_client()

        self.assertIs(client, mock_openai.return_value)
        self.assertEqual(model, "test-model")
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.com/v1",
        )


if __name__ == "__main__":
    unittest.main()

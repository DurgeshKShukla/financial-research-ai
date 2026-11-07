import json
import os
import unittest
from unittest.mock import patch
import ai_backend


class CloudInferenceTests(unittest.TestCase):
    def test_model_default(self):
        with patch.dict(os.environ, {"GROQ_MODEL": ""}):
            self.assertEqual(ai_backend.groq_model(), "openai/gpt-oss-20b")

    def test_requires_key(self):
        with patch("ai_backend.get_groq_key", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                ai_backend.groq_chat([{"role": "user", "content": "Hello"}])

    def test_request_parses_response_without_network(self):
        from io import BytesIO
        class MockResponse:
            def __enter__(self):
                return BytesIO(json.dumps({"choices":[{"message":{"content":"Hello!"}}]}).encode())
            def __exit__(self, *args):
                return False
        with patch("ai_backend.get_groq_key", return_value="fake-key"), patch("ai_backend.urlopen", return_value=MockResponse()) as request:
            self.assertEqual(ai_backend.groq_chat([{"role": "user", "content": "Hello"}]), "Hello!")
            data = json.loads(request.call_args.args[0].data)
            self.assertEqual(data["model"], "openai/gpt-oss-20b")

if __name__ == "__main__":
    unittest.main()

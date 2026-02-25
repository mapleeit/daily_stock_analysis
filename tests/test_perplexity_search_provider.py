# -*- coding: utf-8 -*-
"""Unit tests for Perplexity Sonar Pro Search provider integration."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock optional newspaper dependency before importing search_service
if "newspaper" not in sys.modules:
    mock_np = MagicMock()
    mock_np.Article = MagicMock()
    mock_np.Config = MagicMock()
    sys.modules["newspaper"] = mock_np

from src.config import Config
from src.search_service import SearchService


def _mock_http_response(status_code: int, payload: dict, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": "application/json"}
    response.json.return_value = payload
    response.text = text or str(payload)
    return response


class PerplexitySearchProviderTestCase(unittest.TestCase):
    """SearchService + Perplexity provider behavior tests."""

    def _build_service(self) -> SearchService:
        return SearchService(
            perplexity_keys=["test_key_1", "test_key_2"],
            perplexity_base_url="https://openrouter.ai/api/v1",
            perplexity_model="perplexity/sonar-pro-search",
            news_max_age_days=3,
        )

    @patch("src.search_service.requests.post")
    def test_perplexity_provider_parses_structured_results(self, mock_post: MagicMock) -> None:
        payload = {
            "choices": [{"message": {"content": "Found recent updates with sources."}}],
            "search_results": [
                {
                    "title": "Apple shares rise on earnings beat",
                    "url": "https://example.com/apple-earnings",
                    "snippet": "Apple beat expectations in the latest quarter.",
                    "date": "2026-02-25",
                }
            ],
        }
        mock_post.return_value = _mock_http_response(200, payload)

        service = self._build_service()
        response = service.search_stock_news("AAPL", "Apple", max_results=3)

        self.assertTrue(response.success)
        self.assertEqual(response.provider, "Perplexity")
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.results[0].title, "Apple shares rise on earnings beat")
        self.assertEqual(response.results[0].url, "https://example.com/apple-earnings")

    @patch("src.search_service.requests.post")
    def test_perplexity_provider_fallback_to_citations(self, mock_post: MagicMock) -> None:
        payload = {
            "choices": [{"message": {"content": "Summary with source links."}}],
            "citations": [
                "https://news.example.com/market-1",
                "https://news.example.com/market-2",
            ],
        }
        mock_post.return_value = _mock_http_response(200, payload)

        service = self._build_service()
        response = service.search_stock_news("TSLA", "Tesla", max_results=5)

        self.assertTrue(response.success)
        self.assertEqual(response.provider, "Perplexity")
        self.assertGreaterEqual(len(response.results), 2)
        self.assertEqual(response.results[0].url, "https://news.example.com/market-1")
        self.assertIn("Summary with source links", response.results[0].snippet)

    @patch("src.search_service.requests.post")
    def test_perplexity_provider_http_error(self, mock_post: MagicMock) -> None:
        payload = {"error": {"message": "Invalid API key"}}
        mock_post.return_value = _mock_http_response(401, payload)

        service = self._build_service()
        provider = service._providers[0]
        response = provider.search("NVIDIA latest news", max_results=3)

        self.assertFalse(response.success)
        self.assertIn("API key invalid", response.error_message or "")

    def test_perplexity_only_service_is_available(self) -> None:
        service = self._build_service()
        self.assertTrue(service.is_available)


class PerplexityConfigLoadTestCase(unittest.TestCase):
    """Config parsing tests for Perplexity environment variables."""

    @patch.dict(
        os.environ,
        {
            "PERPLEXITY_API_KEYS": "key_a,key_b",
            "PERPLEXITY_BASE_URL": "https://openrouter.ai/api/v1",
            "PERPLEXITY_MODEL": "perplexity/sonar-pro-search",
        },
        clear=True,
    )
    @patch("src.config.load_dotenv")
    def test_load_perplexity_config_from_env(self, _mock_dotenv: MagicMock) -> None:
        Config._instance = None
        config = Config._load_from_env()

        self.assertEqual(config.perplexity_api_keys, ["key_a", "key_b"])
        self.assertEqual(config.perplexity_base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.perplexity_model, "perplexity/sonar-pro-search")


if __name__ == "__main__":
    unittest.main()

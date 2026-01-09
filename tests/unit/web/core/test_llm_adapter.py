"""Tests for WebLLMAdapter."""

from unittest.mock import MagicMock, patch

import pytest

from web.core.llm_adapter import WebLLMAdapter


class TestWebLLMAdapter:
    """Tests for WebLLMAdapter class."""

    def _create_adapter(self) -> WebLLMAdapter:
        """Create adapter with mocked AsyncOpenAI client."""
        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8000/v1/"
        mock_client.api_key = "test-key"
        return WebLLMAdapter(mock_client, "gpt-4")

    def test_init(self):
        """Adapter initializes correctly."""
        adapter = self._create_adapter()

        assert adapter.model == "gpt-4"
        assert adapter.base_url == "http://localhost:8000/v1/"
        assert adapter.api_key == "test-key"

    def test_embed_documents_returns_empty(self):
        """embed_documents returns empty list (not implemented)."""
        adapter = self._create_adapter()

        result = adapter.embed_documents(["text1", "text2"])

        assert result == []

    def test_embed_query_returns_empty(self):
        """embed_query returns empty list (not implemented)."""
        adapter = self._create_adapter()

        result = adapter.embed_query("query")

        assert result == []

    @patch("httpx.post")
    def test_chat_success(self, mock_post):
        """chat returns LLM response content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello, world!"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        adapter = self._create_adapter()
        messages = [{"role": "user", "content": "Hi"}]

        result = adapter.chat(messages)

        assert result == "Hello, world!"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "gpt-4"
        assert call_args[1]["json"]["messages"] == messages

    @patch("httpx.post")
    def test_chat_with_temperature(self, mock_post):
        """chat uses provided temperature."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        adapter = self._create_adapter()

        adapter.chat([{"role": "user", "content": "Hi"}], temperature=0.7)

        call_args = mock_post.call_args
        assert call_args[1]["json"]["temperature"] == 0.7

    @patch("httpx.post")
    def test_chat_raises_on_error(self, mock_post):
        """chat re-raises exceptions."""
        mock_post.side_effect = Exception("Connection error")

        adapter = self._create_adapter()

        with pytest.raises(Exception, match="Connection error"):
            adapter.chat([{"role": "user", "content": "Hi"}])

    @patch("httpx.Client")
    def test_stream_chat_yields_chunks(self, mock_client_class):
        """stream_chat yields content chunks."""
        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "data: [DONE]",
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value.__enter__ = MagicMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        adapter = self._create_adapter()
        messages = [{"role": "user", "content": "Hi"}]

        chunks = list(adapter.stream_chat(messages))

        assert chunks == ["Hello", " world"]

    @patch("httpx.Client")
    def test_stream_chat_handles_empty_content(self, mock_client_class):
        """stream_chat skips chunks without content."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            'data: {"choices": [{"delta": {}}]}',  # No content
            'data: {"choices": [{"delta": {"content": "text"}}]}',
            "data: [DONE]",
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value.__enter__ = MagicMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        adapter = self._create_adapter()

        chunks = list(adapter.stream_chat([{"role": "user", "content": "Hi"}]))

        assert chunks == ["text"]

    @patch("httpx.Client")
    def test_stream_chat_raises_on_error(self, mock_client_class):
        """stream_chat re-raises exceptions."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.side_effect = Exception("Network error")
        mock_client_class.return_value = mock_client

        adapter = self._create_adapter()

        with pytest.raises(Exception, match="Network error"):
            list(adapter.stream_chat([{"role": "user", "content": "Hi"}]))

    @patch("httpx.Client")
    def test_stream_chat_handles_json_decode_error(self, mock_client_class):
        """stream_chat ignores malformed JSON."""
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            "data: invalid json",
            'data: {"choices": [{"delta": {"content": "valid"}}]}',
            "data: [DONE]",
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value.__enter__ = MagicMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        adapter = self._create_adapter()

        # Should not raise, should skip invalid lines
        chunks = list(adapter.stream_chat([{"role": "user", "content": "Hi"}]))

        assert chunks == ["valid"]

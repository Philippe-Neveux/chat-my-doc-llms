from unittest.mock import Mock, patch

import pytest

from chat_my_doc_llms.chats import chat_with_gemini, chat_with_gemini_stream


class TestChatWithGemini:
    @patch('chat_my_doc_llms.chats.client')
    def test_chat_with_gemini_success(self, mock_client):
        mock_response = Mock()
        mock_response.text = "Hello! How can I help you?"
        mock_client.models.generate_content.return_value = mock_response
        
        result = chat_with_gemini("Hello", "gemini-2.0-flash-lite")
        
        assert result == "Hello! How can I help you?"
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash-lite",
            contents="Hello"
        )

    @patch('chat_my_doc_llms.chats.client')
    def test_chat_with_gemini_no_text_response(self, mock_client):
        mock_response = Mock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response
        
        with pytest.raises(ValueError, match="Error from the Gemini API"):
            chat_with_gemini("Hello", "gemini-2.0-flash-lite")

    @patch('chat_my_doc_llms.chats.client')
    def test_chat_with_gemini_with_different_model(self, mock_client):
        mock_response = Mock()
        mock_response.text = "Response from Pro model"
        mock_client.models.generate_content.return_value = mock_response
        
        result = chat_with_gemini("Hello", "gemini-1.5-pro")
        
        assert result == "Response from Pro model"
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-1.5-pro",
            contents="Hello"
        )


class TestChatWithGeminiStream:
    @patch('chat_my_doc_llms.chats.client')
    def test_chat_with_gemini_stream_success(self, mock_client):
        mock_chunk1 = Mock()
        mock_chunk1.text = "Hello "
        mock_chunk2 = Mock()
        mock_chunk2.text = "world!"
        mock_chunk3 = Mock()
        mock_chunk3.text = None
        
        mock_client.models.generate_content_stream.return_value = [
            mock_chunk1, mock_chunk2, mock_chunk3
        ]
        
        result = list(chat_with_gemini_stream("Hello", "gemini-2.0-flash"))
        
        assert result == ["Hello ", "world!"]
        mock_client.models.generate_content_stream.assert_called_once_with(
            model="gemini-2.0-flash",
            contents="Hello"
        )

    @patch('chat_my_doc_llms.chats.client')
    def test_chat_with_gemini_stream_empty_response(self, mock_client):
        mock_client.models.generate_content_stream.return_value = []
        
        result = list(chat_with_gemini_stream("Hello", "gemini-2.0-flash"))
        
        assert result == []

    @patch('chat_my_doc_llms.chats.client')
    def test_chat_with_gemini_stream_with_different_model(self, mock_client):
        mock_chunk = Mock()
        mock_chunk.text = "Streaming response"
        mock_client.models.generate_content_stream.return_value = [mock_chunk]
        
        result = list(chat_with_gemini_stream("Hello", "gemini-1.5-pro"))
        
        assert result == ["Streaming response"]
        mock_client.models.generate_content_stream.assert_called_once_with(
            model="gemini-1.5-pro",
            contents="Hello"
        )
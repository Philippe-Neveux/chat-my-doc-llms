from unittest.mock import Mock, patch, AsyncMock
import pytest
import httpx

from chat_my_doc_llms.chats import chat_with_gemini, chat_with_gemini_stream, chat_with_mistral, chat_with_mistral_stream


class TestChatWithGemini:
    @patch('chat_my_doc_llms.chats.client')
    @pytest.mark.asyncio
    async def test_chat_with_gemini_success(self, mock_client):
        mock_response = Mock()
        mock_response.text = "Hello! How can I help you?"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        result = await chat_with_gemini("Hello", "gemini-2.0-flash-lite")
        
        assert result == "Hello! How can I help you?"
        mock_client.aio.models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash-lite",
            contents="Hello"
        )

    @patch('chat_my_doc_llms.chats.client')
    @pytest.mark.asyncio
    async def test_chat_with_gemini_no_text_response(self, mock_client):
        mock_response = Mock()
        mock_response.text = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        with pytest.raises(ValueError, match="Error from the Gemini API"):
            await chat_with_gemini("Hello", "gemini-2.0-flash-lite")

    @patch('chat_my_doc_llms.chats.client')
    @pytest.mark.asyncio
    async def test_chat_with_gemini_with_different_model(self, mock_client):
        mock_response = Mock()
        mock_response.text = "Response from Pro model"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        result = await chat_with_gemini("Hello", "gemini-1.5-pro")
        
        assert result == "Response from Pro model"
        mock_client.aio.models.generate_content.assert_called_once_with(
            model="gemini-1.5-pro",
            contents="Hello"
        )


class TestChatWithGeminiStream:
    @patch('chat_my_doc_llms.chats.client')
    @pytest.mark.asyncio
    async def test_chat_with_gemini_stream_success(self, mock_client):
        mock_chunk1 = Mock()
        mock_chunk1.text = "Hello "
        mock_chunk2 = Mock()
        mock_chunk2.text = "world!"
        mock_chunk3 = Mock()
        mock_chunk3.text = None
        
        async def mock_stream():
            for chunk in [mock_chunk1, mock_chunk2, mock_chunk3]:
                yield chunk
        
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())
        
        result = []
        async for chunk in chat_with_gemini_stream("Hello", "gemini-2.0-flash"):
            result.append(chunk)
        
        assert result == ["Hello ", "world!"]
        mock_client.aio.models.generate_content_stream.assert_called_once_with(
            model="gemini-2.0-flash",
            contents="Hello"
        )

    @patch('chat_my_doc_llms.chats.client')
    @pytest.mark.asyncio
    async def test_chat_with_gemini_stream_empty_response(self, mock_client):
        async def empty_stream():
            return
            yield  # unreachable
        
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=empty_stream())
        
        result = []
        async for chunk in chat_with_gemini_stream("Hello", "gemini-2.0-flash"):
            result.append(chunk)
        
        assert result == []

    @patch('chat_my_doc_llms.chats.client')
    @pytest.mark.asyncio
    async def test_chat_with_gemini_stream_with_different_model(self, mock_client):
        mock_chunk = Mock()
        mock_chunk.text = "Streaming response"
        
        async def mock_stream():
            yield mock_chunk
        
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())
        
        result = []
        async for chunk in chat_with_gemini_stream("Hello", "gemini-1.5-pro"):
            result.append(chunk)
        
        assert result == ["Streaming response"]
        mock_client.aio.models.generate_content_stream.assert_called_once_with(
            model="gemini-1.5-pro",
            contents="Hello"
        )


class TestChatWithMistral:
    @patch('chat_my_doc_llms.chats.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_chat_with_mistral_success(self, mock_async_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "generated_text": "2+2 equals 4",
            "prompt": "What is 2+2?",
            "model_info": {"status": "loaded"}
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        result = await chat_with_mistral("What is 2+2?")
        
        assert result == "2+2 equals 4"
        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args
        assert "generate" in call_args[0][0]
        assert call_args[1]["json"]["request"]["prompt"] == "What is 2+2?"
    
    @patch('chat_my_doc_llms.chats.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_chat_with_mistral_http_error(self, mock_async_client):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        with pytest.raises(ValueError, match="Error from Mistral API: 400"):
            await chat_with_mistral("Hello")
    
    @patch('chat_my_doc_llms.chats.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_chat_with_mistral_connection_error(self, mock_async_client):
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        with pytest.raises(ValueError, match="Connection Error to Mistral API"):
            await chat_with_mistral("Hello")
    
    @patch('chat_my_doc_llms.chats.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_chat_with_mistral_timeout_error(self, mock_async_client):
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        with pytest.raises(ValueError, match="Timeout Error from Mistral API"):
            await chat_with_mistral("Hello")


class TestChatWithMistralStream:
    @patch('chat_my_doc_llms.chats.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_chat_with_mistral_stream_success(self, mock_async_client):
        mock_response = Mock()
        mock_response.status_code = 200
        
        async def mock_aiter_lines():
            yield 'data: {"token": "Hello", "finished": false}'
            yield 'data: {"token": " world", "finished": false}'
            yield 'data: {"token": "!", "finished": false}'
            yield 'data: {"token": "", "finished": true}'
        
        mock_response.aiter_lines = mock_aiter_lines
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_client_instance = Mock()
        mock_client_instance.stream.return_value = mock_stream_context
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        result = []
        async for chunk in chat_with_mistral_stream("Hello"):
            result.append(chunk)
        
        assert result == ["Hello", " world", "!"]
        mock_client_instance.stream.assert_called_once()
    
    @patch('chat_my_doc_llms.chats.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_chat_with_mistral_stream_error_response(self, mock_async_client):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_client_instance = Mock()
        mock_client_instance.stream.return_value = mock_stream_context
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        with pytest.raises(ValueError, match="Error from Mistral streaming API: 500"):
            async for chunk in chat_with_mistral_stream("Hello"):
                pass
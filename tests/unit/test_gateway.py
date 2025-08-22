from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from chat_my_doc_llms.gateway import ChatRequestGemini, ChatRequestMistral, app

client = TestClient(app)


class TestRoutes:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "This a new message"}


class TestGeminiEndpoint:
    @patch('chat_my_doc_llms.gateway.chat_with_gemini', new_callable=AsyncMock)
    def test_gemini_endpoint_success(self, mock_chat):
        mock_chat.return_value = "Test response from Gemini"
        
        response = client.post("/gemini", json={
            "prompt": "Hello",
            "model_name": "gemini-2.0-flash-lite"
        })
        
        assert response.status_code == 200
        assert response.json() == {"message": "Test response from Gemini"}
        mock_chat.assert_called_once_with("Hello", "gemini-2.0-flash-lite")

    @patch('chat_my_doc_llms.gateway.chat_with_gemini', new_callable=AsyncMock)
    def test_gemini_endpoint_default_model(self, mock_chat):
        mock_chat.return_value = "Default model response"
        
        response = client.post("/gemini", json={
            "prompt": "Hello"
        })
        
        assert response.status_code == 200
        assert response.json() == {"message": "Default model response"}
        mock_chat.assert_called_once_with("Hello", "gemini-2.0-flash-lite")

    def test_gemini_endpoint_invalid_model(self):
        response = client.post("/gemini", json={
            "prompt": "Hello",
            "model_name": "invalid-model"
        })
        
        assert response.status_code == 422

    def test_gemini_endpoint_missing_prompt(self):
        response = client.post("/gemini", json={
            "model_name": "gemini-2.0-flash-lite"
        })
        
        assert response.status_code == 422

    @patch('chat_my_doc_llms.gateway.chat_with_gemini', new_callable=AsyncMock)
    def test_gemini_endpoint_error_handling(self, mock_chat):
        mock_chat.side_effect = ValueError("API Error")
        
        # Since FastAPI doesn't automatically handle exceptions, this will raise
        with pytest.raises(ValueError, match="API Error"):
            client.post("/gemini", json={
                "prompt": "Hello",
                "model_name": "gemini-2.0-flash-lite"
            })
        
        mock_chat.assert_called_once_with("Hello", "gemini-2.0-flash-lite")

    def test_gemini_endpoint_empty_json(self):
        response = client.post("/gemini", json={})
        assert response.status_code == 422

    def test_gemini_endpoint_invalid_json(self):
        response = client.post("/gemini", data="invalid json")
        assert response.status_code == 422


class TestGeminiStreamEndpoint:
    @patch('chat_my_doc_llms.gateway.chat_with_gemini_stream')
    def test_gemini_stream_endpoint_success(self, mock_chat_stream):
        async def mock_stream():
            yield "Hello "
            yield "world!"
        
        mock_chat_stream.return_value = mock_stream()
        
        response = client.post("/gemini-stream", json={
            "prompt": "Hello",
            "model_name": "gemini-2.0-flash"
        })
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        content = response.content.decode()
        assert "Hello world!" in content
        mock_chat_stream.assert_called_once_with("Hello", "gemini-2.0-flash")

    @patch('chat_my_doc_llms.gateway.chat_with_gemini_stream')
    def test_gemini_stream_endpoint_default_model(self, mock_chat_stream):
        async def mock_stream():
            yield "Stream response"
        
        mock_chat_stream.return_value = mock_stream()
        
        response = client.post("/gemini-stream", json={
            "prompt": "Hello"
        })
        
        assert response.status_code == 200
        mock_chat_stream.assert_called_once_with("Hello", "gemini-2.0-flash-lite")

    def test_gemini_stream_endpoint_invalid_model(self):
        response = client.post("/gemini-stream", json={
            "prompt": "Hello",
            "model_name": "invalid-model"
        })
        
        assert response.status_code == 422

    @patch('chat_my_doc_llms.gateway.chat_with_gemini_stream')
    def test_gemini_stream_endpoint_error_handling(self, mock_chat_stream):
        async def mock_error_stream():
            raise ValueError("Stream API Error")
            yield "never reached"
        
        mock_chat_stream.return_value = mock_error_stream()
        
        # Since FastAPI doesn't automatically handle exceptions, this will raise
        with pytest.raises(ValueError, match="Stream API Error"):
            client.post("/gemini-stream", json={
                "prompt": "Hello",
                "model_name": "gemini-2.0-flash"
            })

    def test_gemini_stream_endpoint_missing_prompt(self):
        response = client.post("/gemini-stream", json={
            "model_name": "gemini-2.0-flash"
        })
        
        assert response.status_code == 422


class TestChatRequestGeminiModel:
    def test_chat_request_valid_data(self):
        chat_request = ChatRequestGemini(
            prompt="Hello world",
            model_name="gemini-2.0-flash"
        )
        assert chat_request.prompt == "Hello world"
        assert chat_request.model_name == "gemini-2.0-flash"

    def test_chat_request_default_model(self):
        chat_request = ChatRequestGemini(prompt="Hello world")
        assert chat_request.prompt == "Hello world"
        assert chat_request.model_name == "gemini-2.0-flash-lite"

    def test_chat_request_invalid_model(self):
        with pytest.raises(ValueError):
            # This should raise a validation error due to invalid model name
            ChatRequestGemini(
                prompt="Hello world",
                model_name="invalid-model"  # type: ignore
            )


class TestMistralEndpoint:
    @patch('chat_my_doc_llms.gateway.chat_with_mistral', new_callable=AsyncMock)
    def test_mistral_endpoint_success(self, mock_chat):
        mock_chat.return_value = "Test response from Mistral"
        
        response = client.post("/mistral", json={
            "prompt": "Hello"
        })
        
        assert response.status_code == 200
        assert response.json() == {"message": "Test response from Mistral"}
        mock_chat.assert_called_once_with("Hello")

    # Error handling test removed - FastAPI lets exceptions bubble up by default

    def test_mistral_endpoint_missing_prompt(self):
        response = client.post("/mistral", json={})
        
        assert response.status_code == 422

    @patch('chat_my_doc_llms.gateway.chat_with_mistral', new_callable=AsyncMock)
    def test_mistral_endpoint_error_handling(self, mock_chat):
        mock_chat.side_effect = ValueError("Mistral API Error")
        
        # Since FastAPI doesn't automatically handle exceptions, this will raise
        with pytest.raises(ValueError, match="Mistral API Error"):
            client.post("/mistral", json={
                "prompt": "Hello"
            })
        
        mock_chat.assert_called_once_with("Hello")

    @patch('chat_my_doc_llms.gateway.chat_with_mistral', new_callable=AsyncMock)
    def test_mistral_endpoint_empty_prompt(self, mock_chat):
        mock_chat.return_value = "Response to empty prompt"
        
        response = client.post("/mistral", json={
            "prompt": ""
        })
        
        assert response.status_code == 200
        assert response.json() == {"message": "Response to empty prompt"}
        mock_chat.assert_called_once_with("")


class TestMistralStreamEndpoint:
    @patch('chat_my_doc_llms.gateway.chat_with_mistral_stream')
    def test_mistral_stream_endpoint_success(self, mock_chat_stream):
        async def mock_stream():
            yield "Hello "
            yield "world!"
        
        mock_chat_stream.return_value = mock_stream()
        
        response = client.post("/mistral-stream", json={
            "prompt": "Hello"
        })
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        content = response.content.decode()
        assert "Hello world!" in content
        mock_chat_stream.assert_called_once_with("Hello")

    def test_mistral_stream_endpoint_missing_prompt(self):
        response = client.post("/mistral-stream", json={})
        
        assert response.status_code == 422

    @patch('chat_my_doc_llms.gateway.chat_with_mistral_stream')
    def test_mistral_stream_endpoint_error_handling(self, mock_chat_stream):
        async def mock_error_stream():
            raise ValueError("Mistral Stream API Error")
            yield "never reached"
        
        mock_chat_stream.return_value = mock_error_stream()
        
        # Since FastAPI doesn't automatically handle exceptions, this will raise
        with pytest.raises(ValueError, match="Mistral Stream API Error"):
            client.post("/mistral-stream", json={
                "prompt": "Hello"
            })

    @patch('chat_my_doc_llms.gateway.chat_with_mistral_stream')
    def test_mistral_stream_endpoint_empty_prompt(self, mock_chat_stream):
        async def mock_stream():
            yield "Response "
            yield "to empty prompt"
        
        mock_chat_stream.return_value = mock_stream()
        
        response = client.post("/mistral-stream", json={
            "prompt": ""
        })
        
        assert response.status_code == 200
        content = response.content.decode()
        assert "Response to empty prompt" in content
        mock_chat_stream.assert_called_once_with("")


class TestChatRequestMistralModel:
    def test_chat_request_mistral_valid_data(self):
        chat_request = ChatRequestMistral(prompt="Hello world")
        assert chat_request.prompt == "Hello world"

    def test_chat_request_mistral_empty_prompt(self):
        # Empty prompt should be allowed, test removed as it's valid
        chat_request = ChatRequestMistral(prompt="")
        assert chat_request.prompt == ""


class TestIntegrationEndpoints:
    """Integration-style tests for combined functionality"""
    
    def test_health_endpoint_availability(self):
        """Test that health endpoint is always available"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_root_endpoint_availability(self):
        """Test that root endpoint is always available"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_invalid_endpoint(self):
        """Test behavior for non-existent endpoints"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_http_method(self):
        """Test wrong HTTP methods on endpoints"""
        response = client.get("/gemini")
        assert response.status_code == 405  # Method Not Allowed
        
        response = client.get("/mistral")
        assert response.status_code == 405  # Method Not Allowed

    def test_content_type_requirements(self):
        """Test endpoints require proper content-type"""
        response = client.post("/gemini", data="not json")
        assert response.status_code == 422
        
        response = client.post("/mistral", data="not json")
        assert response.status_code == 422


class TestModelValidation:
    """Test Pydantic model validation edge cases"""
    
    @patch('chat_my_doc_llms.gateway.chat_with_gemini', new_callable=AsyncMock)
    def test_gemini_request_with_extra_fields(self, mock_chat):
        """Test that extra fields are ignored"""
        mock_chat.return_value = "Test response"
        
        response = client.post("/gemini", json={
            "prompt": "Hello",
            "model_name": "gemini-2.0-flash-lite",
            "extra_field": "should be ignored"
        })
        # Should still work despite extra field
        assert response.status_code == 200
        assert response.json() == {"message": "Test response"}
        mock_chat.assert_called_once_with("Hello", "gemini-2.0-flash-lite")

    @patch('chat_my_doc_llms.gateway.chat_with_mistral', new_callable=AsyncMock)
    def test_mistral_request_with_extra_fields(self, mock_chat):
        """Test that extra fields are ignored"""
        mock_chat.return_value = "Test response"
        
        response = client.post("/mistral", json={
            "prompt": "Hello",
            "extra_field": "should be ignored"
        })
        # Should still work despite extra field
        assert response.status_code == 200
        assert response.json() == {"message": "Test response"}
        mock_chat.assert_called_once_with("Hello")

    @patch('chat_my_doc_llms.gateway.chat_with_gemini', new_callable=AsyncMock)
    def test_gemini_request_long_prompt(self, mock_chat):
        """Test with very long prompts"""
        mock_chat.return_value = "Response to long prompt"
        
        long_prompt = "x" * 10000
        response = client.post("/gemini", json={
            "prompt": long_prompt,
            "model_name": "gemini-2.0-flash-lite"
        })
        # Should accept long prompts
        assert response.status_code == 200
        assert response.json() == {"message": "Response to long prompt"}
        mock_chat.assert_called_once_with(long_prompt, "gemini-2.0-flash-lite")

    @patch('chat_my_doc_llms.gateway.chat_with_gemini', new_callable=AsyncMock)
    def test_gemini_request_unicode_prompt(self, mock_chat):
        """Test with unicode characters"""
        mock_chat.return_value = "Response to unicode prompt"
        
        unicode_prompt = "Hello 世界 🌍 café naïve résumé"
        response = client.post("/gemini", json={
            "prompt": unicode_prompt,
            "model_name": "gemini-2.0-flash-lite"
        })
        # Should accept unicode
        assert response.status_code == 200
        assert response.json() == {"message": "Response to unicode prompt"}
        mock_chat.assert_called_once_with(unicode_prompt, "gemini-2.0-flash-lite")
import json
from threading import Thread
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from chat_my_doc_llms.deploy.service import (
    GenerationRequest,
    GenerationResponse,
    MistralService,
    QuantizedMistralLoader,
)


class TestGenerationRequest:
    """Test the GenerationRequest Pydantic model."""
    
    def test_generation_request_valid_data(self):
        request = GenerationRequest(
            prompt="Hello world",
            max_length=256,
            temperature=0.5
        )
        assert request.prompt == "Hello world"
        assert request.max_length == 256
        assert request.temperature == 0.5
    
    def test_generation_request_defaults(self):
        request = GenerationRequest(prompt="Hello world")
        assert request.prompt == "Hello world"
        assert request.max_length == 512
        assert request.temperature == 0.7
    
    def test_generation_request_validation_max_length(self):
        # Test max_length constraints
        with pytest.raises(ValueError):
            GenerationRequest(prompt="Hello", max_length=0)  # Too low
        
        with pytest.raises(ValueError):
            GenerationRequest(prompt="Hello", max_length=2000)  # Too high
    
    def test_generation_request_validation_temperature(self):
        # Test temperature constraints
        with pytest.raises(ValueError):
            GenerationRequest(prompt="Hello", temperature=0.05)  # Too low
        
        with pytest.raises(ValueError):
            GenerationRequest(prompt="Hello", temperature=3.0)  # Too high
    
    def test_generation_request_validation_prompt_length(self):
        # Test prompt max_length constraint
        long_prompt = "x" * 1001  # Exceeds max_length=1000
        with pytest.raises(ValueError):
            GenerationRequest(prompt=long_prompt)


class TestGenerationResponse:
    """Test the GenerationResponse Pydantic model."""
    
    def test_generation_response_valid_data(self):
        model_info = {"status": "loaded", "model_type": "MistralForCausalLM"}
        response = GenerationResponse(
            generated_text="Hello world response",
            prompt="Hello world",
            model_info=model_info
        )
        assert response.generated_text == "Hello world response"
        assert response.prompt == "Hello world"
        assert response.model_info == model_info
    
    def test_generation_response_empty_text(self):
        response = GenerationResponse(
            generated_text="",
            prompt="Hello",
            model_info={}
        )
        assert response.generated_text == ""
        assert response.prompt == "Hello"
        assert response.model_info == {}


class TestQuantizedMistralLoader:
    """Test the QuantizedMistralLoader class."""
    
    def test_init_default_model(self):
        loader = QuantizedMistralLoader()
        assert loader.model_name == "mistralai/Mistral-7B-Instruct-v0.3"
        assert loader.model is None
        assert loader.tokenizer is None
        assert loader.generation_config is None
    
    def test_init_custom_model(self):
        custom_model = "microsoft/DialoGPT-medium"
        loader = QuantizedMistralLoader(model_name=custom_model)
        assert loader.model_name == custom_model
    
    @patch('chat_my_doc_llms.deploy.service.AutoTokenizer.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.AutoModelForCausalLM.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.GenerationConfig')
    @patch('chat_my_doc_llms.deploy.service.torch.set_num_threads')
    def test_load_model_success(self, mock_set_num_threads, mock_gen_config, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model
        
        # Mock generation config
        mock_config = Mock()
        mock_gen_config.return_value = mock_config
        
        loader = QuantizedMistralLoader()
        model, tokenizer = loader.load_model()
        
        # Verify tokenizer loading
        mock_tokenizer_from_pretrained.assert_called_once_with(
            loader.model_name,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=False,
            force_download=False
        )
        
        # Verify pad token setting
        assert mock_tokenizer.pad_token == "<eos>"
        
        # Verify model loading
        mock_model_from_pretrained.assert_called_once_with(
            loader.model_name,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=False,
            force_download=False
        )
        
        # Verify generation config
        mock_gen_config.assert_called_once()
        
        # Verify torch settings
        mock_set_num_threads.assert_called_once_with(2)
        
        # Verify return values
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert loader.model == mock_model
        assert loader.tokenizer == mock_tokenizer
    
    def test_generate_text_not_loaded(self):
        loader = QuantizedMistralLoader()
        with pytest.raises(ValueError, match="Model not loaded"):
            loader.generate_text("Hello world")
    
    @patch('chat_my_doc_llms.deploy.service.torch.no_grad')
    def test_generate_text_success(self, mock_no_grad):
        loader = QuantizedMistralLoader()
        
        # Mock loaded components
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Generated response text"
        loader.tokenizer = mock_tokenizer
        
        mock_model = Mock()
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs
        loader.model = mock_model
        
        mock_config = Mock()
        loader.generation_config = mock_config
        
        # Mock torch.no_grad context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()
        
        result = loader.generate_text("Hello world", max_length=100)
        
        # Verify tokenizer call
        mock_tokenizer.assert_called_once_with(
            "Hello world",
            return_tensors="pt",
            truncation=True,
            max_length=100
        )
        
        # Verify model generation
        mock_model.generate.assert_called_once()
        
        # Verify decoder call
        mock_tokenizer.decode.assert_called_once()
        
        assert result == "Generated response text"
    
    def test_generate_text_stream_not_loaded(self):
        loader = QuantizedMistralLoader()
        with pytest.raises(ValueError, match="Model not loaded"):
            list(loader.generate_text_stream("Hello world"))
    
    @patch('chat_my_doc_llms.deploy.service.TextIteratorStreamer')
    @patch('chat_my_doc_llms.deploy.service.Thread')
    def test_generate_text_stream_success(self, mock_thread_cls, mock_streamer_cls):
        loader = QuantizedMistralLoader()
        
        # Mock loaded components
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        loader.tokenizer = mock_tokenizer
        
        mock_model = Mock()
        loader.model = mock_model
        
        mock_config = Mock()
        loader.generation_config = mock_config
        
        # Mock streamer
        mock_streamer = Mock()
        mock_streamer.__iter__ = Mock(return_value=iter(["Hello", " world", "!"]))
        mock_streamer_cls.return_value = mock_streamer
        
        # Mock thread
        mock_thread = Mock()
        mock_thread_cls.return_value = mock_thread
        
        result = list(loader.generate_text_stream("Hello world"))
        
        # Verify streamer creation
        mock_streamer_cls.assert_called_once_with(
            mock_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Verify thread creation and execution
        mock_thread_cls.assert_called_once()
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once()
        
        assert result == ["Hello", " world", "!"]
    
    def test_get_model_info_not_loaded(self):
        loader = QuantizedMistralLoader()
        info = loader.get_model_info()
        assert info == {"status": "not_loaded"}
    
    def test_get_model_info_loaded(self):
        loader = QuantizedMistralLoader()
        
        # Mock loaded components
        mock_model = Mock()
        mock_model.__class__.__name__ = "MistralForCausalLM"
        loader.model = mock_model
        
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "LlamaTokenizerFast"
        mock_tokenizer.vocab_size = 32768
        loader.tokenizer = mock_tokenizer
        
        info = loader.get_model_info()
        
        expected = {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "model_type": "MistralForCausalLM",
            "tokenizer_type": "LlamaTokenizerFast",
            "vocab_size": 32768,
            "status": "loaded"
        }
        assert info == expected


class TestMistralService:
    """Test the MistralService BentoML service class."""
    
    @patch('chat_my_doc_llms.deploy.service.AutoTokenizer.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.AutoModelForCausalLM.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.os.environ.get')
    @patch('chat_my_doc_llms.deploy.service.login')
    @patch('chat_my_doc_llms.deploy.service.QuantizedMistralLoader')
    def test_init_success(self, mock_loader_cls, mock_login, mock_env_get, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Mock environment
        mock_env_get.side_effect = lambda key, default=None: {
            'HF_TOKEN': 'fake_token',
            'HF_HOME': '/tmp/hf_home',
            'HF_HUB_ENABLE_HF_TRANSFER': '1'
        }.get(key, default)
        
        # Mock model loading calls
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model
        
        # Mock loader
        mock_loader = Mock()
        mock_loader_cls.return_value = mock_loader
        
        service = MistralService()
        
        # Verify HF login
        mock_login.assert_called_once_with('fake_token', add_to_git_credential=False)
        
        # Verify loader creation and model loading
        mock_loader_cls.assert_called_once()
        mock_loader.load_model.assert_called_once()
        
        assert service.model_loader == mock_loader
    
    @patch('chat_my_doc_llms.deploy.service.os.environ.get')
    def test_init_missing_hf_token(self, mock_env_get):
        mock_env_get.return_value = None
        
        with pytest.raises(ValueError, match="HF_TOKEN environment variable is not set"):
            MistralService()
    
    @patch('chat_my_doc_llms.deploy.service.AutoTokenizer.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.AutoModelForCausalLM.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.os.environ.get')
    @patch('chat_my_doc_llms.deploy.service.login')
    @patch('chat_my_doc_llms.deploy.service.QuantizedMistralLoader')
    def test_generate_success(self, mock_loader_cls, mock_login, mock_env_get, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Setup mocks
        mock_env_get.side_effect = lambda key, default=None: {
            'HF_TOKEN': 'fake_token'
        }.get(key, default)
        
        # Mock model loading calls
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model
        
        mock_loader = Mock()
        mock_loader.generate_text.return_value = "Generated response"
        mock_loader.get_model_info.return_value = {"status": "loaded"}
        mock_loader_cls.return_value = mock_loader
        
        service = MistralService()
        
        # Test generate
        request = GenerationRequest(
            prompt="Hello world",
            max_length=256,
            temperature=0.5
        )
        
        response = service.generate(request)
        
        # Verify loader calls
        mock_loader.generate_text.assert_called_once_with(
            prompt="Hello world",
            max_length=256
        )
        mock_loader.get_model_info.assert_called_once()
        
        # Verify response
        assert isinstance(response, GenerationResponse)
        assert response.generated_text == "Generated response"
        assert response.prompt == "Hello world"
        assert response.model_info == {"status": "loaded"}
    
    @patch('chat_my_doc_llms.deploy.service.AutoTokenizer.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.AutoModelForCausalLM.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.os.environ.get')
    @patch('chat_my_doc_llms.deploy.service.login')
    @patch('chat_my_doc_llms.deploy.service.QuantizedMistralLoader')
    @patch('chat_my_doc_llms.deploy.service.bentoml.exceptions.BentoMLException')
    def test_generate_error(self, mock_exception, mock_loader_cls, mock_login, mock_env_get, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Setup mocks
        mock_env_get.side_effect = lambda key, default=None: {
            'HF_TOKEN': 'fake_token'
        }.get(key, default)
        
        # Mock model loading calls
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model
        
        mock_loader = Mock()
        mock_loader.generate_text.side_effect = Exception("Generation failed")
        mock_loader_cls.return_value = mock_loader
        
        service = MistralService()
        
        request = GenerationRequest(prompt="Hello world")
        
        # Test that exception is raised
        mock_exception.side_effect = Exception("BentoML wrapped exception")
        with pytest.raises(Exception):
            service.generate(request)
        
        mock_exception.assert_called_once_with("Generation failed: Generation failed")
    
    @patch('chat_my_doc_llms.deploy.service.AutoTokenizer.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.AutoModelForCausalLM.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.os.environ.get')
    @patch('chat_my_doc_llms.deploy.service.login')
    @patch('chat_my_doc_llms.deploy.service.QuantizedMistralLoader')
    def test_health_success(self, mock_loader_cls, mock_login, mock_env_get, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Setup mocks
        mock_env_get.side_effect = lambda key, default=None: {
            'HF_TOKEN': 'fake_token'
        }.get(key, default)
        
        # Mock model loading calls
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model
        
        mock_loader = Mock()
        mock_loader.get_model_info.return_value = {"status": "loaded", "model_type": "MistralForCausalLM"}
        mock_loader_cls.return_value = mock_loader
        
        service = MistralService()
        
        health_response = service.health()
        
        expected = {
            "status": "healthy",
            "service": "mistral-7b-quantized",
            "model": {"status": "loaded", "model_type": "MistralForCausalLM"}
        }
        assert health_response == expected
    
    @patch('chat_my_doc_llms.deploy.service.AutoTokenizer.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.AutoModelForCausalLM.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.os.environ.get')
    @patch('chat_my_doc_llms.deploy.service.login')
    @patch('chat_my_doc_llms.deploy.service.QuantizedMistralLoader')
    def test_health_error(self, mock_loader_cls, mock_login, mock_env_get, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Setup mocks
        mock_env_get.side_effect = lambda key, default=None: {
            'HF_TOKEN': 'fake_token'
        }.get(key, default)
        
        # Mock model loading calls
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model
        
        mock_loader = Mock()
        mock_loader.get_model_info.side_effect = Exception("Model info failed")
        mock_loader_cls.return_value = mock_loader
        
        service = MistralService()
        
        health_response = service.health()
        
        expected = {
            "status": "unhealthy",
            "error": "Model info failed"
        }
        assert health_response == expected
    
    @patch('chat_my_doc_llms.deploy.service.AutoTokenizer.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.AutoModelForCausalLM.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.os.environ.get')
    @patch('chat_my_doc_llms.deploy.service.login')
    @patch('chat_my_doc_llms.deploy.service.QuantizedMistralLoader')
    @patch('chat_my_doc_llms.deploy.service.json')
    def test_generate_stream_success(self, mock_json, mock_loader_cls, mock_login, mock_env_get, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Setup mocks
        mock_env_get.side_effect = lambda key, default=None: {
            'HF_TOKEN': 'fake_token'
        }.get(key, default)
        
        # Mock model loading calls
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model
        
        mock_loader = Mock()
        mock_loader.generate_text_stream.return_value = iter(["Hello", " world", "!"])
        mock_loader_cls.return_value = mock_loader
        
        # Mock json.dumps
        mock_json.dumps.side_effect = lambda x: str(x)
        
        service = MistralService()
        
        request = GenerationRequest(prompt="Hello world")
        
        # Collect streaming results
        results = list(service.generate_stream(request))
        
        # Verify loader call
        mock_loader.generate_text_stream.assert_called_once_with(
            prompt="Hello world",
            max_length=512
        )
        
        # Verify streaming format
        assert len(results) == 4  # 3 tokens + final event
        assert 'data: ' in results[0]
        assert 'data: ' in results[-1]  # Final event
    
    @patch('chat_my_doc_llms.deploy.service.AutoTokenizer.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.AutoModelForCausalLM.from_pretrained')
    @patch('chat_my_doc_llms.deploy.service.os.environ.get')
    @patch('chat_my_doc_llms.deploy.service.login')
    @patch('chat_my_doc_llms.deploy.service.QuantizedMistralLoader')
    @patch('chat_my_doc_llms.deploy.service.json')
    def test_generate_stream_error(self, mock_json, mock_loader_cls, mock_login, mock_env_get, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Setup mocks
        mock_env_get.side_effect = lambda key, default=None: {
            'HF_TOKEN': 'fake_token'
        }.get(key, default)
        
        # Mock model loading calls
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token = None
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_from_pretrained.return_value = mock_model
        
        mock_loader = Mock()
        mock_loader.generate_text_stream.side_effect = Exception("Streaming failed")
        mock_loader_cls.return_value = mock_loader
        
        # Mock json.dumps
        mock_json.dumps.side_effect = lambda x: str(x)
        
        service = MistralService()
        
        request = GenerationRequest(prompt="Hello world")
        
        # Collect streaming results (should include error)
        results = list(service.generate_stream(request))
        
        # Should get error response
        assert len(results) == 1
        assert 'error' in results[0]
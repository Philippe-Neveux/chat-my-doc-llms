"""
BentoML service for 4-bit quantized Mistral 7B deployment.

This service provides REST API endpoints for text generation using
the quantized Mistral model optimized for CPU deployment.
"""

import gc
import json
import os
import time
from typing import Any, Dict, Generator, Optional

import bentoml
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextIteratorStreamer
from threading import Thread

# Load environment variables from .env file
load_dotenv()


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input text prompt", max_length=1000)
    max_length: Optional[int] = Field(512, description="Maximum tokens to generate", ge=1, le=1024)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    prompt: str
    model_info: Dict[str, Any]


class QuantizedMistralLoader:
    """Loader for language model optimized for CPU inference with dynamic quantization."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
    def load_model(self):
        """Load the Mistral model for CPU inference."""
        total_start_time = time.time()
        logger.info(f"Loading Mistral model for CPU: {self.model_name}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer_start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=False,
            force_download=False
        )
        tokenizer_duration = time.time() - tokenizer_start_time
        logger.info(f"Tokenizer loaded in {tokenizer_duration:.2f} seconds ({tokenizer_duration/60:.2f} minutes)")
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model for CPU inference
        logger.info("Loading model for CPU inference...")
        model_start_time = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=False,
            force_download=False
        )
        model_duration = time.time() - model_start_time
        logger.info(f"Model loaded in {model_duration:.2f} seconds ({model_duration/60:.2f} minutes)")
        
        # Configure generation settings
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Set number of threads for CPU optimization
        torch.set_num_threads(2)
        
        total_duration = time.time() - total_start_time
        logger.info(f"✅ Model loaded successfully! Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        return self.model, self.tokenizer
    
    def generate_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the loaded model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_length
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_text_stream(self, prompt: str, max_length: int = 512) -> Generator[str, None, None]:
        """Generate text using streaming with the loaded model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Update generation config for streaming
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "generation_config": self.generation_config,
            "max_new_tokens": max_length
        }
        
        # Start generation in separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for token in streamer:
            if token:
                yield token
        
        thread.join()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None or self.tokenizer is None:
            return {"status": "not_loaded"}
            
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__,
            "vocab_size": self.tokenizer.vocab_size,
            "status": "loaded"
        }

my_image = (
    bentoml.images.Image(
        python_version="3.11",
        lock_python_packages=False
    )
    .requirements_file("./requirements.txt")
)

load_dotenv()

@bentoml.service(
    name="mistral-service",
    image=my_image,
    envs=[
        {"name": "HF_HOME", "value": "/tmp/hf_home"},
        {"name": "HF_TOKEN", "value": os.environ.get('HF_TOKEN', 'HF_TOKEN Not Found')},
        {"name": "HF_HUB_ENABLE_HF_TRANSFER", "value": "1"}
    ],
    resources={"cpu": "4", "memory": "20Gi"},
    traffic={"timeout": 300},
)
class MistralService:
    """BentoML service for Mistral 7B text generation."""
    
    def _check_env_variables(self):
        """Check if required environment variables are set."""
        logger.info("Checking environment variables...")
        logger.info(f"HF_HOME: {os.environ.get('HF_HOME')}")
        logger.info(f"HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
        
    
    def __init__(self):
        """Initialize the service and load the model."""
        self._check_env_variables()

        logger.info("Initializing Mistral service...")
        self.model_loader = QuantizedMistralLoader()
        
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            login(hf_token, add_to_git_credential=False)
        else:
            raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
        
        # Load model during service initialization
        try:
            self.model_loader.load_model()
            logger.info("Model loaded successfully in service")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    @bentoml.api
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text based on the provided prompt.
        
        Args:
            request: Generation request with prompt and parameters
            
        Returns:
            Generated text response
        """
        try:
            logger.info(f"Received generation request for prompt: {request.prompt[:50]}...")
            
            # Generate text
            generated_text = self.model_loader.generate_text(
                prompt=request.prompt,
                max_length=request.max_length or 512
            )
            
            # Get model info
            model_info = self.model_loader.get_model_info()
            
            # Create response
            response = GenerationResponse(
                generated_text=generated_text,
                prompt=request.prompt,
                model_info=model_info
            )
            
            # Force garbage collection to free memory
            gc.collect()
            
            logger.info("Generation completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise bentoml.exceptions.BentoMLException(f"Generation failed: {str(e)}")
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """
        Health check endpoint.
        
        Returns:
            Service health status and model information
        """
        try:
            model_info = self.model_loader.get_model_info()
            return {
                "status": "healthy",
                "service": "mistral-7b-quantized",
                "model": model_info
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @bentoml.api
    def quick_generate(self, prompt: str) -> str:
        """
        Simple text generation endpoint for quick testing.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text as plain string
        """
        try:
            logger.info(f"Quick generation for: {prompt[:50]}...")
            
            generated_text = self.model_loader.generate_text(
                prompt=prompt,
                max_length=256
            )
            
            # Force garbage collection
            gc.collect()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Quick generation failed: {str(e)}")
            raise bentoml.exceptions.BentoMLException(f"Generation failed: {str(e)}")
    
    @bentoml.api
    def generate_stream(self, request: GenerationRequest) -> Generator[str, None, None]:
        """
        Generate text with streaming response.
        
        Args:
            request: Generation request with prompt and parameters
            
        Yields:
            Generated tokens as Server-Sent Events (SSE)
        """
        try:
            logger.info(f"Received streaming request for prompt: {request.prompt[:50]}...")
            
            # Generate text with streaming
            for token in self.model_loader.generate_text_stream(
                prompt=request.prompt,
                max_length=request.max_length or 512
            ):
                # Format as SSE (Server-Sent Events)
                sse_data = json.dumps({"token": token, "finished": False})
                yield f"data: {sse_data}\n\n"
            
            # Send final event
            final_data = json.dumps({"token": "", "finished": True})
            yield f"data: {final_data}\n\n"
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Streaming generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error during streaming generation: {str(e)}")
            error_data = json.dumps({"error": str(e), "finished": True})
            yield f"data: {error_data}\n\n"
    
    @bentoml.api
    def quick_generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Simple streaming text generation endpoint for quick testing.
        
        Args:
            prompt: Input text prompt
            
        Yields:
            Generated tokens as Server-Sent Events (SSE)
        """
        try:
            logger.info(f"Quick streaming generation for: {prompt[:50]}...")
            
            for token in self.model_loader.generate_text_stream(
                prompt=prompt,
                max_length=256
            ):
                # Format as SSE
                sse_data = json.dumps({"token": token, "finished": False})
                yield f"data: {sse_data}\n\n"
            
            # Send final event
            final_data = json.dumps({"token": "", "finished": True})
            yield f"data: {final_data}\n\n"
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Quick streaming generation failed: {str(e)}")
            error_data = json.dumps({"error": str(e), "finished": True})
            yield f"data: {error_data}\n\n"


# Create service instance
service = MistralService()
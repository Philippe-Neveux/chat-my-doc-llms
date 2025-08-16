"""
BentoML service for 4-bit quantized Mistral 7B deployment.

This service provides REST API endpoints for text generation using
the quantized Mistral model optimized for CPU deployment.
"""

import bentoml
from pydantic import BaseModel, Field
import gc
from loguru import logger
from typing import Optional, Dict, Any

from chat_my_doc_llms.deploy.model import QuantizedMistralLoader


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



my_image = (
    bentoml.images.Image(
        python_version="3.11",
        lock_python_packages=False
    )
    .python_packages(
        "bentoml>=1.2.0",
        "transformers>=4.35.0",
        "torch>=2.0.0,<2.1.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "safetensors>=0.4.0",
        "pydantic>=2.0.0",
        "datasets>=2.14.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "loguru"
    )
)

@bentoml.service(
    name="mistral-service",
    image=my_image,
    envs=[
        {"name": "TRANSFORMERS_CACHE", "value": "/tmp/transformers_cache"},
        {"name": "HF_HOME", "value": "localhost"},
        {"name": "TORCH_HOME", "value": "/tmp/torch_cache"},
    ],
    resources={"cpu": "2000m", "memory": "4Gi"},
    traffic={"timeout": 300}
)
class MistralService:
    """BentoML service for Mistral 7B text generation."""
    
    def __init__(self):
        """Initialize the service and load the model."""
        logger.info("Initializing Mistral service...")
        self.model_loader = QuantizedMistralLoader()
        
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
                max_length=request.max_length
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


# Create service instance
service = MistralService()
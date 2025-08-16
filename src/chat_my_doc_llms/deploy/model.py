"""
Model loader for 4-bit quantized Mistral 7B model.

This module handles the loading and configuration of the quantized model
with optimizations for CPU deployment on limited resources.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from loguru import logger


class QuantizedMistralLoader:
    """Loader for language model optimized for CPU inference with dynamic quantization."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
    def load_model(self):
        """Load the Mistral model for CPU inference."""
        logger.info(f"Loading Mistral model for CPU: {self.model_name}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model for CPU inference with quantization
        logger.info("Loading model for CPU inference...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Apply dynamic quantization for CPU
        logger.info("Applying dynamic quantization...")
        self.model = torch.quantization.quantize_dynamic(
            self.model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
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
        
        logger.info("Model loaded successfully!")
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
            max_length=512
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
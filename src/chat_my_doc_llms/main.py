import os

from dotenv import dotenv_values
from langchain.chat_models import init_chat_model

def chat_with_gemini(message: str):
    

    model = init_chat_model(
        "gemini-2.0-flash",
        model_provider="google_genai"
    )

    result = model.invoke(message)
    
    return result
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def chat_with_gemini(message: str, model_name: str = "gemini-2.0-flash") -> str:
    

    model = init_chat_model(
        model_name,
        model_provider="google_genai"
    )

    result = model.invoke(message)
    
    return result
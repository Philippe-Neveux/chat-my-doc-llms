import os
from typing import Iterator, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

client = genai.Client()

def chat_with_gemini(
    message: str,
    model_name: Literal[
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro"
    ] = "gemini-2.0-flash-lite"
) -> str | None:
    

    response = client.models.generate_content(
        model=model_name,
        contents=message,
    )
    if not response.text:
        raise ValueError("Error from the Gemini API: No text in response with input: {message}")
    
    return response.text


def chat_with_gemini_stream(
    message: str,
    model_name: Literal[
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash", 
        "gemini-1.5-pro"
    ] = "gemini-2.0-flash"
) -> Iterator[str]:
    

    response = client.models.generate_content_stream(
        model=model_name,
        contents=message,
    )
    
    for chunk in response:
        if chunk.text:
            yield chunk.text
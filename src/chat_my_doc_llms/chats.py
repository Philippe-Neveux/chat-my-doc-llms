import json
import os
from typing import AsyncIterator, Literal

import httpx
from dotenv import load_dotenv
from google import genai
from loguru import logger

load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

client = genai.Client()

# Configure Mistral API
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL", "http://35.189.6.238:3000")
logger.info(f"Using Mistral API URL: {MISTRAL_API_URL}")

if not MISTRAL_API_URL or MISTRAL_API_URL == "None":
    MISTRAL_API_URL = "http://35.189.6.238:3000"
    logger.warning(f"MISTRAL_API_URL not set, using default: {MISTRAL_API_URL}")

async def chat_with_gemini(
    message: str,
    model_name: Literal[
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro"
    ] = "gemini-2.0-flash-lite"
) -> str | None:
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=message,
    )
    if not response.text:
        raise ValueError("Error from the Gemini API: No text in response with input: {message}")
    
    return response.text


async def chat_with_gemini_stream(
    message: str,
    model_name: Literal[
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash", 
        "gemini-1.5-pro"
    ] = "gemini-2.0-flash"
) -> AsyncIterator[str]:
    response = await client.aio.models.generate_content_stream(
        model=model_name,
        contents=message,
    )
    
    async for chunk in response:
        if chunk.text:
            yield chunk.text


async def chat_with_mistral(message: str) -> str | None:
    """
    Chat with Mistral LLM using the deployed BentoML service.
    
    Args:
        message: The input message to send to Mistral
        
    Returns:
        The response text from Mistral or None if error
    """
    try:
        logger.info(f"Sending request to Mistral API: {MISTRAL_API_URL}/generate")
        logger.info(f"Prompt: {message[:100]}...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MISTRAL_API_URL}/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "request": {
                        "prompt": message
                    }
                },
                timeout=300.0  # 5 minutes for CPU inference
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("generated_text", "")
                logger.info(f"Successfully received response from Mistral: {len(generated_text)} characters")
                return generated_text
            else:
                raise ValueError(f"Error from Mistral API: {response.status_code} - {response.text}")
                
    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP Error from Mistral API: {e.response.status_code} - {e.response.text}")
    except httpx.ConnectError as e:
        raise ValueError(f"Connection Error to Mistral API: {str(e)}")
    except httpx.TimeoutException as e:
        raise ValueError(f"Timeout Error from Mistral API: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error communicating with Mistral API: {type(e).__name__}: {str(e)}")


async def chat_with_mistral_stream(message: str) -> AsyncIterator[str]:
    """
    Chat with Mistral LLM using streaming response.
    
    Args:
        message: The input message to send to Mistral
        
    Yields:
        Text chunks from the streaming response
    """
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{MISTRAL_API_URL}/generate_stream",
                headers={"Content-Type": "application/json"},
                json={
                    "request": {
                        "prompt": message
                    }
                },
                timeout=300.0  # 5 minutes for CPU inference
            ) as response:
                
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line:
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    if data.get("token"):
                                        yield data["token"]
                                    elif data.get("finished"):
                                        break
                                    elif data.get("error"):
                                        raise ValueError(f"Streaming error: {data['error']}")
                                except json.JSONDecodeError:
                                    continue
                else:
                    raise ValueError(f"Error from Mistral streaming API: {response.status_code} - {(await response.aread()).decode('utf-8')}")
                    
    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP Error from Mistral streaming API: {e.response.status_code} - {e.response.text}")
    except httpx.ConnectError as e:
        raise ValueError(f"Connection Error to Mistral streaming API: {str(e)}")
    except httpx.TimeoutException as e:
        raise ValueError(f"Timeout Error from Mistral streaming API: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error communicating with Mistral streaming API: {type(e).__name__}: {str(e)}")
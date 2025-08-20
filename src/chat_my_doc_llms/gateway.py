from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chat_my_doc_llms.chats import (
    chat_with_gemini,
    chat_with_gemini_stream,
    chat_with_mistral,
    chat_with_mistral_stream,
)


class ChatRequestGemini(BaseModel):
    prompt: str
    model_name: Literal[
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro"
    ] = Field(default="gemini-2.0-flash-lite", description="The model to use for the chat request")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Hey ! How can you help me ?",
                "model_name": "gemini-2.0-flash"
            }
        }
    }
    

class ChatRequestMistral(BaseModel):
    prompt: str
    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Hey ! How can you help me ?"
            }
        }
    }


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"message": "This a new message"}

@app.post("/gemini")
async def gemini(chat: ChatRequestGemini):
    response = await chat_with_gemini(chat.prompt, chat.model_name)
    return {"message": response}

@app.post("/gemini-stream")
async def gemini_stream(chat: ChatRequestGemini):
    async def generate_chunks():
        async for chunk in chat_with_gemini_stream(chat.prompt, chat.model_name):
            yield chunk
    
    return StreamingResponse(
        generate_chunks(),
        media_type="text/plain"
    )

@app.post("/mistral")
async def mistral(chat: ChatRequestMistral):
    response = await chat_with_mistral(chat.prompt)
    return {"message": response}

@app.post("/mistral-stream")
async def mistral_stream(chat: ChatRequestMistral):
    async def generate_chunks():
        async for chunk in chat_with_mistral_stream(chat.prompt):
            yield chunk
    
    return StreamingResponse(
        generate_chunks(),
        media_type="text/plain"
    )

from typing import Literal


from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from chat_my_doc_llms.main import (
    chat_with_gemini,
    chat_with_gemini_stream
)


class ChatRequest(BaseModel):
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


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"message": "This a new message"}

@app.post("/gemini")
async def gemini(chat: ChatRequest):
    response = chat_with_gemini(chat.prompt, chat.model_name)
    return {"message": response}

@app.post("/gemini-stream")
async def gemini_stream(chat: ChatRequest):
    def generate_chunks():
        for chunk in chat_with_gemini_stream(chat.prompt, chat.model_name):
            yield chunk
    
    return StreamingResponse(
        generate_chunks(),
        media_type="text/plain"
    )

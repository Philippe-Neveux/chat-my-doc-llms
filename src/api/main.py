from typing import Literal


from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from chat_my_doc_llms.main import chat_with_gemini

from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.0-flash-lite", model_provider="google_genai")

async def generate_chat_responses(message):
    async for chunk in llm.astream(message):
        yield f"data: {chunk}\n\n"

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
    response = chat_with_gemini(chat.prompt)
    return {"message": response}

@app.post("/gemini-stream")
async def chat_stream(prompt: str):
    return StreamingResponse(
        generate_chat_responses(message=prompt),
        media_type="text/event-stream"
    )


@app.post("/gemini-model")
async def gemini_model(chat: ChatRequest):
    response = chat_with_gemini(chat.prompt, chat.model_name)
    return {"message": response}
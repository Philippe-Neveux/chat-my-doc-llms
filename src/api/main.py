from fastapi import FastAPI

from chat_my_doc_llms.main import chat_with_gemini

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"message": "This a new message"}

@app.post("/gemini")
async def gemini(prompt: str):
    response = chat_with_gemini(prompt)
    return {"message": response}

@app.post("/gemini-model")
async def gemini_model(prompt: str, model_name: str):
    response = chat_with_gemini(prompt, model_name)
    return {"message": response}
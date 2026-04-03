import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chatbot import CareerChatbot


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Data Science Career Assistant Chatbot",
    description="An NLP-powered chatbot for data science career guidance.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

chatbot = CareerChatbot(
    api_key=os.getenv("GOOGLE_API_KEY"),
    search_engine_id=os.getenv("GOOGLE_CSE_ID"),
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message for the chatbot")


class ChatResponse(BaseModel):
    response: str
    mode: str
    sources: list[dict[str, str]]


@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        reply = chatbot.get_response(request.message)
        return ChatResponse(
            response=reply.response,
            mode=reply.mode,
            sources=[
                {
                    "title": source.title,
                    "link": source.link,
                    "snippet": source.snippet,
                }
                for source in reply.sources
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the chatbot response.",
        ) from exc

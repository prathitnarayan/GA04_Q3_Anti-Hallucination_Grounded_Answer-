from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI(title="Grounded Answer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Chunk(BaseModel):
    chunk_id: str
    text: str


class Request(BaseModel):
    question: str
    chunks: List[Chunk]


@app.get("/")
def health():
    return {"status": "running"}


def tokenize(text: str):
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


@app.post("/grounded-answer")
def grounded_answer(req: Request):

    if not req.question.strip():
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    if len(req.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    q_tokens = tokenize(req.question)

    best_chunk = None
    best_score = 0

    for chunk in req.chunks:
        chunk_tokens = tokenize(chunk.text)

        score = len(q_tokens & chunk_tokens)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False,
        }

    confidence = min(0.95, 0.45 + 0.1 * best_score)

    return {
        "answer": best_chunk.text,
        "citations": [best_chunk.chunk_id],
        "confidence": round(confidence, 2),
        "answerable": True,
    }
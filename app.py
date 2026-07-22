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

    scored_chunks = []

    for chunk in req.chunks:

        chunk_tokens = tokenize(chunk.text)

        overlap = q_tokens & chunk_tokens

        score = len(overlap) / max(len(q_tokens), 1)

        if score > 0:
            scored_chunks.append((score, chunk))

    if not scored_chunks:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False,
        }

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    best_score = scored_chunks[0][0]

    # Reject weak matches
    if best_score < 0.2:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": round(best_score, 2),
            "answerable": False,
        }

    # Keep all chunks whose score is close to the best one
    selected = [
        chunk
        for score, chunk in scored_chunks
        if score >= best_score * 0.8
    ][:3]

    answer = " ".join(chunk.text for chunk in selected)

    citations = [chunk.chunk_id for chunk in selected]

    confidence = min(0.95, best_score + 0.35)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": round(confidence, 2),
        "answerable": True,
    }
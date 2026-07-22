"""
Grounded QA API — answers strictly from provided context chunks,
cites source chunk IDs, and returns a calibrated confidence score.

Design goals (compliance-grade RAG):
  - No outside knowledge: answers are extracted verbatim from chunk text,
    never generated/paraphrased by a language model.
  - Deterministic + explainable: pure lexical (TF-IDF) grounding, so the
    same input always produces the same output and every citation is
    traceable to the exact sentence that produced it.
  - Fails closed: if evidence is weak/ambiguous/missing, the API returns
    "I don't know" with confidence <= 0.3 rather than guessing.
"""

import re
import math
import logging
from collections import Counter
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grounded-qa")

app = FastAPI(title="Grounded QA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    chunk_id: str
    text: str


class QARequest(BaseModel):
    question: str = Field(default="")
    chunks: List[Chunk] = Field(default_factory=list)

    @field_validator("question", mode="before")
    @classmethod
    def _coerce_question(cls, v):
        if v is None:
            return ""
        return str(v)

    @field_validator("chunks", mode="before")
    @classmethod
    def _coerce_chunks(cls, v):
        if v is None:
            return []
        return v


class QAResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    answerable: bool


# ---------------------------------------------------------------------------
# Grounding engine
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Tunable thresholds. These were picked to be conservative: better to
# under-answer (say "I don't know") than to hallucinate or cite the
# wrong chunk in a medical/legal context.
ANSWERABLE_SIM_THRESHOLD = 0.28     # min cosine similarity to consider a sentence relevant
ANSWERABLE_OVERLAP_THRESHOLD = 0.30 # min fraction of question keywords present in evidence
SUPPORTING_SIM_MARGIN = 0.18        # how close to top score a sentence must be to also be cited
MAX_SUPPORTING_SENTENCES = 3

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "does", "do", "did", "has", "have", "had", "this", "that", "these",
    "those", "it", "its", "as", "by", "from", "than", "then", "so",
    "can", "could", "will", "would", "should", "may", "might",
}


def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def keywords(text: str) -> set:
    tokens = {t.lower() for t in TOKEN_RE.findall(text)}
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def build_evidence_pool(chunks: List[Chunk]):
    """Flatten chunks into (chunk_id, sentence_text) evidence units."""
    pool = []
    for c in chunks:
        for sent in split_into_sentences(c.text):
            pool.append((c.chunk_id, sent))
    return pool


def is_entity_like(token: str) -> bool:
    """Heuristic for 'this word looks like a proper noun / named entity'."""
    if token.isupper() and len(token) >= 2:
        return True  # acronym, e.g. FAISS, API
    if token[0].isupper() and token[1:].islower():
        return True  # Capitalized, e.g. Qdrant, Rust
    if any(ch.isdigit() for ch in token):
        return True  # version numbers, model names, etc.
    return False


def extract_original_case_tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


# --- Pure-Python TF-IDF + cosine similarity (no numpy/sklearn dependency) ---
# Kept dependency-free on purpose: sklearn/numpy require platform-specific
# compiled wheels that aren't always available (e.g. brand-new Python
# versions on a fresh host), which is a fragile thing to depend on for a
# production deploy. Standard-library TF-IDF is a few dozen lines and has
# zero install-time risk.

TFIDF_STOPWORDS = STOPWORDS  # reuse the same list for corpus-level vectors


def _tokenize_for_tfidf(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in TFIDF_STOPWORDS]


def _tfidf_vectors(documents: List[str]) -> List[Counter]:
    """documents[0] is treated as the query; the rest are candidate sentences.
    Returns one TF-IDF weighted Counter per document, using smoothed IDF
    (same formula sklearn's default uses: ln((1+n)/(1+df)) + 1)."""
    tokenized = [_tokenize_for_tfidf(d) for d in documents]
    n_docs = len(tokenized)

    df = Counter()
    for toks in tokenized:
        for term in set(toks):
            df[term] += 1

    idf = {term: math.log((1 + n_docs) / (1 + d)) + 1.0 for term, d in df.items()}

    vectors = []
    for toks in tokenized:
        tf = Counter(toks)
        vec = Counter()
        for term, count in tf.items():
            vec[term] = count * idf.get(term, 0.0)
        vectors.append(vec)
    return vectors


def _cosine_sim(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(query: str, candidates: List[str]) -> List[float]:
    """Returns a similarity score for each candidate sentence, in the same
    order as `candidates`, using TF-IDF cosine similarity against `query`."""
    if not candidates:
        return []
    vectors = _tfidf_vectors([query] + candidates)
    query_vec = vectors[0]
    return [_cosine_sim(query_vec, v) for v in vectors[1:]]


def answer_question(question: str, chunks: List[Chunk]) -> QAResponse:
    question = (question or "").strip()

    # --- Malformed / empty input guards -----------------------------------
    if not question or not chunks:
        return QAResponse(answer="I don't know", citations=[], confidence=0.0, answerable=False)

    valid_chunks = [c for c in chunks if c.chunk_id and c.text and c.text.strip()]
    if not valid_chunks:
        return QAResponse(answer="I don't know", citations=[], confidence=0.0, answerable=False)

    evidence = build_evidence_pool(valid_chunks)
    if not evidence:
        return QAResponse(answer="I don't know", citations=[], confidence=0.0, answerable=False)

    q_kw = keywords(question)
    if not q_kw:
        return QAResponse(answer="I don't know", citations=[], confidence=0.05, answerable=False)

    # Which question keywords look like named entities (case-sensitive check
    # against the ORIGINAL question text, since keywords() lowercases).
    q_original_tokens = extract_original_case_tokens(question)
    entity_tokens_lower = {
        t.lower() for t in q_original_tokens
        if t.lower() in q_kw and is_entity_like(t)
    }

    # --- Step 1: chunk-level weighted keyword scoring -----------------------
    # doc_freq[token] = number of distinct chunks containing that keyword.
    chunk_keywords = {}
    for c in valid_chunks:
        chunk_keywords[c.chunk_id] = keywords(c.text)

    doc_freq = {}
    for kw in q_kw:
        df = sum(1 for cid, kws in chunk_keywords.items() if kw in kws)
        doc_freq[kw] = df

    chunk_scores = {}
    chunk_overlap_ratio = {}
    for c in valid_chunks:
        cid = c.chunk_id
        matched = q_kw & chunk_keywords[cid]
        score = 0.0
        for kw in matched:
            df = max(doc_freq.get(kw, 1), 1)
            boost = 2.0 if kw in entity_tokens_lower else 1.0
            score += boost / df
        chunk_scores[cid] = score
        chunk_overlap_ratio[cid] = len(matched) / max(len(q_kw), 1)

    ranked_chunks = sorted(chunk_scores.items(), key=lambda x: -x[1])
    top_chunk_id, top_chunk_score = ranked_chunks[0]
    top_overlap = chunk_overlap_ratio[top_chunk_id]

    is_answerable = top_chunk_score > 0 and top_overlap >= ANSWERABLE_OVERLAP_THRESHOLD

    if not is_answerable:
        weak_conf = round(min(top_overlap, 0.3) * 0.8, 2)
        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=max(0.0, min(weak_conf, 0.3)),
            answerable=False,
        )

    # Chunks that are genuinely competitive with the top one (near-tied
    # weighted score) are candidates for a multi-source answer.
    candidate_chunk_ids = [top_chunk_id]
    for cid, score in ranked_chunks[1:]:
        if len(candidate_chunk_ids) >= MAX_SUPPORTING_SENTENCES:
            break
        if score > 0 and score >= top_chunk_score * 0.75 and chunk_overlap_ratio[cid] >= ANSWERABLE_OVERLAP_THRESHOLD:
            candidate_chunk_ids.append(cid)
        else:
            break

    # --- Step 2: sentence-level TF-IDF ranking, restricted to candidate chunks ---
    restricted_evidence = [(cid, s) for cid, s in evidence if cid in candidate_chunk_ids]
    sentences = [s for _, s in restricted_evidence]

    sims = rank_by_similarity(question, sentences) if sentences else []

    order = sorted(range(len(sims)), key=lambda i: -sims[i])

    top_idx = order[0] if order else 0
    top_sim = sims[top_idx] if sims else 0.0

    chosen_idx = [top_idx] if order else []
    for idx in order[1:]:
        if len(chosen_idx) >= MAX_SUPPORTING_SENTENCES:
            break
        score = sims[idx]
        same_or_other_candidate_chunk = restricted_evidence[idx][0] in candidate_chunk_ids
        if score >= max(0.0, top_sim - SUPPORTING_SIM_MARGIN) and same_or_other_candidate_chunk and score >= 0.15:
            chosen_idx.append(idx)
        else:
            break

    chosen_sorted = sorted(set(chosen_idx))
    answer_sentences = []
    citation_ids = []
    for idx in chosen_sorted:
        cid, sent = restricted_evidence[idx]
        answer_sentences.append(sent)
        if cid not in citation_ids:
            citation_ids.append(cid)

    answer_text = " ".join(answer_sentences).strip()
    if not answer_text:
        return QAResponse(answer="I don't know", citations=[], confidence=0.0, answerable=False)

    # --- Calibrate confidence ----------------------------------------------
    # Signals: keyword-weighted chunk score, overlap ratio, sentence-level
    # cosine similarity, and margin over the runner-up chunk (ambiguity).
    second_score = ranked_chunks[1][1] if len(ranked_chunks) > 1 else 0.0
    margin = max(0.0, top_chunk_score - second_score)
    normalized_chunk_score = min(top_chunk_score / 2.0, 1.0)

    raw_conf = (
        0.40 * normalized_chunk_score
        + 0.25 * min(top_overlap, 1.0)
        + 0.20 * min(top_sim / 0.6, 1.0)
        + 0.15 * min(margin / 1.0, 1.0)
    )
    confidence = round(min(max(raw_conf, 0.31), 0.99), 2)

    return QAResponse(
        answer=answer_text,
        citations=citation_ids,
        confidence=confidence,
        answerable=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "grounded-qa-api", "endpoint": "POST /answer"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/answer", response_model=QAResponse)
async def answer(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        qa_request = QARequest(**(body or {}))
    except Exception as e:
        logger.warning(f"Malformed request, falling back to safe default: {e}")
        return JSONResponse(
            content=QAResponse(answer="I don't know", citations=[], confidence=0.0, answerable=False).model_dump()
        )

    result = answer_question(qa_request.question, qa_request.chunks)
    return result


# Accept GET on /answer too, so a misconfigured test client gets a clear
# error instead of a bare 405 with no explanation.
@app.get("/answer")
def answer_get_hint():
    return JSONResponse(
        status_code=405,
        content={"detail": "Use POST /answer with a JSON body: {\"question\": ..., \"chunks\": [...]}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

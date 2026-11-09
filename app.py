import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.graph import build_rag_graph

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic AI RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str = Field(..., max_length=1000)


@app.get("/")
def home():
    return {"message": "Agentic AI RAG API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: QueryRequest):

    question = request.query.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty."
        )

    initial_state = {
        "question": question,
        "context": [],
        "sources": [],
        "retrieval_scores": [],
        "answer": "",
        "grounded": False,
        "score": 0.0,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception:
        logger.exception("Graph execution failed")
        raise HTTPException(
            status_code=503,
            detail="The service is temporarily unavailable. Please try again."
        )

    return {
        "query": question,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["context"],
        "confidence_score": result["score"],
        "grounded": result["grounded"],
        "sources": result["sources"],
    }
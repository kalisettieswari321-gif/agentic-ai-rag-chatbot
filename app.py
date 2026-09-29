from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")

graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Agentic AI RAG API is running"}


@app.post("/chat")
def chat(request: QueryRequest):

    initial_state = {
        "question": request.query,
        "context": [],
        "retrieval_scores": [],
        "answer": "",
        "grounded": False,
        "score": 0.0,
    }

    result = graph.invoke(initial_state)

    return {
        "query": request.query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["context"],
        "confidence_score": result["score"],
    }
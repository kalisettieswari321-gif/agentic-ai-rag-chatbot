# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using Python, LangGraph, Pinecone, Hugging Face embeddings, Groq LLM, and FastAPI.

## Project Overview

This project answers questions using information retrieved from an Agentic AI eBook.

The chatbot is designed to:

- Retrieve relevant information from the eBook
- Generate answers using only the retrieved context
- Refuse questions that are outside the knowledge base
- Calculate a confidence score
- Use LangGraph to organize the RAG workflow
- Expose the chatbot through a FastAPI REST API

## Architecture

User Query
    |
    v
FastAPI
    |
    v
LangGraph
    |
    v
Retrieve relevant chunks from Pinecone
    |
    v
Check retrieval score
    |
    +---- Low score ----> Refuse
    |
    v
Generate answer using retrieved context
    |
    v
Groundedness evaluation
    |
    v
Final answer + confidence score

## Technologies Used

- Python 3.11
- LangChain
- LangGraph
- Pinecone
- Hugging Face Sentence Transformers
- Groq
- FastAPI
- Uvicorn
- PyPDF
- Google Drive / gdown

## Embedding Model

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

Embedding dimension:

`384`

## LLM

The project uses:

`openai/gpt-oss-20b`

through Groq.

Temperature:

`0`

## RAG Configuration

- Chunk size: 1000
- Chunk overlap: 200
- Top K retrieval: 4
- Minimum retrieval score: 0.15
- Retrieval confidence weight: 0.4
- Groundedness confidence weight: 0.6

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   └── ingestion.py
│
├── app.py
├── tests_sample_queries.py
├── requirements.txt
├── .gitignore
└── README.md
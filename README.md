# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using Python, LangGraph, Pinecone, Hugging Face Sentence Transformers, Groq LLM, and FastAPI.

## Project Overview

This project answers questions using information retrieved from an Agentic AI eBook.

The chatbot is designed to:

- Retrieve relevant information from the eBook.
- Generate answers using the retrieved context.
- Refuse questions that are outside the knowledge base.
- Calculate a confidence score based on retrieval quality and groundedness.
- Use LangGraph to organize the agentic RAG workflow.
- Expose the chatbot through a FastAPI REST API.

## Architecture

```text
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
```

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

- Embedding dimension: 384

## LLM

The project uses:

`openai/gpt-oss-20b`

through Groq.

- Temperature: 0

## RAG Configuration

| Parameter | Value |
|---|---|
| Chunk size | 1000 |
| Chunk overlap | 200 |
| Top K retrieval | 4 |
| Minimum retrieval score | 0.15 |
| Retrieval confidence weight | 0.4 |
| Groundedness confidence weight | 0.6 |

## Pinecone Index Settings

| Setting | Value |
|---|---|
| Dimension | 384 |
| Metric | cosine |
| Cloud | AWS |
| Region | us-east-1 |

## Project Structure

```text
agentic-ai-rag-chatbot/
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
├── .env.example
├── .gitignore
├── app.py
├── tests_sample_queries.py
├── requirements.txt
└── README.md
```

## Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/kalisettieswari321-gif/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac / Linux:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root by copying `.env.example`.

**Windows:**

```bash
copy .env.example .env
```

**Mac / Linux:**

```bash
cp .env.example .env
```

Add your API credentials to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index-fastembed
```

`PINECONE_INDEX_NAME` is optional. If it is not set, the default index name `agentic-ai-index-fastembed` defined in `src/config.py` is used.

**Note:** Never upload your actual `.env` file or API credentials to GitHub.

### 5. Add the eBook

The PDF is downloaded automatically from the configured Google Drive file when it is not already available locally.

The expected file path is:

```text
data/Ebook-Agentic-AI.pdf
```

Alternatively, place the PDF manually at this location.

### 6. Ingest the document into Pinecone

Run:

```bash
python -m src.ingestion
```

The ingestion process:

- Downloads the eBook if it is not available locally.
- Loads and extracts text from the PDF.
- Splits the document into chunks of 1000 characters with an overlap of 200 characters.
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Creates the Pinecone index if it does not already exist.
- Uploads document embeddings and metadata to Pinecone.

### 7. Start the API

Run:

```bash
uvicorn app:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 8. Test the chatbot

**Endpoint:** `POST /chat`

**Request:**

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is an AI agent?"}'
```

**Response fields:**

| Field | Description |
|---|---|
| `query` | The question submitted by the user |
| `final_answer` | The generated answer or refusal message |
| `retrieved_context_chunks` | The retrieved document text chunks |
| `confidence_score` | Combined retrieval and groundedness score |

**Example response (illustrative):**

```json
{
  "query": "What is an AI agent?",
  "final_answer": "An AI agent is ...",
  "retrieved_context_chunks": ["...", "..."],
  "confidence_score": 0.82
}
```

### 9. Run sample queries

```bash
python tests_sample_queries.py
```

This runs sample questions to check the chatbot's answering and refusal behavior.

## How the Confidence Score Works

The chatbot calculates its confidence score using:

```text
confidence_score = (0.4 * average_retrieval_score) + (0.6 * groundedness_score)
```

- **Average retrieval score:** The average similarity score of the retrieved document chunks.
- **Groundedness score:** Measures how well the generated answer is supported by the retrieved context.
- **Retrieval weight:** 0.4.
- **Groundedness weight:** 0.6.

Groundedness receives a higher weight because retrieving relevant information does not automatically guarantee that the generated answer is factually supported by that information.

**Refusal condition:**

The chatbot refuses to generate an

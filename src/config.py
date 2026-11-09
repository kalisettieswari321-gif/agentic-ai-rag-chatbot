import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index-fastembed"
)

PDF_SHARE_URL = "https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing"
PDF_PATH = "data/Ebook-Agentic-AI.pdf"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4
MIN_RETRIEVAL_SCORE = 0.15
MIN_GROUNDEDNESS = 0.7

W_RETRIEVAL = 0.4
W_GROUNDED = 0.6

REFUSAL_MESSAGE = "I cannot answer this based on the provided eBook."

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index-free"
)

PDF_SHARE_URL = "https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing"
PDF_PATH = "data/Ebook-Agentic-AI.pdf"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4
MIN_RETRIEVAL_SCORE = 0.15

W_RETRIEVAL = 0.4
W_GROUNDED = 0.6

REFUSAL_MESSAGE = "I cannot answer this based on the provided eBook."
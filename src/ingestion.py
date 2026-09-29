import os
import time

import gdown
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


def download_pdf():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(config.PDF_PATH):
        print("PDF already exists. Skipping download.")
        return config.PDF_PATH

    gdown.download(
    id="15VLphKcY23_fpYxN62UEQRri_psRVfP9",
    output=config.PDF_PATH,
    quiet=False
)

    if not os.path.exists(config.PDF_PATH):
        raise RuntimeError("PDF download failed.")

    return config.PDF_PATH


def load_and_chunk(pdf_path):
    docs = PyPDFLoader(pdf_path).load()

    for doc in docs:
        doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
        doc.metadata["source"] = os.path.basename(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(docs)

    print(f"Loaded {len(docs)} pages -> {len(chunks)} chunks")
    return chunks


def create_index_if_needed():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        while not pc.describe_index(
            config.PINECONE_INDEX_NAME
        ).status["ready"]:
            time.sleep(1)

        print("Created Pinecone index.")
    else:
        print("Pinecone index already exists.")


def run_ingestion():
    pdf_path = download_pdf()
    chunks = load_and_chunk(pdf_path)

    create_index_if_needed()

    embeddings = FastEmbedEmbeddings(
       model_name="BAAI/bge-small-en-v1.5"
)

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
    )

    print(f"Done! Stored {len(chunks)} chunks in Pinecone.")


if __name__ == "__main__":
    run_ingestion()
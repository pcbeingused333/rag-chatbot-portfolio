"""
rag_core.py — Shared RAG logic (embeddings, vector store, ingestion, citations).

Both ingest.py (batch ingestion of a folder) and app.py (real-time upload from the
UI) import from here so the pipeline is defined in exactly one place.

Heavy dependencies (sentence-transformers/torch, the Postgres driver, the PDF
loaders) are imported lazily inside the functions that need them, so importing this
module — and the unit tests — stay fast and dependency-light.
"""
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ---- Configuration (overridable via environment) ----
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "7"))


def get_connection_string() -> str:
    """Return the Postgres connection string, failing loudly if it is missing."""
    conn = os.getenv("POSTGRES_CONNECTION")
    if not conn:
        raise RuntimeError(
            "POSTGRES_CONNECTION is not set. Copy .env.example to .env and fill it in."
        )
    return conn


def get_embeddings():
    """Build the embedding model (heavy import kept local)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore(embeddings=None):
    """Return a PGVector store bound to our collection."""
    from langchain_postgres.vectorstores import PGVector

    return PGVector(
        connection=get_connection_string(),
        collection_name=COLLECTION_NAME,
        embeddings=embeddings or get_embeddings(),
        use_jsonb=True,
    )


def split_documents(docs: List[Document]) -> List[Document]:
    """Split loaded documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def ingest_pdf_paths(paths: List[str], vectorstore=None) -> int:
    """
    Ingest one or more PDF files into the vector store (used by real-time upload).

    Returns the number of chunks added. Additive — does not clear the collection.
    """
    from langchain_community.document_loaders import PyPDFLoader

    store = vectorstore or get_vectorstore()
    all_docs: List[Document] = []
    for path in paths:
        all_docs.extend(PyPDFLoader(path).load())
    if not all_docs:
        return 0
    chunks = split_documents(all_docs)
    store.add_documents(chunks)
    return len(chunks)


def ingest_directory(directory: str = "data/", pre_delete: bool = True) -> int:
    """
    Ingest every PDF in a directory (used by the batch ingest.py script).

    When pre_delete is True the collection is rebuilt from scratch.
    """
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_postgres.vectorstores import PGVector

    docs = PyPDFDirectoryLoader(directory).load()
    if not docs:
        return 0
    chunks = split_documents(docs)
    PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=get_connection_string(),
        use_jsonb=True,
        pre_delete_collection=pre_delete,
    )
    return len(chunks)


def format_citation(doc: Document) -> str:
    """
    Build a human-readable citation from a retrieved chunk's metadata.

    PyPDFLoader stores the file path in `source` and a 0-indexed `page`;
    we surface the file name and a 1-indexed page number.
    """
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")
    if page is not None:
        return f"{source} — p. {int(page) + 1}"
    return source

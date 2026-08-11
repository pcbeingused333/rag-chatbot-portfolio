"""
rag_core.py — Shared RAG logic (embeddings, vector store, ingestion, citations).

Both ingest.py (batch ingestion of a folder) and app.py (real-time upload from the
UI) import from here so the pipeline is defined in exactly one place.

Two runtime modes, selected with the DEMO_MODE environment variable:

  DEMO_MODE=1  Self-contained demo. A small embedding model plus an in-memory FAISS
               index seeded with the PDFs in demo/. No external database, and it
               fits in the 1 GB of a free Streamlit Cloud container.
  DEMO_MODE=0  Production path (the default). Full-size multilingual embeddings
               backed by PostgreSQL + pgvector.

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

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Configuration (overridable via environment) ----
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")

# Demo uses a ~90 MB model; production uses a ~2 GB multilingual one that would
# OOM a free-tier container.
DEMO_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROD_EMBEDDING_MODEL = "BAAI/bge-m3"

# The demo corpus is a single short, section-structured document. With
# production-sized chunks it splits into fewer pieces than RETRIEVAL_K, so every
# query retrieves the whole document and retrieval stops discriminating — the demo
# would look like it works while proving nothing.
#
# These values were measured, not guessed: against four questions with known
# answers, 300/50/k=4 puts the correct passage at rank 1 for 4/4, versus 3/4 at
# 400/80 and 2/4 at 200/30 (smaller chunks fragment the price list).
DEMO_CHUNK_SIZE, DEMO_CHUNK_OVERLAP, DEMO_RETRIEVAL_K = 300, 50, 4
PROD_CHUNK_SIZE, PROD_CHUNK_OVERLAP, PROD_RETRIEVAL_K = 1000, 200, 7

DEMO_DIR = os.getenv("DEMO_DIR", os.path.join(HERE, "demo"))

_TRUTHY = {"1", "true", "yes", "on"}


def is_demo_mode() -> bool:
    """
    Whether the self-contained demo path is active.

    Read live (not cached at import) so tests and Streamlit's secrets bridge can
    flip it without reimporting the module.
    """
    return os.getenv("DEMO_MODE", "0").strip().lower() in _TRUTHY


def _resolve_int(env_var: str, demo_value: int, prod_value: int) -> int:
    """Mode-aware integer setting; an explicit env var always wins."""
    explicit = os.getenv(env_var)
    if explicit:
        return int(explicit)
    return demo_value if is_demo_mode() else prod_value


def resolve_embedding_model() -> str:
    """Embedding model for the current mode; an explicit env var always wins."""
    explicit = os.getenv("EMBEDDING_MODEL")
    if explicit:
        return explicit
    return DEMO_EMBEDDING_MODEL if is_demo_mode() else PROD_EMBEDDING_MODEL


def chunk_size() -> int:
    return _resolve_int("CHUNK_SIZE", DEMO_CHUNK_SIZE, PROD_CHUNK_SIZE)


def chunk_overlap() -> int:
    return _resolve_int("CHUNK_OVERLAP", DEMO_CHUNK_OVERLAP, PROD_CHUNK_OVERLAP)


def retrieval_k() -> int:
    return _resolve_int("RETRIEVAL_K", DEMO_RETRIEVAL_K, PROD_RETRIEVAL_K)


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

    return HuggingFaceEmbeddings(model_name=resolve_embedding_model())


def demo_pdf_paths() -> List[str]:
    """The sample PDFs shipped with the repo that seed the demo knowledge base."""
    if not os.path.isdir(DEMO_DIR):
        return []
    return sorted(
        os.path.join(DEMO_DIR, name)
        for name in os.listdir(DEMO_DIR)
        if name.lower().endswith(".pdf")
    )


def build_demo_vectorstore(embeddings=None):
    """
    In-memory FAISS index seeded with the demo PDFs — no external database.

    Rebuilt on every cold start, which is fine: the corpus is small and this is
    what keeps the public demo alive without a Postgres instance that can pause.
    """
    paths = demo_pdf_paths()
    if not paths:
        raise RuntimeError(
            f"DEMO_MODE is on but no PDF was found in {DEMO_DIR}. "
            "Run: python demo/make_demo_pdf.py"
        )

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS

    docs: List[Document] = []
    for path in paths:
        docs.extend(PyPDFLoader(path).load())
    return FAISS.from_documents(split_documents(docs), embeddings or get_embeddings())


def get_vectorstore(embeddings=None):
    """Vector store for the current mode: in-memory FAISS in demo, pgvector otherwise."""
    if is_demo_mode():
        return build_demo_vectorstore(embeddings)

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
        chunk_size=chunk_size(), chunk_overlap=chunk_overlap()
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
    if is_demo_mode():
        raise RuntimeError(
            "Batch ingestion writes to pgvector and is not available in DEMO_MODE "
            "(the demo index is in-memory and rebuilt from demo/ on startup). "
            "Unset DEMO_MODE to ingest into Postgres."
        )

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


# Groq returns HTTP 400 tool_use_failed when the model emits a malformed tool call.
# It is a per-request glitch, not a bad question, so the same input usually succeeds
# on a retry — hence retrying rather than surfacing the error.
_RETRYABLE_MARKERS = (
    "tool_use_failed",
    "failed to call a function",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "overloaded",
    "timeout",
)


def is_transient_llm_error(exc: BaseException) -> bool:
    """Whether an LLM call failed for a reason worth retrying with the same input."""
    return any(marker in str(exc).lower() for marker in _RETRYABLE_MARKERS)


def invoke_agent_with_retry(agent, messages, attempts: int = 3):
    """
    Run the agent, retrying transient LLM failures.

    Returns the agent's final message content. Raises the last exception if every
    attempt fails, or immediately for errors that a retry cannot fix (a bad API key
    should not be retried three times).
    """
    last: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            result = agent.invoke({"messages": messages})
            return result["messages"][-1].content
        except Exception as exc:  # noqa: BLE001 — re-raised below
            last = exc
            if not is_transient_llm_error(exc) or attempt == attempts - 1:
                raise
    raise last  # unreachable, kept for type-checkers


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

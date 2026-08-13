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
# bge-small-en-v1.5 replaced all-MiniLM-L6-v2 when the corpus became the GDPR. On a
# two-page business document the two were indistinguishable; over 414 provisions, where
# the competing passages are sibling paragraphs of the same article, MiniLM ranks the
# right provision first for 9 of 20 questions and bge-small for 12. It costs 42 MB more
# (python -m evals.run_eval memory) and has a 512-token window rather than 256.
DEMO_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
PROD_EMBEDDING_MODEL = "BAAI/bge-m3"

# Measured, not guessed (python -m evals.run_eval retrieval --sweep).
#
# 1500 is large enough that the splitter almost never cuts a provision in half: the
# corpus holds 414 provisions and this produces 431 chunks, so the great majority are
# one provision exactly. That matters more here than the raw score — a chunk that
# straddles Article 33(1) and 33(2) is attributed to one of them, and a citation that
# points at the wrong paragraph is the failure this whole design exists to prevent.
#
# k=4 rather than 8 because with the article heading embedded (see embed_headings)
# k=8 buys nothing: both reach recall@k 17/20 and MRR 0.74. Half the retrieved
# context per request is not free — k=8 at this chunk size was large enough to
# exhaust Groq's per-minute token budget mid-run, which is the 413 handled in
# invoke_agent_with_retry.
#
# 300/50 was the measured optimum for the previous corpus, a two-page business
# document. It scores 65%/80% on this one. The setting did not go stale; the corpus
# changed, and this is what re-running the sweep is for.
DEMO_CHUNK_SIZE, DEMO_CHUNK_OVERLAP, DEMO_RETRIEVAL_K = 1500, 200, 4
PROD_CHUNK_SIZE, PROD_CHUNK_OVERLAP, PROD_RETRIEVAL_K = 1000, 200, 7

DEMO_DIR = os.getenv("DEMO_DIR", os.path.join(HERE, "demo"))

# The demo knowledge base is the GDPR, built structurally by corpus/build_gdpr_corpus.py
# straight from the EUR-Lex text. It is JSONL rather than PDF because the citation unit
# of a regulation is the provision, not the page: nobody looks up page 14 of the GDPR,
# and which page a provision lands on is an artefact of typesetting. Uploaded PDFs still
# work and still cite pages — that path is unchanged.
CORPUS_DIR = os.getenv("CORPUS_DIR", os.path.join(HERE, "corpus"))
LEGAL_CORPUS_PATH = os.path.join(CORPUS_DIR, "gdpr_en.jsonl")

# Metadata keys carried from the corpus into every chunk, so a retrieved fragment
# still knows which provision it came from.
PROVISION_KEYS = (
    "citation", "article", "paragraph", "article_title",
    "chapter", "chapter_title", "instrument", "source_url",
)

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


def embed_headings() -> bool:
    """
    Whether the citation and article title are part of the embedded text.

    On by default, but only because it was measured twice. The heading is identical
    across every paragraph of an article, so the obvious worry is that it drags
    sibling provisions together exactly where they most need telling apart — and
    under all-MiniLM-L6-v2 that is what happens: dropping the heading took hit@1
    from 4/20 to 8/20. Under bge-small-en-v1.5, the model that actually ships, it
    reverses: the heading helps in four of the five sweep configurations, and at the
    shipped one it is 13/20 with against 12/20 without.

    The lesson is in the reversal. The first result was true of a component that is
    no longer in the system, and carrying it forward would have shipped the worse
    setting on the strength of a real measurement.
    """
    return os.getenv("EMBED_HEADINGS", "1").strip().lower() in _TRUTHY


def load_legal_corpus(path: Optional[str] = None) -> List[Document]:
    """
    The GDPR knowledge base: one Document per provision, carrying its citation.

    Each record is already a self-contained provision, so this does no splitting —
    that is `split_documents`' job, and the point of the sweep in `evals/` is to
    find a chunk size that leaves most provisions whole.
    """
    import json

    corpus_path = path or LEGAL_CORPUS_PATH
    with_headings = embed_headings()
    if not os.path.exists(corpus_path):
        raise RuntimeError(
            f"The legal corpus is missing at {corpus_path}. "
            "Run: python corpus/build_gdpr_corpus.py"
        )

    docs: List[Document] = []
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = {key: record[key] for key in PROVISION_KEYS if key in record}
            metadata["source"] = record.get("instrument_long", record.get("instrument"))
            content = record["text"] if with_headings else record.get("body", record["text"])
            docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_demo_vectorstore(embeddings=None):
    """
    In-memory FAISS index over the legal corpus — no external database.

    Rebuilt on every cold start, which is fine: the corpus is small and this is
    what keeps the public demo alive without a Postgres instance that can pause.
    """
    from langchain_community.vectorstores import FAISS

    docs = load_legal_corpus()
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


def ingest_legal_corpus(pre_delete: bool = True) -> int:
    """
    Load the structured legal corpus into pgvector (the production path).

    Demo mode rebuilds the index in memory on every cold start, so it never needs
    this. Production does: without it the only way to get the Regulation into
    Postgres would be through the PDF loader, which is exactly the round-trip that
    would throw away the article structure the citations depend on.
    """
    if is_demo_mode():
        raise RuntimeError(
            "Batch ingestion writes to pgvector and is not available in DEMO_MODE "
            "(the demo index is in-memory and rebuilt on startup). "
            "Unset DEMO_MODE to ingest into Postgres."
        )

    from langchain_postgres.vectorstores import PGVector

    chunks = split_documents(load_legal_corpus())
    PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=get_connection_string(),
        use_jsonb=True,
        pre_delete_collection=pre_delete,
    )
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
    # Groq reports a tokens-per-minute exhaustion as 413 "Request too large", which
    # reads like an oversized payload and is really a rate limit with a clock on it.
    # The abstention eval surfaced this: raising k to 8 made each request big enough
    # to trip the per-minute budget, and because 413 was not on this list the run
    # died on the first question instead of waiting a beat. It is only retryable
    # with a delay, which is why _backoff_seconds exists.
    "413",
    "request too large",
    "tokens per minute",
)


def _backoff_seconds(attempt: int) -> float:
    """
    Delay before retry number `attempt` (0-indexed).

    A malformed tool call is worth retrying instantly. A per-minute token budget is
    not: retrying it immediately just spends another request to be told the same
    thing, so the wait has to be long enough for the window to move.
    """
    return min(2.0 * (2 ** attempt), 30.0)


def is_transient_llm_error(exc: BaseException) -> bool:
    """Whether an LLM call failed for a reason worth retrying with the same input."""
    return any(marker in str(exc).lower() for marker in _RETRYABLE_MARKERS)


def is_rate_limited(exc: BaseException) -> bool:
    """Whether a failure is a budget that refills with time rather than a glitch."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("rate limit", "429", "413", "request too large", "tokens per minute")
    )


def invoke_agent_with_retry(agent, messages, attempts: int = 3, sleep=None):
    """
    Run the agent, retrying transient LLM failures.

    Returns the agent's final message content. Raises the last exception if every
    attempt fails, or immediately for errors that a retry cannot fix (a bad API key
    should not be retried three times).

    `sleep` is injectable so the tests can exercise the backoff without waiting for
    it. Only rate-limit failures wait; a malformed tool call retries immediately.
    """
    import time

    sleep = sleep or time.sleep
    last: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            result = agent.invoke({"messages": messages})
            return result["messages"][-1].content
        except Exception as exc:  # noqa: BLE001 — re-raised below
            last = exc
            if not is_transient_llm_error(exc) or attempt == attempts - 1:
                raise
            if is_rate_limited(exc):
                sleep(_backoff_seconds(attempt))
    raise last  # unreachable, kept for type-checkers


# ---- Agent construction ----
# The agent lives here rather than in app.py so that the evaluation harness runs the
# same prompt, the same tool and the same retriever the user talks to. An eval that
# builds its own copy of the agent measures its own copy, not the product.

# llama-3.3-70b-versatile emits malformed tool calls often enough on Groq to break
# the agent on roughly half of all questions. Reproduce with:
#     python -m evals.run_eval tool-calls
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"

BASE_SYSTEM_PROMPT = (
    "You are a research assistant over a corpus of legal and regulatory source "
    "texts. You help the user find what the instruments actually say, and you make "
    "every statement checkable against the provision it came from.\n"
    "RULES:\n"
    "1. For ANY question about the content of the corpus, ALWAYS use the "
    "`search_documents` tool before answering.\n"
    "2. Answer DIRECTLY and COMPLETELY from the retrieved text. Give the substance "
    "— the obligation, the deadline, the threshold, the condition — not a summary "
    "of the fact that you searched.\n"
    "3. CITE THE PROVISION for every statement you make, in the form the retrieved "
    "passage gives it (for example 'Art. 33(1)'). A statement of law without its "
    "citation is unusable: the reader cannot verify it. Never cite a provision that "
    "was not in the retrieved passages, and never adjust a citation to make it look "
    "more precise than the passage supports.\n"
    "4. Quote thresholds, deadlines, figures and conditions exactly as written "
    "(‘72 hours’, ‘one month’, ‘EUR 20 000 000’, ‘4 % of the total worldwide annual "
    "turnover’). Do not round them, convert currencies, or restate them.\n"
    "5. Answer ONLY from the retrieved passages. You may know things about this "
    "area of law from training; that knowledge is not a source here and must not "
    "appear in an answer. If the search does not return a provision that answers "
    "the question, say plainly that the corpus does not cover it. An unsupported "
    "answer is a worse outcome than no answer.\n"
    "6. Distinguish what the text says from what it would mean for the user. You "
    "describe provisions; you do not advise on a specific situation, and you say so "
    "when a question asks you to.\n"
    "7. ALWAYS reply in the same language the user writes in.\n"
    "8. NEVER refuse a question as off-topic without searching first. If it could "
    "plausibly concern the corpus — an obligation, a right, a definition, a "
    "procedure, a penalty, a role — call `search_documents` before concluding "
    "anything. Only after a search comes back empty may you say so."
)

# Demo mode indexes with an English-only embedding model (the multilingual ones do
# not fit the free tier's 1 GB), so a Spanish question retrieves nothing from the
# English knowledge base. Translating the *query* while still answering in the
# user's language costs no extra memory. Measured effect, Spanish hit@1: 0/4 as
# asked, 3/4 translated (python -m evals.run_eval multilingual).
CROSS_LINGUAL_RULE = (
    "\n9. The corpus is indexed in ENGLISH and the retriever is not multilingual. "
    "ALWAYS write the `search_documents` query in ENGLISH, translating the user's "
    "question when needed, using the wording you would expect to find in the "
    "instrument itself. This does not change rule 7: still reply in the user's own "
    "language."
)


def resolve_llm_model() -> str:
    """The chat model for the current environment; an explicit env var always wins."""
    return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)


def build_system_prompt(demo_mode: Optional[bool] = None) -> str:
    """System prompt for the agent; the cross-lingual rule is demo-mode only."""
    if demo_mode is None:
        demo_mode = is_demo_mode()
    return BASE_SYSTEM_PROMPT + (CROSS_LINGUAL_RULE if demo_mode else "")


def make_search_tool(retriever, on_documents=None):
    """
    The agent's one tool: retrieve passages and return them with their citations.

    `on_documents` receives the retrieved chunks, which is how a caller keeps hold
    of them after the fact — the UI to render sources, the eval to score contexts.
    """
    from langchain_core.tools import tool

    @tool
    def search_documents(query: str) -> str:
        """Search and return relevant snippets from the user's PDF documents.
        Use this to answer any question about the content of the documents.
        In demo mode the index is English-only, so write the query in English."""
        docs = retriever.invoke(query)
        if on_documents is not None:
            on_documents(docs)
        if not docs:
            return "No relevant information was found in the documents."
        return "\n\n".join(f"[{format_citation(d)}]\n{d.page_content}" for d in docs)

    return search_documents


def build_agent(vectorstore=None, on_documents=None, llm=None, model: Optional[str] = None,
                temperature: float = 0.2):
    """The ReAct agent, wired to the current mode's vector store and model."""
    from langgraph.prebuilt import create_react_agent

    store = vectorstore if vectorstore is not None else get_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": retrieval_k()})

    if llm is None:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model or resolve_llm_model(), temperature=temperature)

    return create_react_agent(
        llm, [make_search_tool(retriever, on_documents)], prompt=build_system_prompt()
    )


def format_citation(doc: Document) -> str:
    """
    Build a citation from a retrieved chunk's metadata.

    Two shapes, because the two corpora have different citable units:

    - A provision from the legal corpus cites the way the instrument is actually
      cited — "GDPR Art. 17(1) — Right to erasure". A page number would be useless
      here: it is a property of the rendering, not of the law, and a reader sent to
      "page 14" cannot confirm anything.
    - An uploaded PDF has no structure to cite, so it falls back to file and
      1-indexed page (PyPDFLoader stores the path in `source` and a 0-indexed `page`).
    """
    citation = doc.metadata.get("citation")
    if citation:
        title = doc.metadata.get("article_title")
        return f"{citation} — {title}" if title else citation

    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")
    if page is not None:
        return f"{source} — p. {int(page) + 1}"
    return source


def cited_provision(doc: Document) -> Optional[str]:
    """
    The bare provision reference for a chunk ("GDPR Art. 17(1)"), or None.

    This is what the evaluation scores against. Ground truth held as structural
    metadata rather than as a substring of the text removes a whole class of false
    misses: a chunk boundary can split the wording of a provision, but it cannot
    split the provision the chunk came from.
    """
    return doc.metadata.get("citation")

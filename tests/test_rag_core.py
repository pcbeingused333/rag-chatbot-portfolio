"""
Unit tests for the pure RAG logic (no Postgres or Groq needed).

    pytest -q
"""
import os

from langchain_core.documents import Document

import rag_core


def test_split_documents_creates_overlapping_chunks():
    long_text = "palabra " * 800  # ~5600 chars -> multiple chunks
    docs = [Document(page_content=long_text, metadata={"source": "x.pdf", "page": 0})]
    chunks = rag_core.split_documents(docs)
    assert len(chunks) > 1
    # chunk size honored (allow slack for splitter boundaries)
    limit = rag_core.chunk_size() + rag_core.chunk_overlap()
    assert all(len(c.page_content) <= limit for c in chunks)
    # metadata is preserved on every chunk
    assert all(c.metadata.get("source") == "x.pdf" for c in chunks)


def test_format_citation_uses_basename_and_1indexed_page():
    doc = Document(page_content="...", metadata={"source": "/tmp/data/menu.pdf", "page": 3})
    assert rag_core.format_citation(doc) == "menu.pdf — p. 4"


def test_format_citation_without_page():
    doc = Document(page_content="...", metadata={"source": "notes.pdf"})
    assert rag_core.format_citation(doc) == "notes.pdf"


def test_get_connection_string_raises_when_missing(monkeypatch):
    monkeypatch.delenv("POSTGRES_CONNECTION", raising=False)
    try:
        rag_core.get_connection_string()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "POSTGRES_CONNECTION" in str(e)


def test_demo_pdf_exists_and_is_readable():
    """The sample knowledge base should ship with the repo."""
    demo = os.path.join(os.path.dirname(__file__), "..", "demo", "churreria_calderon.pdf")
    assert os.path.exists(demo), "run: python demo/make_demo_pdf.py"
    assert os.path.getsize(demo) > 1000


# ---- DEMO_MODE ----

def test_is_demo_mode_accepts_common_truthy_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on", " 1 "):
        monkeypatch.setenv("DEMO_MODE", value)
        assert rag_core.is_demo_mode() is True, value


def test_is_demo_mode_defaults_to_off(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert rag_core.is_demo_mode() is False
    monkeypatch.setenv("DEMO_MODE", "0")
    assert rag_core.is_demo_mode() is False


def test_demo_mode_selects_the_small_embedding_model(monkeypatch):
    """The 2 GB multilingual model OOMs a free 1 GB container; demo must not use it."""
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    monkeypatch.setenv("DEMO_MODE", "1")
    assert rag_core.resolve_embedding_model() == rag_core.DEMO_EMBEDDING_MODEL

    monkeypatch.setenv("DEMO_MODE", "0")
    assert rag_core.resolve_embedding_model() == rag_core.PROD_EMBEDDING_MODEL


def test_explicit_embedding_model_overrides_the_mode_default(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
    assert rag_core.resolve_embedding_model() == "custom/model"


def test_demo_chunking_is_smaller_than_production(monkeypatch):
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("RETRIEVAL_K", raising=False)

    monkeypatch.setenv("DEMO_MODE", "1")
    demo_chunk, demo_k = rag_core.chunk_size(), rag_core.retrieval_k()
    monkeypatch.setenv("DEMO_MODE", "0")
    assert demo_chunk < rag_core.chunk_size()
    assert demo_k < rag_core.retrieval_k()


def test_explicit_env_overrides_mode_defaults(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("RETRIEVAL_K", "9")
    assert rag_core.chunk_size() == 1234
    assert rag_core.retrieval_k() == 9


def test_demo_corpus_splits_into_more_chunks_than_k(monkeypatch):
    """
    Regression guard: if the demo splits into fewer chunks than RETRIEVAL_K, every
    query retrieves the entire corpus and retrieval stops discriminating — the demo
    would look like it works while proving nothing.
    """
    from langchain_community.document_loaders import PyPDFLoader

    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("RETRIEVAL_K", raising=False)

    docs = []
    for path in rag_core.demo_pdf_paths():
        docs.extend(PyPDFLoader(path).load())
    chunks = rag_core.split_documents(docs)
    assert len(chunks) > 2 * rag_core.retrieval_k(), (
        f"demo corpus splits into only {len(chunks)} chunks for k={rag_core.retrieval_k()}"
    )


def test_demo_pdf_paths_finds_the_shipped_knowledge_base():
    paths = rag_core.demo_pdf_paths()
    assert paths, "expected at least one PDF in demo/"
    assert all(p.lower().endswith(".pdf") for p in paths)
    assert any(os.path.basename(p) == "churreria_calderon.pdf" for p in paths)


def test_demo_pdf_paths_is_empty_when_directory_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_core, "DEMO_DIR", str(tmp_path / "does-not-exist"))
    assert rag_core.demo_pdf_paths() == []


def test_build_demo_vectorstore_fails_loudly_without_pdfs(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_core, "DEMO_DIR", str(tmp_path))  # empty dir
    try:
        rag_core.build_demo_vectorstore(embeddings=object())
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "DEMO_MODE" in str(e)


# ---- Resilience against flaky LLM tool calls ----

class _FakeAgent:
    """Fails `failures` times with `error`, then returns `answer`."""

    def __init__(self, failures, error, answer="ok"):
        self.remaining, self.error, self.answer, self.calls = failures, error, answer, 0

    def invoke(self, _payload):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error
        return {"messages": [type("M", (), {"content": self.answer})()]}


def test_transient_llm_errors_are_recognised():
    tool_fail = Exception(
        "Error code: 400 - {'error': {'message': \"Failed to call a function.\", "
        "'code': 'tool_use_failed'}}"
    )
    assert rag_core.is_transient_llm_error(tool_fail)
    assert rag_core.is_transient_llm_error(Exception("429 rate limit exceeded"))
    assert not rag_core.is_transient_llm_error(Exception("Invalid api_key provided"))


def test_agent_retry_recovers_from_a_flaky_tool_call():
    """Groq fails ~50% of tool calls with llama-3.3; one retry must not surface that."""
    agent = _FakeAgent(1, Exception("400 tool_use_failed"), answer="9.50 CAD")
    assert rag_core.invoke_agent_with_retry(agent, []) == "9.50 CAD"
    assert agent.calls == 2


def test_agent_retry_gives_up_after_the_attempt_budget():
    agent = _FakeAgent(99, Exception("400 tool_use_failed"))
    try:
        rag_core.invoke_agent_with_retry(agent, [], attempts=3)
        assert False, "expected the error to propagate"
    except Exception as e:
        assert "tool_use_failed" in str(e)
    assert agent.calls == 3


def test_agent_retry_does_not_retry_permanent_errors():
    """A bad key is not going to fix itself; burning three calls on it is waste."""
    agent = _FakeAgent(99, Exception("Invalid api_key provided"))
    try:
        rag_core.invoke_agent_with_retry(agent, [], attempts=3)
        assert False, "expected the error to propagate"
    except Exception as e:
        assert "api_key" in str(e)
    assert agent.calls == 1


def test_batch_ingestion_is_refused_in_demo_mode(monkeypatch):
    """Demo index is in-memory; writing to pgvector would silently do nothing useful."""
    monkeypatch.setenv("DEMO_MODE", "1")
    try:
        rag_core.ingest_directory("demo/")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "DEMO_MODE" in str(e)

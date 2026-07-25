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
    assert all(len(c.page_content) <= rag_core.CHUNK_SIZE + rag_core.CHUNK_OVERLAP for c in chunks)
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

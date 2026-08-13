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


def test_a_provision_cites_the_article_and_paragraph_not_a_page():
    """
    The page a provision lands on is a property of the typesetting, not of the law.
    A reader sent to "page 14" cannot confirm anything; Art. 17(1) they can.
    """
    doc = Document(
        page_content="...",
        metadata={
            "citation": "GDPR Art. 17(1)",
            "article_title": "Right to erasure",
            "source": "Regulation (EU) 2016/679",
            "page": 13,  # present and deliberately ignored
        },
    )
    assert rag_core.format_citation(doc) == "GDPR Art. 17(1) — Right to erasure"
    assert rag_core.cited_provision(doc) == "GDPR Art. 17(1)"


def test_an_uploaded_pdf_still_falls_back_to_the_page():
    """Uploaded documents have no structure to cite, and that path must keep working."""
    doc = Document(page_content="...", metadata={"source": "/tmp/contract.pdf", "page": 2})
    assert rag_core.format_citation(doc) == "contract.pdf — p. 3"
    assert rag_core.cited_provision(doc) is None


def test_every_chunk_of_a_split_provision_keeps_its_citation():
    """
    Splitting must not orphan text from its provision.

    A chunk whose citation was lost would be shown to the user with whatever
    citation the formatter falls back to — which for the legal corpus is the
    instrument name, i.e. a claim that the sentence is somewhere in the GDPR.
    """
    long_provision = Document(
        page_content="obligation. " * 400,
        metadata={"citation": "GDPR Art. 70(1)", "article_title": "Tasks of the Board"},
    )
    chunks = rag_core.split_documents([long_provision])
    assert len(chunks) > 1, "expected this provision to be split"
    assert all(rag_core.cited_provision(c) == "GDPR Art. 70(1)" for c in chunks)


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


def test_demo_chunk_size_is_large_enough_to_hold_a_typical_provision(monkeypatch):
    """
    The demo chunk size exists to keep provisions whole, not to be small.

    An earlier version of this test asserted the demo chunked *smaller* than
    production, which was right when the corpus was a two-page business document.
    Against the GDPR the requirement inverted: a chunk that straddles Article 33(1)
    and 33(2) gets attributed to one of them, and the citation then points at a
    paragraph the text did not come from.
    """
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.delenv("CHUNK_SIZE", raising=False)

    lengths = sorted(len(doc.page_content) for doc in rag_core.load_legal_corpus())
    median = lengths[len(lengths) // 2]
    assert rag_core.chunk_size() > median * 2, (
        f"chunk size {rag_core.chunk_size()} is too close to the median provision "
        f"({median} chars); provisions will routinely be split"
    )


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
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("RETRIEVAL_K", raising=False)

    chunks = rag_core.split_documents(rag_core.load_legal_corpus())
    assert len(chunks) > 2 * rag_core.retrieval_k(), (
        f"demo corpus splits into only {len(chunks)} chunks for k={rag_core.retrieval_k()}"
    )


def test_demo_pdf_paths_finds_the_sample_upload():
    """
    demo/ is no longer the knowledge base — corpus/gdpr_en.jsonl is. The PDF stays
    as a fixture for the uploaded-document path, which has different citation
    behaviour (file and page) and would otherwise go untested.
    """
    paths = rag_core.demo_pdf_paths()
    assert paths, "expected at least one sample PDF in demo/"
    assert all(p.lower().endswith(".pdf") for p in paths)


def test_demo_pdf_paths_is_empty_when_directory_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_core, "DEMO_DIR", str(tmp_path / "does-not-exist"))
    assert rag_core.demo_pdf_paths() == []


def test_load_legal_corpus_fails_loudly_when_it_has_not_been_built(monkeypatch, tmp_path):
    missing = str(tmp_path / "gdpr_en.jsonl")
    monkeypatch.setattr(rag_core, "LEGAL_CORPUS_PATH", missing)
    try:
        rag_core.load_legal_corpus()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        # The message has to name the command that fixes it: this is the first thing
        # a fresh clone hits, and the corpus is built, not committed by hand.
        assert "build_gdpr_corpus.py" in str(e)


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


def test_a_token_budget_error_is_retried_but_only_after_waiting():
    """
    Groq returns 413 "Request too large ... tokens per minute" when the per-minute
    budget is gone. The abstention eval hit it and died, because 413 was classified
    as permanent. Retrying it instantly would be just as useless as not retrying.
    """
    error = Exception(
        "APIStatusError: Error code: 413 - Request too large for model "
        "`openai/gpt-oss-120b` ... on tokens per minute (TPM)"
    )
    assert rag_core.is_transient_llm_error(error)
    assert rag_core.is_rate_limited(error)

    waits = []
    agent = _FakeAgent(1, error, answer="recovered")
    assert rag_core.invoke_agent_with_retry(agent, [], sleep=waits.append) == "recovered"
    assert agent.calls == 2
    assert waits and all(w > 0 for w in waits), "a token budget must be waited out"


def test_a_malformed_tool_call_is_retried_without_waiting():
    waits = []
    agent = _FakeAgent(1, Exception("400 tool_use_failed"), answer="ok")
    rag_core.invoke_agent_with_retry(agent, [], sleep=waits.append)
    assert waits == [], "a per-request glitch should retry immediately"


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

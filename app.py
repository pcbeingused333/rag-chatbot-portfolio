"""
app.py — Streamlit UI for the RAG chatbot.

Features:
  • Real-time PDF upload (ingested into the vector store on the fly)
  • ReAct agent (LangGraph) that reasons before answering
  • Citations surfaced under each answer: the provision for corpus text
    (Art. 33(1)), the file and page for an uploaded PDF

With DEMO_MODE=1 the app is fully self-contained: a small embedding model, an
in-memory FAISS index preloaded from corpus/gdpr_en.jsonl, and suggested questions so
a first-time visitor sees a working assistant without uploading anything.
"""
import os
import tempfile

import streamlit as st

st.set_page_config(page_title="RAG Assistant 2026 — Agent", page_icon="🧠", layout="wide")

# Bridge Streamlit Cloud secrets -> environment variables BEFORE importing modules
# that read os.getenv at import time (rag_core). Harmless locally: with no
# secrets.toml, st.secrets raises and we simply fall back to the .env file.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

import rag_core

DEMO_MODE = rag_core.is_demo_mode()

# Shown as one-click buttons in demo mode. The first three are answerable from the
# corpus; the fourth deliberately is not, because the behaviour worth showing a
# first-time visitor is the refusal — see the abstention eval in evals/.
DEMO_QUESTIONS = [
    "How quickly must a personal data breach be reported?",
    "On what grounds can someone demand that their data be erased?",
    "What is the maximum administrative fine?",
    "Which countries have an adequacy decision?",
]


# ==================== CACHED RESOURCES ====================
# Cache the heavy objects so the embedding model is loaded ONCE, not on every rerun.
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return rag_core.get_embeddings()


@st.cache_resource(show_spinner="Connecting to the vector store...")
def load_vectorstore():
    return rag_core.get_vectorstore(load_embeddings())


@st.cache_resource(show_spinner="Preparing the agent...")
def load_agent():
    # The prompt, the tool and the retriever all live in rag_core, so the evaluation
    # harness scores this exact agent rather than a lookalike built for testing.
    def remember_sources(docs):
        st.session_state["last_sources"] = docs

    return rag_core.build_agent(
        vectorstore=load_vectorstore(), on_documents=remember_sources
    )


def ask_agent(agent, messages) -> str:
    """Run the agent and turn any unrecoverable failure into a readable message."""
    try:
        return rag_core.invoke_agent_with_retry(agent, messages)
    except Exception as exc:  # noqa: BLE001 — the UI must never show a traceback
        if "api_key" in str(exc).lower() or "authentication" in str(exc).lower():
            return (
                "⚠️ The language model rejected the API key. If you are running this "
                "locally, check `GROQ_API_KEY` in your `.env`."
            )
        return (
            "⚠️ The language model failed to answer after several attempts. "
            "Please try rephrasing your question.\n\n"
            f"<sub>{type(exc).__name__}</sub>"
        )


# ==================== SIDEBAR: UPLOAD + CONTROLS ====================
with st.sidebar:
    st.title("📚 RAG Agent")
    st.caption(
        ("FAISS (in-memory) + Groq + LangGraph" if DEMO_MODE
         else "pgvector + Groq + LangGraph")
    )

    st.subheader("📤 Upload documents")
    if DEMO_MODE:
        st.caption(
            "Demo mode: uploads are added to an in-memory index and disappear when "
            "the app restarts. Nothing is stored on a server."
        )
    uploaded = st.file_uploader(
        "Drag & drop PDFs to add them to the knowledge base",
        type="pdf",
        accept_multiple_files=True,
    )
    if uploaded and st.button("➕ Add to knowledge base", use_container_width=True):
        with st.spinner("Processing and indexing..."):
            tmp_paths = []
            for f in uploaded:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(f.getbuffer())
                tmp.flush()
                tmp_paths.append(tmp.name)
            try:
                n = rag_core.ingest_pdf_paths(tmp_paths, load_vectorstore())
                st.success(f"✅ {len(uploaded)} PDF(s) indexed ({n} chunks).")
            finally:
                for p in tmp_paths:
                    os.unlink(p)

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pop("last_sources", None)
        st.rerun()


# ==================== MAIN CHAT ====================
st.title("🧠 RAG Assistant with Intelligent Agent")
st.caption(
    "Answers only from the indexed sources, and cites the provision behind every "
    "statement — article and paragraph, not a page number."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Ask about your documents...")

# In demo mode the knowledge base is already loaded, so a first-time visitor should
# be able to see a real answer without uploading anything.
if DEMO_MODE and not st.session_state.chat_history:
    st.info(
        "**Demo loaded:** the full text of the **GDPR** (Regulation (EU) 2016/679), "
        "indexed as 414 provisions straight from EUR-Lex, so every answer cites the "
        "article and paragraph it came from rather than a page number. Ask anything "
        "below, or try one of these — the last one is not in the Regulation, and "
        "saying so is the correct answer. You can also upload your own PDFs in the "
        "sidebar."
    )
    cols = st.columns(2)
    for i, suggested in enumerate(DEMO_QUESTIONS):
        if cols[i % 2].button(suggested, key=f"suggested_{i}", use_container_width=True):
            question = suggested

for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

if question:
    st.session_state.chat_history.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("The agent is reasoning..."):
            st.session_state["last_sources"] = []
            agent = load_agent()
            response = ask_agent(agent, st.session_state.chat_history)
            st.markdown(response)

            sources = st.session_state.get("last_sources", [])
            if sources:
                seen, citations = set(), []
                for d in sources:
                    c = rag_core.format_citation(d)
                    if c not in seen:
                        seen.add(c)
                        citations.append((c, d.page_content))
                with st.expander(f"🔍 Sources ({len(citations)})"):
                    for c, snippet in citations:
                        st.markdown(f"**{c}**")
                        st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

    st.session_state.chat_history.append(AIMessage(content=response))

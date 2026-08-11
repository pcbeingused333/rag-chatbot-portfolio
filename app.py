"""
app.py — Streamlit UI for the RAG chatbot.

Features:
  • Real-time PDF upload (ingested into the vector store on the fly)
  • ReAct agent (LangGraph) that reasons before answering
  • Real source citations with page numbers, surfaced under each answer

With DEMO_MODE=1 the app is fully self-contained: a small embedding model, an
in-memory FAISS index preloaded from demo/, and suggested questions so a first-time
visitor sees a working assistant without uploading anything.
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

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import rag_core

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
DEMO_MODE = rag_core.is_demo_mode()

# Shown as one-click buttons in demo mode; all answerable from demo/churreria_calderon.pdf.
DEMO_QUESTIONS = [
    "What are the opening hours?",
    "How much is a churros and chocolate combo?",
    "Do you have gluten-free or vegan options?",
    "What do I need to book catering for 80 people?",
]

SYSTEM_PROMPT = (
    "You are an expert assistant that answers questions about the user's PDF "
    "documents.\n"
    "RULES:\n"
    "1. For ANY question about the content of the documents, ALWAYS use the "
    "`search_documents` tool before answering.\n"
    "2. Answer DIRECTLY and COMPLETELY using the retrieved information. Include "
    "the concrete data you find (figures, names, dates, items, totals). Do NOT "
    "just say that you searched: give the answer.\n"
    "3. If the information is not in the documents, say so clearly instead of "
    "making it up.\n"
    "4. ALWAYS reply in the same language the user writes in.\n"
    "5. Be concise and helpful."
)


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
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": rag_core.retrieval_k()})

    @tool
    def search_documents(query: str) -> str:
        """Search and return relevant snippets from the user's PDF documents.
        Use this to answer any question about the content of the documents."""
        docs = retriever.invoke(query)
        st.session_state["last_sources"] = docs
        if not docs:
            return "No relevant information was found in the documents."
        return "\n\n".join(
            f"[{rag_core.format_citation(d)}]\n{d.page_content}" for d in docs
        )

    llm = ChatGroq(model=LLM_MODEL, temperature=0.2)
    return create_react_agent(llm, [search_documents], prompt=SYSTEM_PROMPT)


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
st.caption("Reasons step by step and cites the source and page of every answer.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Ask about your documents...")

# In demo mode the knowledge base is already loaded, so a first-time visitor should
# be able to see a real answer without uploading anything.
if DEMO_MODE and not st.session_state.chat_history:
    st.info(
        "**Demo loaded:** the knowledge base of *Churrería Calderón* (a sample "
        "small-business document) is already indexed. Ask anything below, or try one "
        "of these. You can also upload your own PDFs in the sidebar."
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
            result = agent.invoke({"messages": st.session_state.chat_history})
            response = result["messages"][-1].content
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

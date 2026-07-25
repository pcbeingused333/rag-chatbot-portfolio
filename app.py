"""
app.py — Streamlit UI for the RAG chatbot.

Features:
  • Real-time PDF upload (ingested into pgvector on the fly)
  • ReAct agent (LangGraph) that reasons before answering
  • Real source citations with page numbers, surfaced under each answer
"""
import os
import tempfile

import streamlit as st

st.set_page_config(page_title="Asistente RAG 2026 — Agente", page_icon="🧠", layout="wide")

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


# ==================== CACHED RESOURCES ====================
# Cache the heavy objects so the embedding model is loaded ONCE, not on every rerun.
@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_embeddings():
    return rag_core.get_embeddings()


@st.cache_resource(show_spinner="Conectando al vector store...")
def load_vectorstore():
    return rag_core.get_vectorstore(load_embeddings())


@st.cache_resource(show_spinner="Preparando el agente...")
def load_agent():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": rag_core.RETRIEVAL_K})

    @tool
    def buscar_en_documentos(query: str) -> str:
        """Busca información relevante en los documentos PDF subidos."""
        docs = retriever.invoke(query)
        # Stash the retrieved docs so the UI can render real citations after the run.
        st.session_state["last_sources"] = docs
        return "\n\n".join(
            f"[{rag_core.format_citation(d)}]\n{d.page_content}" for d in docs
        )

    llm = ChatGroq(model=LLM_MODEL, temperature=0.3)
    return create_react_agent(llm, [buscar_en_documentos])


# ==================== SIDEBAR: UPLOAD + CONTROLS ====================
with st.sidebar:
    st.title("📚 RAG Agent")
    st.caption("pgvector + Groq + LangGraph")

    st.subheader("📤 Sube documentos")
    uploaded = st.file_uploader(
        "Arrastra PDFs para añadirlos a la base de conocimiento",
        type="pdf",
        accept_multiple_files=True,
    )
    if uploaded and st.button("➕ Añadir a la base", use_container_width=True):
        with st.spinner("Procesando e indexando..."):
            tmp_paths = []
            for f in uploaded:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(f.getbuffer())
                tmp.flush()
                tmp_paths.append(tmp.name)
            try:
                n = rag_core.ingest_pdf_paths(tmp_paths, load_vectorstore())
                st.success(f"✅ {len(uploaded)} PDF(s) indexados ({n} fragmentos).")
            finally:
                for p in tmp_paths:
                    os.unlink(p)

    st.divider()
    if st.button("🗑️ Limpiar historial", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pop("last_sources", None)
        st.rerun()


# ==================== MAIN CHAT ====================
st.title("🧠 Asistente RAG con Agente Inteligente")
st.caption("Razona paso a paso y cita la fuente y la página de cada respuesta.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

if question := st.chat_input("Pregunta sobre tus documentos..."):
    st.session_state.chat_history.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("El agente está razonando..."):
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
                with st.expander(f"🔍 Fuentes ({len(citations)})"):
                    for c, snippet in citations:
                        st.markdown(f"**{c}**")
                        st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))

    st.session_state.chat_history.append(AIMessage(content=response))

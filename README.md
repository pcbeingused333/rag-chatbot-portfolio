# 🤖 RAG Chatbot — Chat with your PDFs

A Retrieval-Augmented Generation chatbot that answers questions about your PDF
documents, with **real source citations (file + page number)** and **real-time
document upload**. Built with **pgvector + LangChain + LangGraph + Groq**.

> **Live demo:** _coming soon_ — deployed on Streamlit Community Cloud (see [Deploy](#-deploy)).

![Demo screenshot](docs/screenshot.png) <!-- add a real screenshot here -->

## ✨ Features

- **Chat with your documents** — ask questions in natural language over your PDFs.
- **Real-time upload** — drop PDFs in the sidebar and they're indexed on the fly (`st.file_uploader`).
- **Real citations** — every answer lists the source file **and page number** it used.
- **Agent architecture** — a LangGraph ReAct agent decides when to search the knowledge base.
- **Vector search** — PostgreSQL + `pgvector`, `BAAI/bge-m3` multilingual embeddings.
- **Fast + free LLM** — Groq (Llama-3.3-70B).
- **Docker Compose** — one command to run app + database locally.

## 🧱 Tech Stack

| Layer            | Technology                          |
|------------------|-------------------------------------|
| Backend / agent  | LangChain, LangGraph, Groq          |
| Vector DB        | PostgreSQL + pgvector               |
| Embeddings       | Hugging Face (BAAI/bge-m3)          |
| Frontend         | Streamlit                           |
| Deployment       | Docker + Docker Compose             |

## 📂 Project structure

```
rag_core.py     Shared logic: embeddings, vector store, ingestion, citations
app.py          Streamlit UI: chat, real-time upload, citations
ingest.py       Batch-ingest a folder of PDFs
demo/           Sample knowledge base (Churrería Calderón) to try it instantly
data/           Your own private PDFs (gitignored — never committed)
tests/          Unit tests (pytest)
```

## 🚀 Quick start

### Option A — Docker (recommended)

```bash
cp .env.example .env         # add your GROQ_API_KEY
docker compose up --build -d # starts Postgres + the app
# ingest the sample knowledge base, then open http://localhost:8501
docker compose exec app python ingest.py demo
```

### Option B — Local

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # add GROQ_API_KEY + POSTGRES_CONNECTION
python ingest.py demo        # load the sample knowledge base (or: python ingest.py  for data/)
streamlit run app.py
```

Get a free Groq API key at <https://console.groq.com/keys>.

## 🧪 Try the demo

The repo ships with a sample knowledge base — `demo/churreria_calderon.pdf`, the
"company docs" of a fictional Toronto churrería. After `python ingest.py demo`, ask:

- _"What are your opening hours on Saturday?"_
- _"Do you have gluten-free options?"_
- _"How much notice do you need for catering?"_

Each answer cites the exact page it came from. Swap in your own PDFs via the sidebar
uploader or by dropping them in `data/` and running `python ingest.py`.

## ✅ Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## ☁️ Deploy

Runs on any host with a Postgres database that has the `pgvector` extension:

1. Create a free Postgres (Supabase or Neon) and enable `pgvector`.
2. Set `POSTGRES_CONNECTION` and `GROQ_API_KEY` as secrets.
3. Deploy the app to Streamlit Community Cloud (or Railway/Render).
4. Run the ingestion once against the hosted DB.

## 🔒 Note on data

`data/*.pdf` is gitignored — put your private documents there and they will **not**
be committed. Only the demo knowledge base under `demo/` ships with the repo.

## 🔮 Roadmap

- Hybrid search (vector + BM25) and reranking (FlashRank)
- LangSmith tracing
- Authentication + multi-user collections

## 📄 License

MIT — feel free to use this project for your own portfolio.

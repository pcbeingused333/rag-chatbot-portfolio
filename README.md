# 🤖 RAG Chatbot — Chat with your PDFs

A Retrieval-Augmented Generation chatbot that answers questions about your PDF
documents, with **real source citations (file + page number)** and **real-time
document upload**. Built with **pgvector + LangChain + LangGraph + Groq**.

> **Live demo:** _coming soon_ — deployed on Streamlit Community Cloud. See the
> **Deploy** section below; it needs no database and no paid tier.

## ✨ Features

- **Chat with your documents** — ask questions in natural language over your PDFs.
- **Real-time upload** — drop PDFs in the sidebar and they're indexed on the fly (`st.file_uploader`).
- **Real citations** — every answer lists the source file **and page number** it used.
- **Agent architecture** — a LangGraph ReAct agent decides when to search the knowledge base.
- **Two runtime modes** — a self-contained demo, or pgvector-backed production (below).
- **Fast + free LLM** — Groq (`gpt-oss-120b`), with retry on transient tool-call failures.
- **Docker Compose** — one command to run app + database locally.

## 🔀 Two runtime modes

The public demo and the production path have genuinely different constraints, so the
app supports both behind a single `DEMO_MODE` environment variable.

|                | `DEMO_MODE=1` (public demo)             | unset / `0` (production)        |
|----------------|-----------------------------------------|---------------------------------|
| Vector store   | FAISS, in memory                        | PostgreSQL + `pgvector`         |
| Embeddings     | `all-MiniLM-L6-v2` (~90 MB)             | `BAAI/bge-m3` (~2 GB, multilingual) |
| Knowledge base | `demo/` preloaded at startup            | ingested into Postgres          |
| External deps  | none                                    | a Postgres instance             |
| Chunking       | 300 / 50, `k=4`                         | 1000 / 200, `k=7`               |
| Query language | translated to English before retrieval  | native (embeddings are multilingual) |
| Peak RSS       | ~580 MB (fits the free 1 GB tier)       | ~2.5 GB                         |

**Why:** a free Streamlit Community Cloud container has 1 GB of RAM, so `bge-m3`
OOMs it, and a free hosted Postgres pauses when idle — which would leave the public
demo dead at exactly the moment someone opens it. Demo mode removes both
dependencies so the link always works.

The demo chunking values were measured, not guessed. Against four questions with
known answers, `300/50` puts the correct passage at rank 1 for **4/4**, versus 3/4 at
`400/80` and 2/4 at `200/30`. Production keeps the larger chunks, which suit longer
and more varied corpora.

> Note the smaller `k` in demo mode: the sample corpus splits into 13 chunks, so a
> production `k=7` would return half the document on every query and retrieval would
> stop discriminating at all. A test (`test_demo_corpus_splits_into_more_chunks_than_k`)
> guards against that regressing.

### Keeping the demo multilingual on 1 GB

Dropping `bge-m3` cost multilingual retrieval: `all-MiniLM-L6-v2` is English-only, so
_"¿Abrís los lunes?"_ retrieved nothing from an English knowledge base even though the
document says `Monday: closed`. Measured peak RSS for the alternatives:

| Embedding model                          | Peak RSS | EN  | ES  | Fits 1 GB |
|------------------------------------------|---------:|-----|-----|-----------|
| `all-MiniLM-L6-v2`                       |   580 MB | 4/4 | 3/4 | ✅        |
| `multilingual-e5-small`                  |   939 MB | 4/4 | 4/4 | ⚠️ no margin |
| `paraphrase-multilingual-MiniLM-L12-v2`  | 1 183 MB | 4/4 | 4/4 | ❌        |

No multilingual model fits safely, so the fix is at a different layer: the agent
translates the **retrieval query** into English while still answering in the user's
own language. That costs no memory and takes the demo to 4/4 in Spanish (and works
for French too). The rule is only added to the system prompt when `DEMO_MODE` is on —
the production path has multilingual embeddings and must query in the original
language.

### Surviving a flaky tool-calling model

The original default, `llama-3.3-70b-versatile`, emits malformed tool calls often
enough on Groq to break the agent on **5 of 10** requests (HTTP 400
`tool_use_failed`). `openai/gpt-oss-120b` handled 10/10 of the same set and is now the
default. As defence in depth, `invoke_agent_with_retry` retries transient LLM
failures and leaves permanent ones (a rejected API key) to fail immediately instead of
burning the whole retry budget.

## 🧱 Tech Stack

| Layer            | Technology                                  |
|------------------|---------------------------------------------|
| Backend / agent  | LangChain, LangGraph, Groq (gpt-oss-120b)   |
| Vector DB        | PostgreSQL + pgvector · FAISS (demo)        |
| Embeddings       | Hugging Face (`bge-m3` · `all-MiniLM-L6-v2`)|
| Frontend         | Streamlit                                   |
| Deployment       | Docker + Docker Compose                     |

## 📂 Project structure

```
rag_core.py     Shared logic: mode resolution, embeddings, vector stores,
                ingestion, citations — the single place the pipeline is defined
app.py          Streamlit UI: chat, real-time upload, citations, demo prompts
ingest.py       Batch-ingest a folder of PDFs into pgvector
demo/           Sample knowledge base (Churrería Calderón) + the script that builds it
data/           Your own private PDFs (gitignored — never committed)
tests/          Unit tests (pytest) — no Postgres or API key required
.streamlit/     secrets.toml.example for Streamlit Cloud
```

## 🚀 Quick start

### Option A — Demo mode (fastest, no database)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # add your GROQ_API_KEY
DEMO_MODE=1 streamlit run app.py
```

The sample knowledge base is indexed at startup — no ingestion step, no Postgres.

### Option B — Docker (recommended for the production path)

```bash
cp .env.example .env         # add your GROQ_API_KEY
docker compose up --build -d # starts Postgres + the app
# ingest the sample knowledge base, then open http://localhost:8501
docker compose exec app python ingest.py demo
```

### Option C — Local, with your own Postgres

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
"company docs" of a sample Toronto churrería.

In demo mode it is indexed automatically at startup and the UI offers one-click
questions, so a first-time visitor sees a working assistant without uploading
anything. On the production path, load it with `python ingest.py demo` first.

- _"What are the opening hours?"_
- _"How much is the churros and chocolate combo?"_
- _"Do you have gluten-free or vegan options?"_
- _"What do I need to book catering for 80 people?"_

Each answer cites the exact page it came from. Swap in your own PDFs via the sidebar
uploader, or drop them in `data/` and run `python ingest.py`.

## ✅ Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## ☁️ Deploy

### Public demo — Streamlit Community Cloud (free, no database)

1. Get a free Groq API key at <https://console.groq.com/keys>.
2. On <https://share.streamlit.io>, create an app from this repo with `app.py` as the
   main file.
3. Add two secrets under **Advanced settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   DEMO_MODE = "1"
   ```
4. Optional: add an <https://uptimerobot.com> monitor pinging the URL every ~12 h so
   the app does not go to sleep between visits.

`app.py` bridges Streamlit secrets into environment variables before importing
`rag_core`, so the same code path works locally with `.env` and in the cloud.

### Production — any host with pgvector

1. Create a Postgres (Supabase or Neon) and enable the `pgvector` extension.
2. Set `POSTGRES_CONNECTION` and `GROQ_API_KEY` as secrets. Leave `DEMO_MODE` unset.
3. Deploy the app (Streamlit Cloud, Railway, Render, or the included Dockerfile).
4. Run the ingestion once against the hosted DB: `python ingest.py`.

## 🔒 Note on data

`data/*.pdf` is gitignored — put your private documents there and they will **not**
be committed. Only the demo knowledge base under `demo/` ships with the repo.

## 🔮 Roadmap

- Hybrid search (vector + BM25) and reranking (FlashRank)
- LangSmith tracing
- Authentication + multi-user collections

## 📄 License

MIT — feel free to use this project for your own portfolio.

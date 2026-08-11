# 🤖 RAG Chatbot — Chat with your PDFs

A Retrieval-Augmented Generation chatbot that answers questions about your PDF
documents, with **real source citations (file + page number)** and **real-time
document upload**. Built with **pgvector + LangChain + LangGraph + Groq**.

> **Live demo:** <https://rag-chatbot-demo-0.streamlit.app> — no signup, no
> database, sample knowledge base already loaded.

## ✨ Features

- **Chat with your documents** — ask questions in natural language over your PDFs.
- **Real-time upload** — drop PDFs in the sidebar and they're indexed on the fly (`st.file_uploader`).
- **Real citations** — every answer lists the source file **and page number** it used.
- **Agent architecture** — a LangGraph ReAct agent decides when to search the knowledge base.
- **An evaluation harness** — retrieval, cross-lingual, memory, answer quality and
  tool-call reliability, all reproducible from the repo ([`evals/`](evals/README.md)).
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

## 📊 Measured, not assumed

Every number below is produced by a command in [`evals/`](evals/README.md), against
17 questions with known answers over the demo corpus. Reproduce any of them:

```bash
python -m evals.run_eval retrieval --sweep    # no API key needed
```

### Chunk size: the configuration was chosen by measurement

| Chunking | Chunks | hit@1 | recall@k | MRR |
|---|---:|---|---|---:|
| `200/30`, k=4 | 21 | 5/12 (42%) | 8/12 (67%) | 0.52 |
| **`300/50`, k=4 (shipped)** | **13** | **11/12 (92%)** | **12/12 (100%)** | **0.96** |
| `400/80`, k=4 | 10 | 8/12 (67%) | 10/12 (83%) | 0.74 |
| `600/100`, k=4 | 7 | 6/12 (50%) | 12/12 (100%) | 0.72 |
| `1000/200`, k=7 (production values) | 5 ⚠ ≤ k | 10/12 (83%) | 12/12 (100%) | 0.92 |

The last row is the bug that started this. With production chunking the corpus
splits into **5 chunks and `k` is 7**, so every single query returns the entire
document. recall@k reads 100% because retrieval had stopped filtering anything at
all — the system answered correctly while doing no retrieval. hit@1 still
discriminates, which is why the eval reports all three metrics.
`test_demo_corpus_splits_into_more_chunks_than_k` now guards the invariant.

Smaller chunks are not better either: at `200/30` the price list fragments across
chunk boundaries and recall drops to 8/12.

### Cross-lingual retrieval: what an English-only index costs

`all-MiniLM-L6-v2` is English-only, so _"¿Abrís los lunes?"_ retrieved nothing from
an English knowledge base even though the document says `Monday: closed`:

| Query language | hit@1 as asked | hit@1 translated to English |
|---|---|---|
| en | 11/12 | 11/12 |
| es | **0/4** | 3/4 |
| fr | **0/1** | 1/1 |

Zero. Every Spanish question. The obvious fix is a multilingual embedding model, and
measured peak RSS says it does not fit (`python -m evals.run_eval memory`):

| Embedding model | Peak RSS | Fits 1 GB |
|---|---:|---|
| `all-MiniLM-L6-v2` | 599 MB | ✅ |
| `multilingual-e5-small` | 988 MB | ⚠️ no margin |
| `paraphrase-multilingual-MiniLM-L12-v2` | 1232 MB | ❌ OOM |

So the fix moved layers instead: the agent translates the **retrieval query** into
English while still answering in the user's own language. Zero extra memory. The
rule is only added to the system prompt under `DEMO_MODE` — production embeddings
are multilingual and must query in the original language.

### Answer quality, and a refusal the eval caught

`python -m evals.run_eval answers` runs the real agent over all 17 questions and
judges what it said. It surfaced a defect no unit test could: asked _"Are you on Uber
Eats or DoorDash?"_ — an ordinary customer question — the agent decided it was
off-topic and **answered without ever calling the retrieval tool**. Adding a rule
forbidding a refusal before a search:

| Metric | Before | After |
|---|---|---|
| Grounded retrieval (deterministic) | 16/17 | **17/17** |
| Faithfulness | 0.87 | **0.96** |
| Answer relevancy | 0.94 | **1.00** |
| Answer correctness | 0.94 | **1.00** |
| Context precision | 0.34 | 0.32 |

Context precision barely moves and is not supposed to: with `k=4` and roughly one
relevant passage per question, ~0.25 is the floor. It measures whether `k` is
oversized for the corpus, not answer quality.

### Surviving a flaky tool-calling model

The agent has exactly one tool, so a model that emits a malformed tool call cannot
answer at all. Ten requests per model, **no retry** — the raw per-request rate
(`python -m evals.run_eval tool-calls`):

| Model | Answered | Malformed tool calls | Failure rate |
|---|---|---|---:|
| `openai/gpt-oss-120b` (default) | 10/10 | 0/10 | 0% |
| `llama-3.3-70b-versatile` (previous default) | 5/10 | 5/10 | **50%** |

`invoke_agent_with_retry` is defence in depth on top of that: it retries transient
failures and lets permanent ones (a rejected API key) fail immediately instead of
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
rag_core.py     Shared logic: mode resolution, embeddings, vector stores, the
                agent and its prompt — the single place the pipeline is defined
app.py          Streamlit UI: chat, real-time upload, citations, demo prompts
ingest.py       Batch-ingest a folder of PDFs into pgvector
evals/          Evaluation harness — every number in this README comes from here
demo/           Sample knowledge base (Churrería Calderón) + the script that builds it
data/           Your own private PDFs (gitignored — never committed)
tests/          Unit tests (pytest) — no Postgres or API key required
.streamlit/     secrets.toml.example for Streamlit Cloud
```

The agent lives in `rag_core.build_agent`, not in `app.py`, so the evaluation
harness scores the same prompt, tool and retriever the user talks to. An eval that
builds its own copy of the agent measures its own copy.

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

## ✅ Tests and evals

```bash
pip install -r requirements-dev.txt
pytest -q                                  # 61 tests, no Postgres or API key needed
python -m evals.run_eval retrieval --sweep # scores retrieval, also no API key
```

The two answer different questions. Tests check that the code does what it says;
evals check whether what it says is any good. See [`evals/README.md`](evals/README.md)
for what each command measures, what it costs, and where the numbers are weak.

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

- Hybrid search (vector + BM25) and reranking (FlashRank) — with the eval harness in
  place, the point is that any of these can now be shown to help or not
- Tracing, so a failing answer can be replayed rather than reasoned about
- Authentication + multi-user collections

## 📄 License

MIT — feel free to use this project for your own portfolio.

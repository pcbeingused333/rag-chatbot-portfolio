# ⚖️ Ask the GDPR — RAG with citations you can check

A retrieval-augmented assistant over the full text of the **GDPR**, built so that
every statement it makes carries the **provision it came from** — `Art. 33(1)`, not
"page 14" — and so that it says nothing when the corpus does not cover the question.

> **Live demo:** <https://rag-chatbot-demo-0.streamlit.app> — no signup, no
> database, the Regulation already indexed.

Upload your own PDFs and it works on those too. The legal corpus is what the design
is built around, and what everything below is measured against.

## Why a citation is the whole product

An assistant over legal text has an unusual failure mode. If it misses, the user
reads a vague answer, does not find what they needed, and goes to the source — the
miss announces itself. If it *invents*, the answer is fluent, confident, and
indistinguishable from a correct one. Attach a citation the model reasoned its way
to rather than read, and the invention becomes more convincing, not less: the reader
checks the reference, finds a real provision, and never notices it does not say what
the answer claimed.

Two consequences run through the whole design.

**The citation unit is the provision, not the page.** Nobody looks up page 14 of the
GDPR; which page a provision lands on is an artefact of typesetting. So the corpus is
not a PDF. `corpus/build_gdpr_corpus.py` parses the Official Journal text from
EUR-Lex into 414 provisions, each carrying its article, paragraph, title and chapter
as metadata, and those travel with every chunk into the vector store. Uploaded PDFs
have no structure to cite and still fall back to file and page.

**Declining is a correct answer.** The system prompt forbids answering from the
model's own knowledge of data protection law, and `evals/run_eval.py abstention`
measures whether that holds against questions chosen to tempt it.

## ✨ Features

- **Provision-level citations** — every answer names the article and paragraph, and
  97% of provisions survive chunking as exactly one chunk, so a citation points at
  the text it actually came from.
- **A corpus built from the authority, reproducibly** — one command rebuilds it from
  EUR-Lex, byte-identically, with sanity checks that refuse to write a bad parse.
- **Measured refusal** — an evaluation for the answers that should never be given.
- **Agent architecture** — a LangGraph ReAct agent decides when to search.
- **An evaluation harness** — retrieval, abstention, cross-lingual, memory, answer
  quality and tool-call reliability, all reproducible ([`evals/`](evals/README.md)).
- **Two runtime modes** — a self-contained demo, or pgvector-backed production.
- **Docker Compose** — one command to run app + database locally.

## 📚 The corpus

```bash
python corpus/build_gdpr_corpus.py        # fetch from EUR-Lex, parse, write JSONL
python corpus/build_gdpr_corpus.py --check # re-parse the cached HTML, offline
```

Source: Regulation (EU) 2016/679, CELEX `32016R0679`, from EUR-Lex. Not
`gdpr-info.eu` or any other mirror — in a system whose entire claim is that its
citations can be checked, the text has to come from the authority the citation names.

| | |
|---|---:|
| Provisions | 414 |
| Articles | 99 (all of them) |
| Characters | 206,003 |
| Median provision | 370 characters |

**The chunk unit is the article paragraph**, with sub-points `(a)`, `(b)`, `(c)`
inlined rather than split out. A point on its own is not a statement of law:
*"(a) the personal data are no longer necessary…"* means nothing without the chapeau
that governs it. Retrieving the point alone returns a fragment that reads like an
answer and is not one.

Article 4 is the one exception. Its 26 definitions are each independently citable —
`controller` is Article 4(7), never "Article 4" — and it is the most frequently cited
article in the instrument, so it is split per definition.

Recitals are excluded. They are interpretive aids, not binding provisions, and
citing one as though it were an obligation is a domain error, not a retrieval one.

### Three parse bugs the sanity checks now catch

The build refuses to write the corpus if any of these reappear, because each one is
invisible downstream — it degrades an answer without ever raising an error.

| Bug | What it did | Why it mattered |
|---|---|---|
| The closing article was unbounded | Article 99(2) absorbed the signatures, all 21 footnotes and a `$(document).ready` | A question about accreditation retrieved Article 99 and cited the entry-into-force rule for something it does not say |
| Nested sub-points were cut short | Articles 4(16), 4(22), 4(23) lost the text of their `(a)` limb | `'cross-border processing' means either: (a) (b) …` — a definition silently missing half its content |
| Definitions collapsed into one record | All 26 definitions cited as "Article 4" | The most-cited article in the Regulation, with its citations made useless |

The second one is worth naming. Sub-points render as one-row tables, so a regex that
scans to the next `</td>` finds the *inner* label cell and truncates. Depth counting
fixed it; a non-greedy match was simply the wrong tool for nested markup.

## 📊 Measured, not assumed

Every number below is produced by a command in [`evals/`](evals/README.md), against
25 questions whose answering provision is known, plus 7 the corpus cannot answer.
Ground truth is a **citation**, not a substring — a chunk boundary can split the
wording of Article 33(1), but it cannot change which provision the chunk came from.

```bash
python -m evals.run_eval retrieval --sweep    # no API key needed
```

### The configuration was re-chosen, because the corpus changed

`300/50, k=4` was the measured optimum for the previous corpus — a two-page business
document. Against the Regulation:

| Chunking | hit@1 | recall@k | MRR |
|---|---|---|---|
| `300/50`, k=4 — 925 chunks | 13/20 (65%) | 16/20 (80%) | 0.72 |
| `600/100`, k=4 — 554 chunks | 11/20 (55%) | 15/20 (75%) | 0.63 |
| `1000/150`, k=4 — 460 chunks | 12/20 (60%) | 17/20 (85%) | 0.71 |
| **`1500/200`, k=4 (shipped) — 433 chunks** | **13/20 (65%)** | **17/20 (85%)** | **0.74** |
| `1500/200`, k=8 — 433 chunks | 13/20 (65%) | 17/20 (85%) | 0.74 |
| `2000/200`, k=8 — 423 chunks | 13/20 (65%) | 17/20 (85%) | 0.74 |

1500 wins for a reason that is not in the table: 414 provisions produce 433 chunks,
so **97% of provisions are exactly one chunk**. A chunk straddling Article 33(1) and
33(2) gets attributed to one of them, and a citation pointing at the wrong paragraph
is the failure this design exists to prevent. Citation integrity becomes structural
rather than statistical.

`k=4` over `k=8` because they score identically and half the retrieved context per
request is not free — see the token budget below.

### The dominant failure: right article, wrong paragraph

Sibling paragraphs share vocabulary, structure and subject matter, so they sit close
together in embedding space — and the operative one is rarely the one that reads most
like the question. At the shipped configuration:

| Question | Wants | Retrieved at k=4 |
|---|---|---|
| maximum fine, serious infringements | `83(5)` | `83(4)`, `83(3)`, **`83(5)`**, `83(2)` — rank 3 |
| when a DPIA is required | `35(1)` | `35(11)`, `35(3)`, `35(7)`, `35(4)` — right article, missed |
| when a DPO is mandatory | `37(1)` | `38(5)`, `38(3)`, `37(5)`, `38(2)` — right topic, missed |

The first row is the good case and still shows the problem: the answer is grounded,
because the agent sees all four passages, but the citation surfaced first is `83(4)` —
the *lower* fine tier. The answer would be right and the reference beside it wrong,
which is the failure this project is built to avoid.

This is why hit@1 is reported next to recall@k. A system that finds the right article
is not a system that can cite. Three of the twenty questions fail this way, they fail
inside the correct article, and reranking with a cross-encoder is the obvious next
move: the distinction between "may impose a fine up to 10 000 000" and "up to
20 000 000" is precisely what a bi-encoder embedding of two similar paragraphs blurs.

### A finding that reversed when a component changed

The article title is prefixed to each provision before embedding. The obvious worry
is that it drags siblings together — every paragraph of Article 83 would begin
"Art. 83(x) — General conditions for imposing administrative fines". Measured under
`all-MiniLM-L6-v2`, that is exactly what happens: removing the heading took hit@1
from 4/20 to 8/20 at `300/50`.

Measured under `bge-small-en-v1.5`, the model that actually ships, it reverses:

| Configuration | Heading embedded | Provision text only |
|---|---|---|
| `300/50`, k=4 | **13/20, MRR 0.72** | 11/20, MRR 0.65 |
| `600/100`, k=4 | 11/20, MRR 0.63 | **12/20, MRR 0.69** |
| `1000/150`, k=4 | **12/20, MRR 0.71** | 10/20, MRR 0.61 |
| `1500/200`, k=4 | **13/20, MRR 0.74** | 12/20, MRR 0.66 |
| `1500/200`, k=8 | **13/20, MRR 0.74** | 12/20, MRR 0.68 |

Four of five configurations favour keeping it, so it ships on (`EMBED_HEADINGS=1`).

The reversal is the point. The first measurement was real, and it was a fact about a
component no longer in the system. Carrying it forward would have shipped the worse
setting on the strength of genuine evidence.

### The embedding model, re-chosen for the same reason

On a two-page business document the two models were indistinguishable. Over 414
provisions, where the competing passages are sibling paragraphs, they are not
(`1500/200`, k=4, heading embedded):

| Embedding model | hit@1 | recall@k | MRR | Peak RSS (model) | Peak RSS (model + index) |
|---|---|---|---|---:|---:|
| `all-MiniLM-L6-v2` | 8/20 | 12/20 | 0.48 | 599 MB | 834 MB |
| **`bge-small-en-v1.5` (shipped)** | **13/20** | **17/20** | **0.74** | **641 MB** | **928 MB** |
| `multilingual-e5-small` | — | — | — | 988 MB | ✗ does not fit |
| `paraphrase-multilingual-MiniLM-L12-v2` | — | — | — | 1232 MB | ✗ OOM |

42 MB for +5 questions at rank 1 is a straightforward trade. The **928 MB** column is
the honest one and it is new: the model alone understates what the container holds,
and against a free tier's 1 GB the remaining margin is about 95 MB. It fits, and it
does not fit comfortably. The `all-MiniLM-L6-v2` figure of 599 MB reproduces the
number this project measured before the corpus changed, which is what makes the two
columns comparable.

### Cross-lingual retrieval: what an English-only index costs

`bge-small-en-v1.5` is English-only, so a Spanish question retrieves nothing from an
English index even when the Regulation answers it plainly:

| Query language | hit@1 as asked | hit@1 translated to English |
|---|---|---|
| en | 13/20 | 13/20 |
| es | **0/4** | 3/4 |
| fr | **0/1** | 0/1 (recall 0/1 → 1/1) |

Zero, every Spanish question. The obvious fix is a multilingual embedding model, and
the memory table above says it does not fit. So the fix moved layers: the agent
translates the **retrieval query** into English while still answering in the user's
own language. Zero extra memory. The rule is added to the system prompt only under
`DEMO_MODE` — production embeddings are multilingual and must query natively.

The French question recovers its provision but not rank 1, which the four-question
Spanish set is too small to distinguish from noise. It is one question.

### Refusing what the corpus does not contain

`python -m evals.run_eval abstention` — 7 questions the Regulation does not answer,
each adjacent enough that the model has certainly read about it.

| Outcome | |
|---|---:|
| Abstained (correct) | **4/4 scored** |
| Answered anyway | 0 |
| Answers citing a provision never retrieved | 0 |
| Not scored — daily token quota exhausted | 3/7 |

The refusals are specific rather than blanket, which is the behaviour worth having.
Asked which countries have an adequacy decision, it retrieved Article 45(3), (4), (5)
and (8) — the mechanism, exactly the provisions the dataset predicts as adjacent —
noted that Article 45(8) says *the Commission publishes a list*, and then said the
list itself is not in the indexed text. Asked for the wording of the standard
contractual clauses, it cited Article 28(7) and 28(8) as the provisions that empower
the Commission to adopt them, and said the wording is not present.

**Three of the seven are unscored, and that is a real gap, not a rounding error.**
Groq's free tier ran out of *tokens per day* mid-run. The earlier run of the same
seven, at a different retrieval setting, scored 6/6 abstained with 0 answered and 0
fabricated citations, which is consistent but is not the same measurement. This table
should be regenerated in one clean run.

### An operational finding the eval produced by accident

The first abstention run died on its first question with a Groq `413 Request too
large … tokens per minute`. Two things were wrong. `k=8` at 1500-character chunks
made each request big enough to trip the per-minute budget — which is part of why the
shipped configuration is `k=4`, since it scores identically. And `413` was not in the
retry classifier's list of transient failures, so a rate limit that a short wait
would have cleared was treated as permanent and killed the run.

Both are fixed: `413` and its tokens-per-minute wording are classified as transient,
and rate-limit failures now back off (2s, 4s, 8s…) instead of retrying instantly,
which for a per-minute budget is just spending another request to be told the same
thing. A malformed tool call still retries immediately — the two failures deserve
different treatment, and `test_a_token_budget_error_is_retried_but_only_after_waiting`
pins that distinction.

### Not yet re-measured against this corpus

Two commands in the harness still report numbers taken against the previous corpus,
so they are **not quoted in this README**. Both need a clean run once the daily quota
resets:

| Command | Why it has to be re-run |
|---|---|
| `answers` | End-to-end quality against 25 legal questions and a rewritten system prompt. Roughly 100 API calls. |
| `tool-calls` | The raw malformed-tool-call rate per model. The tool description and system prompt both changed, and the old figure was measured under the old ones. |

Carrying those numbers forward would be the exact mistake the heading experiment
above exists to warn about.

## 🧱 Tech Stack

| Layer            | Technology                                  |
|------------------|---------------------------------------------|
| Backend / agent  | LangChain, LangGraph, Groq (gpt-oss-120b)   |
| Vector DB        | PostgreSQL + pgvector · FAISS (demo)        |
| Embeddings       | Hugging Face (`bge-m3` · `bge-small-en-v1.5`)|
| Frontend         | Streamlit                                   |
| Deployment       | Docker + Docker Compose                     |

## 🔀 Two runtime modes

The public demo and the production path have genuinely different constraints, so the
app supports both behind a single `DEMO_MODE` environment variable.

|                | `DEMO_MODE=1` (public demo)             | unset / `0` (production)        |
|----------------|-----------------------------------------|---------------------------------|
| Vector store   | FAISS, in memory                        | PostgreSQL + `pgvector`         |
| Embeddings     | `bge-small-en-v1.5` (~130 MB)           | `BAAI/bge-m3` (~2 GB, multilingual) |
| Knowledge base | `corpus/gdpr_en.jsonl` at startup       | ingested into Postgres          |
| External deps  | none                                    | a Postgres instance             |
| Chunking       | 1500 / 200, `k=4`                       | 1000 / 200, `k=7`               |
| Query language | translated to English before retrieval  | native (embeddings are multilingual) |
| Peak RSS       | ~928 MB (fits the free 1 GB tier)       | ~2.5 GB                         |

**Why:** a free Streamlit Community Cloud container has 1 GB of RAM, so `bge-m3`
OOMs it, and a free hosted Postgres pauses when idle — which would leave the public
demo dead at exactly the moment someone opens it. Demo mode removes both
dependencies so the link always works.

## 📂 Project structure

```
rag_core.py     Shared logic: mode resolution, embeddings, vector stores, citation
                formatting, the agent and its prompt — one place for the pipeline
app.py          Streamlit UI: chat, real-time upload, citations, demo prompts
corpus/         The GDPR builder and the JSONL it produces
ingest.py       Batch-ingest a folder of PDFs into pgvector
evals/          Evaluation harness — every number in this README comes from here
demo/           A sample PDF, used to exercise the uploaded-document path
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
python corpus/build_gdpr_corpus.py
DEMO_MODE=1 streamlit run app.py
```

### Option B — Docker (recommended for the production path)

```bash
cp .env.example .env         # add your GROQ_API_KEY
docker compose up --build -d # starts Postgres + the app
docker compose exec app python ingest.py gdpr
```

### Option C — Local, with your own Postgres

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # add GROQ_API_KEY + POSTGRES_CONNECTION
python ingest.py demo        # or: python ingest.py   for data/
streamlit run app.py
```

Get a free Groq API key at <https://console.groq.com/keys>.

## 🧪 Try the demo

The Regulation is indexed at startup, so a first-time visitor sees a working
assistant without uploading anything.

- _"How quickly must a personal data breach be reported?"_ → Art. 33(1)
- _"On what grounds can someone demand that their data be erased?"_ → Art. 17(1)
- _"What is the maximum administrative fine?"_ → Art. 83(5)
- _"Which countries have an adequacy decision?"_ → **not in the Regulation**, and
  saying so is the correct answer

The last one is the interesting button. Article 45 creates the adequacy mechanism and
names no country; the decisions are separate Commission acts. Every model has read
the list.

## ✅ Tests and evals

```bash
pip install -r requirements-dev.txt
pytest -q                                  # 72 tests, no Postgres or API key needed
python -m evals.run_eval retrieval --sweep # scores retrieval, also no API key
```

The two answer different questions. Tests check that the code does what it says;
evals check whether what it says is any good. See [`evals/README.md`](evals/README.md)
for what each command measures, what it costs, and where the numbers are weak.

Five of them run the app itself, headlessly, with Streamlit's `AppTest`. "The server
starts" is not the same as "the script runs": a Streamlit app boots fine and then
throws on the first browser connection, which is exactly when a visitor opens it.

One of those five is worth naming. It replaces `get_embeddings` with a function that
raises, then asserts the first render still succeeds — proving no render path touches
the model. That is the reason `DEMO_MODE` exists: the production embedding model is
~2 GB against the free tier's 1 GB, and the moment a render reaches for it, the first
visitor either waits for a download or the container is killed and they get nothing.

They run on every push and pull request ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
Streamlit Community Cloud redeploys from `main` on every push, so there is no deploy
job — and that is precisely why the suite is not decorative: it is the only thing
between a bad commit and the public demo.

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
be committed. Only the corpus and the sample PDF ship with the repo.

## ⚠️ What this is not

It is a retrieval system over legislative text. It is not a source of legal advice,
it holds one instrument in one language, and it has no case law, no guidance, no
national implementing measures and no amendments. The system prompt makes it describe
provisions rather than apply them to a situation, and three of twenty benchmark
questions still fail to put the right provision first.

## 🔮 Roadmap

- **Reranking** — the measured failure is sibling paragraphs of the right article,
  which is precisely what a cross-encoder is for
- Hybrid search (vector + BM25); legal queries carry exact terms that keyword search
  handles better than embeddings
- The other language versions of the same instrument — EUR-Lex publishes 24, which
  would replace the query-translation workaround with a real multilingual index
- Tracing, so a failing answer can be replayed rather than reasoned about

## 📄 License

MIT — feel free to use this project for your own portfolio. The GDPR text is
© European Union, reproduced from EUR-Lex; reuse of Commission documents is
authorised under Decision 2011/833/EU.

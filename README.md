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
| `all-MiniLM-L6-v2` | 8/20 | 12/20 | 0.48 | 600 MB | 685 MB |
| **`bge-small-en-v1.5` (shipped)** | **13/20** | **17/20** | **0.74** | **643 MB** | **743 MB** |
| `multilingual-e5-small` | — | — | — | 988 MB | ✗ does not fit |
| `paraphrase-multilingual-MiniLM-L12-v2` | — | — | — | 1232 MB | ✗ OOM |

43 MB for +5 questions at rank 1 is a straightforward trade. The `all-MiniLM-L6-v2`
figure of 600 MB reproduces the number this project measured before the corpus
changed, which is what makes the columns comparable.

The **model + index** column is new, and it is the one that matters: the model alone
understates what the container holds. Getting it required fixing the probe, which
built its own embeddings object and so measured a configuration that is not deployed.

That mattered more than it sounds. Encoding 32 texts per forward pass — the library
default — costs about 185 MB of transient activation memory during the cold-start
index build, against batches of 8. The vectors are identical either way: same hit@1,
same recall, same per-question ranks. Measured end to end with Streamlit loaded, the
first visitor peaked at **936 MB against a 1 GB tier**; at `EMBED_BATCH_SIZE=8` it
peaks at **775 MB**. 88 MB of headroom is not a margin, it is a coin flip on the
first person who opens the link.

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
each adjacent enough that the model has certainly read about it. Re-run 2026-09-01 and
judged by `qwen/qwen3.6-27b`; the version of this table published on 25 August had been
scored by the model under test and is superseded.

| Outcome | |
|---|---:|
| Abstained (correct) | **7/7** |
| Hedged | 0 |
| Answered anyway | 0 |
| Searched before replying | 7/7 |
| Answers citing a provision never retrieved | 0 |

The refusals are specific rather than blanket, which is the behaviour worth having.
Asked which countries have an adequacy decision, it retrieved Article 45(3), (4), (5)
and (8) — the mechanism, exactly the provisions the dataset predicts as adjacent —
noted that Article 45(8) says *the Commission publishes a list*, and then said the
list itself is not in the indexed text. Asked for the wording of the standard
contractual clauses, it cited Article 28(7) and 28(8) as the provisions that empower
the Commission to adopt them, and said the wording is not present.

All seven in one clean run, on the shipped configuration. An earlier attempt scored
only four of them: Groq's free tier ran out of *tokens per day* mid-run and the three
questions it never asked a judge were reported as failures rather than as unmeasured —
which is the defect described two sections down, and is why this table can now only be
printed for the questions that were actually scored.

**One caveat on this table, stated rather than buried.** It was judged by
`openai/gpt-oss-120b`, which is also the model under test — the defect described two
sections down, found after this run. The verdict a judge returns here is a
classification (*did this answer decline, hedge, or answer?*) rather than a quality
score it awards itself, so it is the least exposed of the judged numbers, and the
deterministic columns — searched before replying, citations never retrieved — need no
judge at all. It gets regenerated against the independent judge regardless.

### Tool-call reliability per model

`python -m evals.run_eval tool-calls` — the same 10 questions to each model with **no
retry**, so what it reports is the raw per-request rate. The agent has exactly one
tool, so a model that emits a malformed tool call cannot answer at all.

| Model | Answered | Malformed tool calls | Failure rate |
|---|---|---|---:|
| `openai/gpt-oss-120b` (shipped) | 9/10 | 0/10 | 10% |
| `openai/gpt-oss-20b` | 9/10 | 0/10 | 10% |
| `qwen/qwen3.6-27b` | 10/10 | 0/10 | 0% |

Zero malformed tool calls across 30 requests. The two failures were per-minute rate
limits, which is a property of the free tier rather than of the model — the number
this command exists to isolate is the middle column, and on these three it is a flat
zero. That is a change from what this table used to say: the second model was
`llama-3.3-70b-versatile`, which emitted malformed tool calls often enough to break
the agent on roughly half of all questions, and is the reason the retry classifier
treats `tool_use_failed` as transient at all.

**The probe was wrong about it, though.** Groq retired that model, and the run
reported it as 0/10 answered, 100% failure rate — a number that reads as *this model
cannot emit a tool call* when in fact no request ever reached a model. A missing model
is now detected on the first refusal, reported as `not available on Groq` with no
score at all, and the remaining nine requests are not spent proving it again.

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

**The same run found two more, one layer up.** The backoff above protected the agent
under test and nothing else, so the *judge* still called the API directly: the
`answers` run died at question 7 of 25 on a per-minute limit raised by a judge, and
threw away the six questions already paid for. Both judges now go through the same
policy.

And a per-day budget was being treated as a per-minute one — on both paths, in
different ways. Groq reports both through
a 429, but the remedy differs in kind: the per-minute window refills while the run
waits, whereas the per-day one answers *"try again in 13m54s"*, so backing off just
spends every attempt to be told the same thing. `DailyBudgetExhausted` is now raised
on the first sight of it, and a run that hits it **stops, keeps what it scored, says
how far it got and exits non-zero** — because the failure that matters is not the
stop, it is an eleven-question mean being read as the score for twenty-five.

The agent path needed the opposite change. It already caught everything per question
and scored it as a miss, which is right for a timeout and wrong for a spent budget: a
run against an exhausted allowance came back as twenty-five agent failures, a table
that looks like a broken agent rather than a run that never happened. That one
exception now travels up to the same handler.

### The judge stopped being independent, and nothing failed

The rule was written into the harness on day one: the model under test does not grade
its own homework. It stopped holding without a single error.

`DEFAULT_JUDGE_MODEL` was `openai/gpt-oss-120b` because the system under test was
`llama-3.3-70b-versatile` — a separate model, and a bigger one. When the shipped model
became `gpt-oss-120b`, the judge stayed where it was. From then on both constants read
the same string, three feet apart in two files, under a comment calling that exact
arrangement *the one shortcut that invalidates the whole exercise*. Every judged number
since was self-assessed.

The fix is a different family rather than a bigger sibling — independence is the
property the metric needs, not size — so the judge is now `qwen/qwen3.6-27b`, which
answered 10/10 in the probe above. `check_judge_independence` raises before a run
starts if the two ever collapse again, and a test pins the two defaults apart, because
this failure produced no symptom to notice.

**It also unblocked the budget.** Groq meters the daily token allowance *per model*: a
request to `gpt-oss-120b` was being refused for the day while `gpt-oss-20b` and
`qwen/qwen3.6-27b` answered a 3,000-token request without complaint. Judge and system
under test had been sharing one allowance, which is why the 25-question run kept dying
half-way. With the judge drawing on its own, roughly three quarters of the cost moves
off the critical budget.

### Answer quality, end to end

`python -m evals.run_eval answers` — all 25 legal questions, answered by the agent and
scored by an independent judge. Run 2026-09-01, `openai/gpt-oss-120b` under test,
`qwen/qwen3.6-27b` judging.

| Metric | Score | Scored |
|---|---:|---:|
| Grounded retrieval | **24/25** | 25/25 |
| Faithfulness | **0.97** | 25/25 |
| Answer relevancy | 1.00 | 25/25 |
| Answer correctness | 1.00 | 25/25 |
| Context precision | 0.28 | 25/25 |

The `Scored` column is not decoration. An earlier run of this table printed
`Faithfulness 1.00` and exited zero over an average of **2 of 25** questions, and
nothing on the page said so. Coverage is now printed beside every mean and a metric
scored on less than half the run exits non-zero.

Context precision at 0.28 is the honest number here: the retriever returns four
passages and typically one or two carry the answer, so three quarters of what reaches
the model is on-topic ballast. That is the cost of a small top-k over provisions that
share vocabulary, and it is the number reranking would move.

#### Two of those columns are 1.00 on every question, so they were tested

`answer_relevancy` and `answer_correctness` came back at exactly 1.00 on all 25, while
`faithfulness` varied (0.75 / 0.88 / 1.00) and `context_precision` varied (0.25 / 0.50)
— one of them from the *same* judge call. A metric that never moves is
indistinguishable from a metric that is not measuring, and no parsing test separates
them, because the parser is working perfectly in both cases.

The only thing that separates them is a negative control: hand the judge answers that
are definitely wrong and see whether the number drops.

| Answer given to a question whose reference says *"72 hours"* | relevancy | correctness |
|---|---:|---:|
| No later than 72 hours after becoming aware | 1.00 | 1.00 |
| Within **30 days** of becoming aware | 1.00 | **0.00** |
| There is **no obligation** to notify | 1.00 | **0.00** |
| The controller must appoint a data protection officer | **0.00** | 0.00 |
| *I don't know* | **0.00** | 0.00 |

The judge reads. It penalises a contradicted figure and an inverted yes/no, it drops
relevancy for an answer about something else, and it holds relevancy high for a
confidently wrong answer to the right question — which is what the prompt asks it to
do, and the reason the two metrics are separate in the first place. So the 1.00s are
real: on this set the system answers relevantly and correctly, and those two columns
currently confirm rather than discriminate. The signal lives in faithfulness and
context precision.

That control is now two tests rather than a memory, behind `EVAL_LIVE_JUDGE=1` so a
plain `pytest` never spends the daily budget. The gate is a variable of its own and
deliberately not the API key: `rag_core` calls `load_dotenv()` on import, so keying the
skip on `GROQ_API_KEY` meant the tests ran on every ordinary test run and quietly spent
the quota a full eval pass needs — a check that was not checking, which is the same
shape as everything else on this page.

### What it took to get one clean run

The numbers above are the first complete pass since the corpus was rebuilt, and the
six-week gap was not a modelling problem.

At roughly four calls a question, a full pass used to cost about the whole 200,000-token
daily allowance of Groq's free tier, because the judge and the system under test drew on
the same one. Moving the judge to a different family fixed the methodology *and* the
budget — only agent turns now come out of the shipped model's allowance — but it changed
the shape of a judge reply, and nothing checked that. The `gpt-oss` models return their
reasoning out of band, so the content was clean JSON; Qwen inlines it in a `<think>`
block, and reasoning is where a model restates the schema it was asked for, so it
contains braces. The parser's greedy `{.*}` ran from a brace inside the reasoning to the
last brace of the real answer: a span that could never parse.

What that looked like is the part worth keeping. Every abstention verdict came back
`unparsed`, so a run reported **0/6 abstained** while the agent had abstained correctly
on all six. And an `answers` run printed a full table of `n/a` under a zero exit status,
which reads as a completed run rather than a failed one. One judged metric coming back
empty is ordinary — a refusal has no claims to be unfaithful about. Every metric empty on
every question is the judge being unreadable.

Then the judge ran out of room. It spent its whole budget reasoning before emitting the
JSON, so the output was cut off, the parser correctly recorded *no claims*, and no-claims
is indistinguishable from a legitimate abstention — which is how a mean over 2 of 25
questions printed as 1.00 and exited zero. Raising `max_tokens` tripled the cost and did
not fix it; `reasoning_effort="none"` did, because claim extraction against a fixed
schema does not need deliberation.

Six defects, all in the harness rather than the system it measures, and every one of them
looked like a clean run. The guardrails they bought — coverage beside every mean, a
non-zero exit under half coverage, a refusal to start when judge and subject match, a
daily budget that raises instead of retrying, and now a live negative control — are the
reason these numbers are quoted and the earlier ones are not.

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
| Peak RSS       | ~775 MB with Streamlit (1 GB tier)      | ~2.5 GB                         |

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
pytest -q                                  # 83 tests, no Postgres or API key needed
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

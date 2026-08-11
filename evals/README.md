# Evaluation harness

Every number quoted in the project README is produced by a command in this
directory. None of them were measured by hand, and none of them are estimates.

```bash
pip install -r requirements-dev.txt

python -m evals.run_eval retrieval --sweep     # offline, no API key
python -m evals.run_eval multilingual          # offline, no API key
python -m evals.run_eval memory                # offline; downloads ~1.5 GB of weights
python -m evals.run_eval answers               # needs GROQ_API_KEY (~70 calls)
python -m evals.run_eval tool-calls            # needs GROQ_API_KEY (~20 calls)
```

Add `--json PATH` to any of them to keep the raw per-question results.

## The evaluation set

`dataset.py` holds 17 questions over `demo/churreria_calderon.pdf` — 12 English,
4 Spanish, 1 French. Each one records:

- `expect` — distinctive substrings of the passage that answers it. A retrieved
  chunk is a hit when it contains one, matched with whitespace collapsed and case
  folded, because PyPDFLoader breaks lines mid-sentence and a naive `in` check
  produces false misses.
- `reference` — a human-written correct answer, needed only for answer correctness.
- `question_en` — for non-English questions, the English form. It exists so the
  translate-the-query fix can be measured **without an LLM in the loop**: the eval
  swaps the query itself, which makes the result deterministic and free.

`test_every_expected_passage_actually_appears_in_the_demo_corpus` guards the set
against the PDF being regenerated with different wording — otherwise every question
would silently become unanswerable and the eval would report a broken retriever.

## What each command measures

### `retrieval` — is the right passage found, and is it first?

Builds a FAISS index at each chunk size and scores every question. Three metrics,
because they fail differently:

| Metric | Question it answers | What a bad score means |
|---|---|---|
| hit@1 | Is the correct passage ranked first? | The citation shown to the user is the wrong one |
| recall@k | Was it retrieved at all? | The generator never saw it — a ceiling, not a ranking problem |
| MRR | Where in the top-k, on average? | Separates "rank 2" from "rank 7", which the other two collapse |

`--sweep` compares several chunk sizes; without it, only the shipped configuration
runs. `--langs en es fr` restricts the set, `--translate` sends the English form of
non-English questions.

Configurations where the corpus splits into fewer chunks than `k` are flagged `⚠ ≤ k`.
That is not a warning about the score — it means retrieval returns the entire corpus
on every query, so recall@k is trivially perfect and the metric has stopped measuring
anything.

### `multilingual` — what an English-only index costs, and what fixes it

One index, one set of questions, two retrieval queries: as the user asked, and
translated to English. The gap between the columns is exactly what the agent's
translate-the-query instruction buys. No API call, so the result is reproducible
offline.

### `memory` — which embedding models fit the free tier

One subprocess per model, reporting peak RSS after a real `embed_query`. Separate
processes because sentence-transformers does not release model memory, so measuring
several in one process reports the high-water mark of all of them.

### `answers` — end-to-end quality, judged

Runs the **real agent** (`rag_core.build_agent`, the same one `app.py` serves), so
the agent picks its own retrieval query. A bad self-authored query is a genuine
failure mode that a direct `similarity_search` can never expose.

- `grounded` — deterministic: was the expected passage among the chunks the agent
  actually retrieved? No LLM, cannot drift. **Trust this one when a judged score
  looks surprising.**
- `faithfulness` — the answer is decomposed into claims and each is verified
  against the retrieved passages. Asking a model for a single "is this faithful?"
  score returns a vibe: a long answer with one fabricated detail still reads as
  mostly right.
- `answer_relevancy` — does it address the question, truth aside?
- `answer_correctness` — does it agree with the reference answer?
- `context_precision` — what fraction of the k passages were useful?

**Reading context precision.** With `k=4` and typically one relevant passage per
question, ~0.25 is the floor and the observed ~0.32 is close to expected. It is not
a quality score; it tells you whether `k` is larger than the corpus warrants.

### `tool-calls` — can the model use the one tool it has?

Fires N requests per model **with no retry**, so the reported number is the raw
per-request failure rate, and classifies each failure as a malformed tool call or
something else. `invoke_agent_with_retry` sits on top of this rate in production;
it does not change it.

## Why not RAGAS

RAGAS was the first choice, and the metric definitions above follow it. Every
published version — including 0.4.3, the latest — imports
`langchain_community.chat_models.vertexai`, a module removed in langchain-community
0.4. `import ragas` raises `ModuleNotFoundError` on this project as it stands.

Making it work would mean pinning the whole project to a sunset langchain-community,
degrading a production dependency to satisfy an evaluation tool. Three prompts in
`judges.py` cost less than that and are inspectable, which for these four metrics is
an improvement rather than a compromise.

## Known limitations

Worth stating, because an eval that oversells itself is worse than none:

- **The judge is a language model** and inherits its biases. It is run at
  temperature 0 and is a different call from the system under test, but it is not
  run repeatedly, so self-consistency is unmeasured.
- **The judge is strict about inference.** An answer that reasons correctly beyond
  the literal text of a passage ("the kitchen handles nuts, therefore it is not
  safe") can be scored unsupported. That is the intended reading of faithfulness,
  but it means 1.00 is not the realistic target.
- **The corpus is one short document.** The chunking numbers are tuned to it and do
  not transfer; the production path deliberately keeps different values.
- **Judged scores move slightly between runs.** The deterministic metrics —
  `grounded`, hit@1, recall@k, MRR — do not, which is why the write-ups lead with
  those.

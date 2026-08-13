# Evaluation harness

Every number quoted in the project README is produced by a command in this
directory. None of them were measured by hand, and none of them are estimates.

```bash
pip install -r requirements-dev.txt

python -m evals.run_eval retrieval --sweep     # offline, no API key
python -m evals.run_eval multilingual          # offline, no API key
python -m evals.run_eval memory                # offline; downloads ~2 GB of weights
python -m evals.run_eval abstention            # needs GROQ_API_KEY (~15 calls)
python -m evals.run_eval answers               # needs GROQ_API_KEY (~100 calls)
python -m evals.run_eval tool-calls            # needs GROQ_API_KEY (~20 calls)
```

Add `--json PATH` to any of them to keep the raw per-question results.

## The evaluation set

`dataset.py` holds two lists, and the second one is the reason this harness is
worth having.

**`QUESTIONS`** — 25 questions the GDPR does answer (20 English, 4 Spanish,
1 French). Each records:

- `expect_citations` — the provision that answers it, as a citation
  (`GDPR Art. 33(1)`). A retrieved chunk is a hit when it *came from* that
  provision. The previous version of this file matched substrings of the expected
  passage and had to warn that a phrase could straddle a chunk boundary and score a
  false miss; structural ground truth removes that class of error entirely. It is
  also stricter — retrieving text that happens to contain the right words is not the
  same as retrieving the right authority.
- `reference` — a human-written correct answer, needed only for answer correctness.
- `question_en` — for non-English questions, the English form. It exists so the
  translate-the-query fix can be measured **without an LLM in the loop**: the eval
  swaps the query itself, which makes the result deterministic and free.

**`UNANSWERABLE`** — 7 questions the GDPR does *not* answer, each with the reason
and the provisions that sit next to it without answering it. See `abstention` below.

Two tests guard the set against the corpus being rebuilt underneath it. A citation
that no longer resolves is a dataset bug, and without the guard it would surface as
a retrieval score of 0/25 — a broken retriever, apparently, rather than a broken
expectation.

## What each command measures

### `retrieval` — is the right provision found, and is it first?

Builds a FAISS index at each chunk size and scores every question. Three metrics,
because they fail differently:

| Metric | Question it answers | What a bad score means |
|---|---|---|
| hit@1 | Is the correct provision ranked first? | The citation shown to the user is the wrong one |
| recall@k | Was it retrieved at all? | The generator never saw it — a ceiling, not a ranking problem |
| MRR | Where in the top-k, on average? | Separates "rank 2" from "rank 7", which the other two collapse |

`--sweep` compares several chunk sizes; without it, only the shipped configuration
runs. `--langs en es fr` restricts the set, `--translate` sends the English form of
non-English questions.

The dominant failure mode in this corpus is **the right article, the wrong
paragraph**: asked for the maximum fine, the retriever returned Article 83(3), (4),
(2) and (6) — every paragraph of Article 83 except the one that sets the ceiling.
Sibling paragraphs share vocabulary, structure and subject matter, so they sit close
together in embedding space. This is why hit@1 is reported alongside recall: a system
that finds the right article is not the same as one that can cite.

### `multilingual` — what an English-only index costs, and what fixes it

One index, one set of questions, two retrieval queries: as the user asked, and
translated to English. The gap between the columns is exactly what the agent's
translate-the-query instruction buys. No API call, so the result is reproducible
offline.

### `memory` — which embedding models fit the free tier

One subprocess per model, reporting peak RSS after a real `embed_query`. Separate
processes because sentence-transformers does not release model memory, so measuring
several in one process reports the high-water mark of all of them.

`--with-index` also builds the demo FAISS index over the whole corpus, which is what
the container actually holds. The model on its own understates the only question that
matters, which is whether the app survives in 1 GB.

### `abstention` — does it decline what the corpus cannot answer?

Every other command here asks whether a right answer comes out. This one asks
whether a wrong one stays in, and for a corpus of legal source text that is the
question that decides whether the system is usable.

The asymmetry is the point. A retrieval miss announces itself: the user reads a
vague answer, does not find what they needed, and goes to the source. An answer
assembled from what the model happens to know about data protection law is fluent,
confident and indistinguishable from a correct one — and a citation the model
reasoned its way to rather than read is what makes it convincing. The reader checks
the reference, finds a real provision, and never notices it does not say what the
answer claimed.

So the questions are adjacent to the corpus rather than unrelated to it: adequacy
decisions by country (Article 45 builds the mechanism and names no country), the
wording of the standard contractual clauses (a separate Commission act), Schrems II
(case law), a CCPA penalty (another jurisdiction). The model has read about all of
them.

| Outcome | Meaning |
|---|---|
| `abstained` | Said the corpus does not cover it. Correct. |
| `hedged` | Discussed the topic without claiming an answer and without declining. Not wrong, not useful. |
| `answered` | Made a substantive claim the retrieved passages do not support. The failure. |
| `fabricated_citations` | Cited a provision that was never retrieved. The worse version of it. |

`fabricated_citations` is deterministic — it compares the references in the answer
text against what the retriever actually returned, with no judge involved. A bare
article reference is not counted as fabricated when a paragraph of that article was
retrieved; citing `Art. 45(3)` having read only `Art. 45(1)` is, because it asserts
a precision that was never read.

### `answers` — end-to-end quality, judged

Runs the **real agent** (`rag_core.build_agent`, the same one `app.py` serves), so
the agent picks its own retrieval query. A bad self-authored query is a genuine
failure mode that a direct `similarity_search` can never expose.

- `grounded` — deterministic: was the expected provision among the chunks the agent
  actually retrieved? No LLM, cannot drift. **Trust this one when a judged score
  looks surprising.**
- `faithfulness` — the answer is decomposed into claims and each is verified
  against the retrieved passages. Asking a model for a single "is this faithful?"
  score returns a vibe: a long answer with one fabricated detail still reads as
  mostly right.
- `answer_relevancy` — does it address the question, truth aside?
- `answer_correctness` — does it agree with the reference answer?
- `context_precision` — what fraction of the k passages were useful?

**Reading context precision.** With `k=4` and typically one relevant provision per
question, ~0.25 is the floor. It is not a quality score; it tells you whether `k` is
larger than the corpus warrants.

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

- **25 questions is a small set.** One question is four percentage points of hit@1,
  which is well inside the noise between two configurations that differ by one.
  Differences of a single question are not read as results here; the sweep is judged
  on MRR and on whether a change moves several configurations the same way.
- **The judge is a language model** and inherits its biases. It is run at
  temperature 0 and is a different call from the system under test, but it is not
  run repeatedly, so self-consistency is unmeasured.
- **The abstention questions were chosen by the author**, who also wrote the rule
  the agent is being scored against. They are adjacent to the corpus on purpose, but
  a genuinely adversarial set would be written by someone trying to break it.
- **The judge is strict about inference.** An answer that reasons correctly beyond
  the literal text of a provision can be scored unsupported. That is the intended
  reading of faithfulness, but it means 1.00 is not the realistic target.
- **Retrieval numbers are corpus-specific and component-specific.** The heading
  experiment reversed when the embedding model changed — a conclusion measured under
  one component did not survive swapping it. Nothing here should be carried to a
  different corpus without re-running it.
- **Judged scores move slightly between runs.** The deterministic metrics —
  `grounded`, `fabricated_citations`, hit@1, recall@k, MRR — do not, which is why
  the write-ups lead with those.

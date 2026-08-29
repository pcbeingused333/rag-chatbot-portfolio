"""
End-to-end answer quality: run the real agent, then judge what it said.

The retrieval eval scores the index in isolation. This one scores the whole
system — the agent decides its own search query, so a bad query is a real failure
mode that a direct similarity_search can never expose.

One deterministic metric rides along with the judged ones: `grounded`, which checks
whether the passage the dataset says holds the answer was actually among the chunks
the agent retrieved. It needs no LLM and cannot drift, which makes it the number to
trust when a judged score looks surprising.

Costs about four API calls per question (one agent turn, three judge calls).
Needs GROQ_API_KEY.
"""
import json
import os
from typing import Dict, List, Optional

import rag_core
from evals.dataset import Question, questions as load_questions
from evals.judges import (
    DEFAULT_JUDGE_MODEL,
    QualityScores,
    check_judge_independence,
    judge,
    mean,
)
from evals.metrics import provision_rank


# Enough for the reasoning plus the JSON after it. Verified against the
# claim-extraction prompt, the longest of the three: the default budget truncates
# mid-thought, 4096 closes the block and leaves parseable JSON.
JUDGE_MAX_TOKENS = 4096


def _make_judge_callable(judge_model: str):
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq

    # Temperature 0: the judge should give the same verdict on a rerun.
    #
    # max_tokens is explicit because the judge inlines its reasoning. The `gpt-oss`
    # models this harness used to judge with return reasoning out of band, so the
    # default budget only ever had to cover the JSON. Qwen spends the same budget
    # thinking first, and on the claim-extraction prompt - which asks it to
    # enumerate every claim, so it enumerates them once while reasoning and again
    # in the answer - the default cut generation off mid-thought. The reply then
    # had an unclosed `<think>`, `strip_reasoning` correctly threw the tail away,
    # and the caller recorded "no claims" for an answer full of them.
    llm = ChatGroq(model=judge_model, temperature=0, max_tokens=JUDGE_MAX_TOKENS)

    def call(prompt: str) -> str:
        # Through the retry policy, not straight at the API: a judge is as exposed to
        # the per-minute token budget as the agent is, and a 429 here throws away
        # every question already paid for in this run.
        return rag_core.call_llm_with_retry(
            lambda: llm.invoke([HumanMessage(content=prompt)]).content
        )

    return call


def _answer(agent, question: Question) -> Dict:
    """Run one question through the agent. A failure is a result, not a crash."""
    from langchain_core.messages import HumanMessage

    try:
        answer = rag_core.invoke_agent_with_retry(
            agent, [HumanMessage(content=question.question)]
        )
        return {"answer": answer, "error": None}
    except rag_core.DailyBudgetExhausted:
        # Not a failure of the agent: it never got to answer. Swallowing it here is
        # how a run comes back as twenty-five agent errors instead of one sentence
        # saying the budget is gone.
        raise
    except Exception as exc:  # noqa: BLE001 — recorded and scored as a miss
        return {"answer": "", "error": f"{type(exc).__name__}: {exc}"}


def run(
    langs: Optional[List[str]] = None,
    limit: Optional[int] = None,
    json_path: Optional[str] = None,
    model: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> int:
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
        return 2

    dataset = load_questions(langs)
    if limit:
        dataset = dataset[:limit]

    model = model or rag_core.resolve_llm_model()
    judge_model = judge_model or DEFAULT_JUDGE_MODEL
    check_judge_independence(model, judge_model)
    print(f"System under test: {model}")
    print(f"Judge: {judge_model}")
    print(f"Questions: {len(dataset)}\n")

    embeddings = rag_core.get_embeddings()
    vectorstore = rag_core.build_demo_vectorstore(embeddings)

    # The agent chooses its own retrieval query; this callback captures whatever it
    # actually pulled back, which is what the judge must score against.
    captured: List = []

    def remember(docs):
        captured.clear()
        captured.extend(docs)

    agent = rag_core.build_agent(
        vectorstore=vectorstore, on_documents=remember, model=model
    )
    call_judge = _make_judge_callable(judge_model)

    rows: List[Dict] = []
    stopped_early: Optional[str] = None
    for index, question in enumerate(dataset, start=1):
        captured.clear()
        try:
            result = _answer(agent, question)
        except rag_core.DailyBudgetExhausted as exc:
            stopped_early = str(exc)
            break
        contexts = [doc.page_content for doc in captured]
        grounded = (
            provision_rank(
                [rag_core.cited_provision(doc) for doc in captured],
                question.expect_citations,
            )
            is not None
        )

        if result["error"]:
            scores = QualityScores(None, None, None, None, note=result["error"][:200])
        else:
            try:
                scores = judge(
                    call_judge,
                    question=question.question,
                    answer=result["answer"],
                    reference=question.reference,
                    contexts=contexts,
                )
            except rag_core.DailyBudgetExhausted as exc:
                # Everything scored so far is still a measurement. Keep it, say what
                # the run did not reach, and let the caller resume tomorrow.
                stopped_early = str(exc)
                break

        rows.append(
            {
                "id": question.id,
                "lang": question.lang,
                "question": question.question,
                "answer": result["answer"],
                "error": result["error"],
                "grounded": grounded,
                "retrieved": len(contexts),
                "citations": [rag_core.format_citation(doc) for doc in captured],
                "faithfulness": scores.faithfulness,
                "answer_relevancy": scores.answer_relevancy,
                "answer_correctness": scores.answer_correctness,
                "context_precision": scores.context_precision,
                "claims": scores.claims,
                "supported_claims": scores.supported_claims,
                "note": scores.note,
            }
        )
        print(
            f"[{index}/{len(dataset)}] {question.id} ({question.lang}) — "
            f"grounded {'yes' if grounded else 'NO '}, "
            f"faith {_fmt(scores.faithfulness)}, "
            f"rel {_fmt(scores.answer_relevancy)}, "
            f"corr {_fmt(scores.answer_correctness)}"
        )

    if stopped_early:
        print(
            f"\nStopped after {len(rows)}/{len(dataset)} questions — the daily token "
            "budget is spent. The scores below cover only what was measured; the rest "
            "of the set has to be run on a fresh budget, not merged with an older run."
        )
        print(f"  {stopped_early[:200]}")

    _summarise(rows)

    if json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote {json_path}")

    thin = _thinly_scored_metrics(rows)
    if rows and thin and not _judge_produced_nothing(rows):
        listed = ", ".join(f"{m} ({n}/{len(rows)})" for m, n in thin)
        print(
            f"\nScored on too few questions to publish: {listed}. A mean over a "
            "handful of questions is not the score for the set, and printing it "
            "beside metrics that did score every question invites reading it as "
            "one. The rest of this run is fine; these numbers are not."
        )
        return 4

    if rows and _judge_produced_nothing(rows):
        print(
            "\nEvery judged metric came back empty across all "
            f"{len(rows)} questions. That is not a score of zero and it is not a "
            "hard set — it is the judge's replies not being readable, which is a "
            "harness fault. The deterministic grounded-retrieval count above still "
            "holds; nothing else in this run is publishable."
        )
        return 4

    # A partial run is not a pass: the exit code has to keep CI and a human from
    # reading an eleven-question mean as the score for twenty-five.
    return 3 if stopped_early else 0


# The judged metrics. Grounded retrieval is deliberately not here: it is computed
# without a judge, so it stays valid exactly when these do not.
_JUDGED_METRICS = ("faithfulness", "answer_relevancy", "answer_correctness", "context_precision")


def _judge_produced_nothing(rows: List[Dict]) -> bool:
    """
    True when not one judged metric was scored in the whole run.

    Worth a separate exit code because of how it looked the first time: a table of
    `n/a` printed under a zero exit status, which reads as a finished run. One
    question scoring None is ordinary — a refusal has no claims to be unfaithful
    about. Every metric on every question scoring None is the judge failing to
    parse, and the run has measured nothing it set out to measure.
    """
    return all(row.get(metric) is None for row in rows for metric in _JUDGED_METRICS)


# Below this share of questions, a judged mean stops describing the set. Set
# loosely on purpose: `faithfulness` is legitimately None when an answer makes no
# claims, so a few Nones are ordinary and only a collapse should trip this.
_MIN_COVERAGE = 0.5


def _thinly_scored_metrics(rows: List[Dict]) -> List[tuple]:
    """
    Judged metrics scored on too small a fraction of the run to report.

    The all-empty case has its own check and its own message. This one catches the
    shape that is harder to see and was actually shipped: one metric collapses,
    the other three score every question, and the table prints a mean over two
    answers in the same column as a mean over twenty-five. Nothing in the output
    says which is which, so the failure reads as a result.
    """
    thin = []
    for metric in _JUDGED_METRICS:
        scored = sum(1 for row in rows if row.get(metric) is not None)
        if 0 <= scored < _MIN_COVERAGE * len(rows):
            thin.append((metric, scored))
    return thin


def _fmt(value: Optional[float]) -> str:
    return " n/a" if value is None else f"{value:.2f}"


def _summarise(rows: List[Dict]) -> None:
    grounded = sum(1 for r in rows if r["grounded"])
    failed = sum(1 for r in rows if r["error"])

    print(f"\nGrounded (deterministic): {grounded}/{len(rows)}")
    if failed:
        print(f"Agent failures: {failed}/{len(rows)}")

    print("\n| Metric | Score | Scored |")
    print("|---|---:|---:|")
    print(f"| Grounded retrieval | {grounded}/{len(rows)} | {len(rows)}/{len(rows)} |")
    for key, label in [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer relevancy"),
        ("answer_correctness", "Answer correctness"),
        ("context_precision", "Context precision"),
    ]:
        values = [r[key] for r in rows]
        value = mean(values)
        scored = sum(1 for v in values if v is not None)
        # Coverage is printed beside the mean, always. A mean over the two
        # questions the judge managed to score reads exactly like a mean over all
        # twenty-five, and that is how a judge failure gets published as a result.
        print(
            f"| {label} | {'n/a' if value is None else f'{value:.2f}'} "
            f"| {scored}/{len(rows)} |"
        )

    weak = [r for r in rows if (r["answer_correctness"] or 1.0) < 0.7 or not r["grounded"]]
    if weak:
        print(f"\nWorth reading ({len(weak)}):")
        for row in weak:
            print(f"  {row['id']} ({row['lang']}): {row['note'] or row['answer'][:120]}")

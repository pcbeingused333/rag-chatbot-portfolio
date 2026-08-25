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
from evals.judges import QualityScores, judge, mean
from evals.metrics import provision_rank

# A separate, larger judge model: having the model under test grade its own homework
# is the one shortcut that invalidates the whole exercise.
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"


def _make_judge_callable(judge_model: str):
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq

    # Temperature 0: the judge should give the same verdict on a rerun.
    llm = ChatGroq(model=judge_model, temperature=0)

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
        result = _answer(agent, question)
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
    # A partial run is not a pass: the exit code has to keep CI and a human from
    # reading an eleven-question mean as the score for twenty-five.
    return 3 if stopped_early else 0


def _fmt(value: Optional[float]) -> str:
    return " n/a" if value is None else f"{value:.2f}"


def _summarise(rows: List[Dict]) -> None:
    grounded = sum(1 for r in rows if r["grounded"])
    failed = sum(1 for r in rows if r["error"])

    print(f"\nGrounded (deterministic): {grounded}/{len(rows)}")
    if failed:
        print(f"Agent failures: {failed}/{len(rows)}")

    print("\n| Metric | Score |")
    print("|---|---:|")
    print(f"| Grounded retrieval | {grounded}/{len(rows)} |")
    for key, label in [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer relevancy"),
        ("answer_correctness", "Answer correctness"),
        ("context_precision", "Context precision"),
    ]:
        value = mean([r[key] for r in rows])
        print(f"| {label} | {'n/a' if value is None else f'{value:.2f}'} |")

    weak = [r for r in rows if (r["answer_correctness"] or 1.0) < 0.7 or not r["grounded"]]
    if weak:
        print(f"\nWorth reading ({len(weak)}):")
        for row in weak:
            print(f"  {row['id']} ({row['lang']}): {row['note'] or row['answer'][:120]}")

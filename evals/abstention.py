"""
Abstention: does the agent decline when the corpus does not hold the answer?

Every other eval here asks whether a right answer comes out. This one asks whether
a wrong one stays in, and in a legal corpus that is the question that decides
whether the system is usable at all.

The asymmetry is the point. A retrieval miss is self-announcing: the user reads a
vague answer, does not find what they needed, and goes to the source. An answer
invented from what the model happens to know about data protection law is fluent,
confident, and indistinguishable from a correct one — and if it carries a citation
the model reasoned its way to rather than read, the citation is what makes it
convincing. The reader checks the reference, sees a real provision, and never
notices it does not say what the answer claimed.

So this eval feeds the agent questions that sound exactly like corpus questions and
are not answerable from it: adequacy decisions by country (Article 45 builds the
mechanism and names no country), the text of the standard contractual clauses (a
separate Commission act), Schrems II (case law), a CCPA penalty (another
jurisdiction). The model has read about all of them.

Three outcomes per question, scored by a judge that never sees the ground truth as
a hint, only the answer and the passages actually retrieved:

  abstained   Said the corpus does not cover it. Correct.
  hedged      Discussed the topic without claiming an answer, or answered a
              different question than the one asked. Not wrong, not useful.
  answered    Made a substantive claim the retrieved passages do not support.
              This is the failure, and `fabricated_citation` records the worse
              version of it: the answer cited a provision that was never retrieved.

Costs about two API calls per question. Needs GROQ_API_KEY.
"""
import json
import os
import re
from typing import Dict, List, Optional

import rag_core
from evals.dataset import Unanswerable, unanswerable as load_unanswerable

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"

# "Art. 17(1)", "Article 33", "Art 4(7)" — how an answer refers to a provision.
CITATION_IN_TEXT = re.compile(r"\bArt(?:icle|\.)?\s*(\d{1,2})\s*(?:\(\s*(\d{1,2})\s*\))?", re.I)

JUDGE_PROMPT = """You are grading a research assistant that answers ONLY from a \
fixed corpus of legal source text. It must never answer from its own knowledge of \
the law.

The user asked:
{question}

The assistant retrieved these passages from the corpus:
{contexts}

The assistant replied:
{answer}

Classify the reply into exactly one category:

- "abstained": it states that the corpus/documents do not contain the answer, or \
that it cannot answer from the available material. It may also explain what the \
corpus does cover. This is correct behaviour.
- "hedged": it neither claims an answer nor says the corpus lacks one. It discusses \
surrounding material, or answers a related question, without asserting the answer \
to what was asked and without declining.
- "answered": it makes a substantive factual claim that purports to answer the \
question asked.

Judge ONLY what the reply does. Do not reward or punish it for being factually \
right about the law: an answer that is true but not supported by the retrieved \
passages is still "answered".

Reply with JSON only: {{"verdict": "abstained"|"hedged"|"answered", "why": "<one short sentence>"}}"""


def _make_judge_callable(judge_model: str):
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=judge_model, temperature=0)

    def call(prompt: str) -> str:
        return llm.invoke([HumanMessage(content=prompt)]).content

    return call


def cited_in_text(answer: str) -> set:
    """Provision references the answer itself makes, normalised to 'Art. N(M)'."""
    found = set()
    for article, paragraph in CITATION_IN_TEXT.findall(answer or ""):
        found.add(f"Art. {article}({paragraph})" if paragraph else f"Art. {article}")
    return found


def retrieved_provisions(docs) -> set:
    """The same normalised form, for whatever the retriever actually returned."""
    found = set()
    for doc in docs:
        citation = rag_core.cited_provision(doc)
        if not citation:
            continue
        for article, paragraph in CITATION_IN_TEXT.findall(citation):
            found.add(f"Art. {article}({paragraph})" if paragraph else f"Art. {article}")
    return found


def fabricated_citations(answer: str, docs) -> List[str]:
    """
    Provisions the answer cites that were never retrieved.

    A bare article reference ("Article 45") is not counted as fabricated when a
    paragraph of that article was retrieved — citing the article that the retrieved
    paragraph belongs to is legitimate. The reverse is not: citing Article 45(3)
    having retrieved only Article 45(1) asserts a precision that was never read.
    """
    retrieved = retrieved_provisions(docs)
    retrieved_articles = {ref.split("(")[0] for ref in retrieved}
    invented = []
    for reference in sorted(cited_in_text(answer)):
        if reference in retrieved:
            continue
        if "(" not in reference and reference in retrieved_articles:
            continue
        invented.append(reference)
    return invented


def _parse_verdict(raw: str) -> Dict:
    """Pull the JSON object out of a judge reply, tolerating stray prose."""
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return {"verdict": "unparsed", "why": (raw or "")[:120]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "unparsed", "why": (raw or "")[:120]}
    verdict = str(data.get("verdict", "unparsed")).lower().strip()
    if verdict not in {"abstained", "hedged", "answered"}:
        verdict = "unparsed"
    return {"verdict": verdict, "why": str(data.get("why", ""))[:200]}


def _ask(agent, question: Unanswerable) -> Dict:
    from langchain_core.messages import HumanMessage

    try:
        answer = rag_core.invoke_agent_with_retry(
            agent, [HumanMessage(content=question.question)]
        )
        return {"answer": answer, "error": None}
    except Exception as exc:  # noqa: BLE001 — recorded and scored as a failure
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

    dataset = load_unanswerable(langs)
    if limit:
        dataset = dataset[:limit]

    model = model or rag_core.resolve_llm_model()
    judge_model = judge_model or DEFAULT_JUDGE_MODEL
    print(f"System under test: {model}")
    print(f"Judge: {judge_model}")
    print(f"Questions the corpus cannot answer: {len(dataset)}\n")

    vectorstore = rag_core.build_demo_vectorstore(rag_core.get_embeddings())
    captured: List = []

    def remember(docs):
        captured.clear()
        captured.extend(docs)

    agent = rag_core.build_agent(
        vectorstore=vectorstore, on_documents=remember, model=model
    )
    call_judge = _make_judge_callable(judge_model)

    rows: List[Dict] = []
    for index, question in enumerate(dataset, start=1):
        captured.clear()
        result = _ask(agent, question)
        contexts = [doc.page_content for doc in captured]

        if result["error"]:
            verdict = {"verdict": "error", "why": result["error"][:200]}
        else:
            verdict = _parse_verdict(
                call_judge(
                    JUDGE_PROMPT.format(
                        question=question.question,
                        contexts="\n\n".join(contexts) or "(nothing retrieved)",
                        answer=result["answer"],
                    )
                )
            )

        invented = fabricated_citations(result["answer"], captured)
        rows.append(
            {
                "id": question.id,
                "lang": question.lang,
                "question": question.question,
                "why_unanswerable": question.why,
                "answer": result["answer"],
                "error": result["error"],
                "verdict": verdict["verdict"],
                "why": verdict["why"],
                "searched": len(contexts) > 0,
                "retrieved": len(contexts),
                "citations": [rag_core.format_citation(doc) for doc in captured],
                "fabricated_citations": invented,
            }
        )
        print(
            f"[{index}/{len(dataset)}] {question.id} ({question.lang}) — "
            f"{verdict['verdict']}"
            + (f", INVENTED {', '.join(invented)}" if invented else "")
        )

    _summarise(rows)

    if json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote {json_path}")
    return 0


def _summarise(rows: List[Dict]) -> None:
    total = len(rows)
    counts = {key: sum(1 for r in rows if r["verdict"] == key)
              for key in ("abstained", "hedged", "answered", "error", "unparsed")}
    searched = sum(1 for r in rows if r["searched"])
    invented = [r for r in rows if r["fabricated_citations"]]

    print("\n| Outcome | Count |")
    print("|---|---:|")
    print(f"| Abstained (correct) | {counts['abstained']}/{total} |")
    print(f"| Hedged | {counts['hedged']}/{total} |")
    print(f"| Answered anyway | {counts['answered']}/{total} |")
    if counts["error"] or counts["unparsed"]:
        print(f"| Errors / unparsed | {counts['error'] + counts['unparsed']}/{total} |")
    print(f"| Searched before replying | {searched}/{total} |")
    print(f"| Answers citing a provision never retrieved | {len(invented)}/{total} |")

    worth_reading = [r for r in rows if r["verdict"] != "abstained" or r["fabricated_citations"]]
    if worth_reading:
        print(f"\nWorth reading ({len(worth_reading)}):")
        for row in worth_reading:
            detail = ", ".join(row["fabricated_citations"])
            print(
                f"  {row['id']} ({row['verdict']}): {row['why']}"
                + (f" [invented: {detail}]" if detail else "")
            )

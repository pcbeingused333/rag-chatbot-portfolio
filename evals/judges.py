"""
LLM-as-judge scoring for answer quality, with the prompts written out here.

Why not RAGAS. RAGAS is the obvious library for this and it was the first choice.
Every published version of it imports `langchain_community.chat_models.vertexai`,
a module langchain-community removed in 0.4. Making it import would mean pinning a
sunset langchain-community across the whole project — degrading a production
dependency to satisfy an evaluation tool. The metrics below follow the same
definitions RAGAS uses; they are three prompts, and they cost nothing structural.

The definitions:

  faithfulness       Of the factual claims in the answer, the fraction supported by
                     the retrieved passages. Measured by decomposing the answer into
                     claims and verifying each one separately — asking a model for a
                     single 0-1 "is this faithful?" score gets you a vibe, not a
                     measurement, because a long answer with one fabricated detail
                     still reads as mostly right.
  answer relevancy   Whether the answer actually addresses the question asked, apart
                     from whether it is true.
  context precision  Of the k retrieved passages, the fraction that were relevant.
                     Low precision means the generator is working around noise.
  answer correctness Agreement with the human-written reference answer. This is the
                     only metric here that needs ground truth, which is why the
                     dataset carries one.

Faithfulness and correctness answer different questions: an answer can be perfectly
faithful to passages that do not contain what was asked for.
"""
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> Dict:
    """
    Pull a JSON object out of a model response.

    Models wrap JSON in prose or fences no matter how firmly the prompt says not
    to, so extract the outermost braces rather than trusting the whole string.
    Raises ValueError if there is nothing parseable — a silent {} would show up as
    a score of zero and look like a bad answer instead of a bad parse.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"No JSON object in judge response: {text[:200]!r}")
    return json.loads(match.group(0))


CLAIM_EXTRACTION_PROMPT = """\
Break the ANSWER into its individual factual claims.

A claim is one verifiable statement. Split compound sentences. Ignore greetings,
hedges and offers to help — keep only statements that could be true or false.
If the answer makes no factual claims at all, return an empty list.

ANSWER:
{answer}

Reply with JSON only, in this exact shape:
{{"claims": ["claim one", "claim two"]}}"""


CLAIM_VERIFICATION_PROMPT = """\
Decide whether each CLAIM is supported by the CONTEXT.

A claim is supported only if the context states it or directly entails it. If the
context is silent on the claim, it is NOT supported, even if the claim is plausible
or true in the real world. Judge only against the context.

CONTEXT:
{context}

CLAIMS:
{claims}

Reply with JSON only, one verdict per claim, in the same order:
{{"verdicts": [{{"claim": "...", "supported": true, "why": "one short sentence"}}]}}"""


QUALITY_PROMPT = """\
Score a question-answering system's output.

QUESTION:
{question}

ANSWER GIVEN:
{answer}

REFERENCE ANSWER (written by a human, treated as correct):
{reference}

RETRIEVED PASSAGES (numbered):
{contexts}

Score two things from 0.0 to 1.0, and identify the useful passages:

- answer_relevancy: does the answer address the question that was asked? Judge only
  relevance, not truth. A confident wrong answer to the right question still scores
  high here; a true statement about something else scores low.
- answer_correctness: does the answer agree with the reference? Wording and language
  may differ — the answer may be in Spanish or French while the reference is not, and
  that is fine. Contradicting a figure, a date or a yes/no is not fine.
- relevant_passages: the numbers of the passages that contain information needed to
  answer the question. Omit passages that are merely on-topic.

Reply with JSON only:
{{"answer_relevancy": 0.0, "answer_correctness": 0.0, "relevant_passages": [1],
  "note": "one short sentence"}}"""


@dataclass(frozen=True)
class QualityScores:
    """The judged scores for one question."""

    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    answer_correctness: Optional[float]
    context_precision: Optional[float]
    claims: int = 0
    supported_claims: int = 0
    note: str = ""


def faithfulness_from_verdicts(verdicts: Sequence[Dict]) -> Optional[float]:
    """
    Fraction of claims supported by the context.

    None when the answer made no claims — a refusal ("that is not in the
    documents") has nothing to be unfaithful about, and scoring it 0.0 would drag
    the average down for behaving correctly.
    """
    if not verdicts:
        return None
    supported = sum(1 for verdict in verdicts if verdict.get("supported") is True)
    return supported / len(verdicts)


def context_precision(relevant_passages: Sequence[int], retrieved: int) -> Optional[float]:
    """Fraction of the retrieved passages the judge found useful."""
    if not retrieved:
        return None
    # Ignore out-of-range indices: a judge that hallucinates passage 9 out of 4
    # should not be able to push precision above 1.0.
    valid = {n for n in relevant_passages if isinstance(n, int) and 1 <= n <= retrieved}
    return len(valid) / retrieved


def clamp(value, low: float = 0.0, high: float = 1.0) -> Optional[float]:
    """Coerce a judge's number into range, or None if it is not a number at all."""
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return None


def format_contexts(contexts: Sequence[str]) -> str:
    """Number the passages so the judge can refer to them by index."""
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Average, skipping the Nones. None when there is nothing to average."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def judge(llm, question: str, answer: str, reference: str, contexts: List[str]) -> QualityScores:
    """
    Score one answer. Three LLM calls: extract claims, verify them, rate quality.

    Failures are contained per call, so one unparseable judge response costs that
    metric for that question rather than the whole run.
    """
    faith: Optional[float] = None
    claims: List[str] = []
    verdicts: List[Dict] = []
    context_blob = format_contexts(contexts)

    try:
        extracted = parse_json_object(llm(CLAIM_EXTRACTION_PROMPT.format(answer=answer)))
        claims = [c for c in extracted.get("claims", []) if isinstance(c, str)]
    except (ValueError, json.JSONDecodeError):
        claims = []

    if claims:
        try:
            verified = parse_json_object(
                llm(
                    CLAIM_VERIFICATION_PROMPT.format(
                        context=context_blob,
                        claims=json.dumps(claims, ensure_ascii=False, indent=2),
                    )
                )
            )
            verdicts = [v for v in verified.get("verdicts", []) if isinstance(v, dict)]
            faith = faithfulness_from_verdicts(verdicts)
        except (ValueError, json.JSONDecodeError):
            faith = None

    relevancy = correctness = precision = None
    note = ""
    try:
        quality = parse_json_object(
            llm(
                QUALITY_PROMPT.format(
                    question=question,
                    answer=answer,
                    reference=reference,
                    contexts=context_blob or "(nothing retrieved)",
                )
            )
        )
        relevancy = clamp(quality.get("answer_relevancy"))
        correctness = clamp(quality.get("answer_correctness"))
        precision = context_precision(quality.get("relevant_passages", []), len(contexts))
        note = str(quality.get("note", ""))[:200]
    except (ValueError, json.JSONDecodeError):
        pass

    return QualityScores(
        faithfulness=faith,
        answer_relevancy=relevancy,
        answer_correctness=correctness,
        context_precision=precision,
        claims=len(claims),
        supported_claims=sum(1 for v in verdicts if v.get("supported") is True),
        note=note,
    )

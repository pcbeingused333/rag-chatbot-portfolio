"""
Tool-call reliability per model.

The agent has exactly one tool, so a model that emits a malformed tool call cannot
answer at all — Groq rejects the request with HTTP 400 `tool_use_failed`. That is a
property of the model, not of the question, and it is invisible to unit tests
because unit tests do not call the API.

This command sends the same questions to each model with **no retry**, so what it
reports is the raw per-request failure rate. `invoke_agent_with_retry` in
production sits on top of that number; it does not change it.

Costs real API calls (attempts × models). Needs GROQ_API_KEY.
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import rag_core
from evals.dataset import QUESTIONS

# The incumbent and the replacement. Order matters only for reading the output.
CANDIDATE_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]


@dataclass
class ModelReport:
    model: str
    attempts: int = 0
    succeeded: int = 0
    tool_call_failures: int = 0
    other_failures: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return 0.0 if not self.attempts else 1 - self.succeeded / self.attempts

    @property
    def summary(self) -> str:
        return (
            f"{self.succeeded}/{self.attempts} answered "
            f"({self.tool_call_failures} malformed tool calls, "
            f"{self.other_failures} other failures)"
        )


def _probe(model: str, attempts: int, vectorstore) -> ModelReport:
    from langchain_core.messages import HumanMessage

    report = ModelReport(model=model)
    agent = rag_core.build_agent(vectorstore=vectorstore, model=model)

    # Cycle the eval questions so a model is not judged on one lucky prompt.
    for index in range(attempts):
        question = QUESTIONS[index % len(QUESTIONS)]
        report.attempts += 1
        try:
            # Deliberately not invoke_agent_with_retry: this measures the raw rate.
            agent.invoke({"messages": [HumanMessage(content=question.question)]})
            report.succeeded += 1
        except Exception as exc:  # noqa: BLE001 — classifying failures is the point
            message = str(exc)
            if "tool_use_failed" in message.lower() or "failed to call a function" in message.lower():
                report.tool_call_failures += 1
            else:
                report.other_failures += 1
            report.errors.append(f"{question.id}: {type(exc).__name__}: {message[:200]}")
        print(f"  {model} [{report.attempts}/{attempts}] {report.summary}", end="\r")
    print()
    return report


def run(
    models: Optional[List[str]] = None,
    attempts: int = 10,
    json_path: Optional[str] = None,
) -> int:
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
        return 2

    models = models or CANDIDATE_MODELS
    print(f"Sending {attempts} requests to each of: {', '.join(models)}")
    print("No retry — this is the raw per-request failure rate.\n")

    # Build the index once and share it: the variable under test is the model.
    embeddings = rag_core.get_embeddings()
    vectorstore = rag_core.build_demo_vectorstore(embeddings)

    reports = [_probe(model, attempts, vectorstore) for model in models]

    print("\n| Model | Answered | Malformed tool calls | Failure rate |")
    print("|---|---|---|---:|")
    for report in reports:
        print(
            f"| `{report.model}` | {report.succeeded}/{report.attempts} "
            f"| {report.tool_call_failures}/{report.attempts} "
            f"| {report.failure_rate:.0%} |"
        )

    if json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump([vars(r) for r in reports], handle, indent=2, ensure_ascii=False)
        print(f"\nWrote {json_path}")
    return 0

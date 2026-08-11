"""
Command line for the evaluation harness.

    python -m evals.run_eval retrieval      # chunking sweep — offline, no API key
    python -m evals.run_eval multilingual   # query-translation fix — offline
    python -m evals.run_eval memory         # peak RSS per embedding model
    python -m evals.run_eval answers        # RAGAS answer quality — needs GROQ_API_KEY
    python -m evals.run_eval tool-calls     # LLM tool-call reliability — needs GROQ_API_KEY

The first two need nothing but the repo and a CPU: they load a ~90 MB embedding
model, build a FAISS index in memory and score fixed questions with known answers.
That is what makes the README's retrieval numbers reproducible by a stranger.
"""
import argparse
import json
import subprocess
import sys
from typing import List, Optional, Sequence

from evals import retrieval as retrieval_mod
from evals.dataset import questions as load_questions
from evals.metrics import RetrievalReport, markdown_table
from evals.retrieval import SWEEP

# The demo ships an English-only model; these are the alternatives that were
# considered for multilingual support and rejected on memory grounds.
EMBEDDING_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

FREE_TIER_MB = 1024


def _print_misses(report: RetrievalReport, limit: int = 5) -> None:
    """Show what was missed. An aggregate score alone does not tell you what to fix."""
    misses = report.misses
    if not misses:
        return
    print(f"  missed ({len(misses)}): ", end="")
    print(", ".join(r.id for r in misses[:limit]), end="")
    print(", …" if len(misses) > limit else "")


def cmd_retrieval(args) -> int:
    questions = load_questions(args.langs)
    configs = SWEEP if args.sweep else [retrieval_mod.DEMO_DEFAULT]

    print(f"Corpus: {', '.join(p.split('/')[-1] for p in _demo_paths())}")
    print(f"Questions: {len(questions)} ({', '.join(sorted({q.lang for q in questions}))})")
    print(f"Retrieval query: {'translated to English' if args.translate else 'as asked'}\n")

    embeddings = _embeddings(args.embedding_model)
    docs = retrieval_mod.load_demo_documents()

    reports: List[RetrievalReport] = []
    for config in configs:
        index, chunk_count = retrieval_mod.build_index(docs, config, embeddings)
        label = f"{config.label} — {chunk_count} chunks"
        if chunk_count <= config.k:
            # The failure that started all of this: fewer chunks than k means every
            # query returns the whole corpus and retrieval stops discriminating.
            label += " ⚠ ≤ k"
        report = retrieval_mod.evaluate(
            index, questions, config, translate=args.translate, label=label
        )
        reports.append(report)
        print(
            f"{label}: hit@1 {report.fraction(report.hits_at_1)}, "
            f"recall@k {report.fraction(report.recalled)}, MRR {report.mrr:.2f}"
        )
        _print_misses(report)

    print("\n" + markdown_table(reports, label_header="Chunking"))
    if args.json:
        _write_json(args.json, [_report_dict(r) for r in reports])
    return 0


def cmd_multilingual(args) -> int:
    """
    Measure the cost of an English-only index, and the fix for it.

    Same index, same questions, two retrieval queries: the question as asked, and
    the question translated to English. The gap between the two columns is exactly
    what the agent's translate-the-query instruction buys.
    """
    embeddings = _embeddings(args.embedding_model)
    docs = retrieval_mod.load_demo_documents()
    config = retrieval_mod.DEMO_DEFAULT
    index, chunk_count = retrieval_mod.build_index(docs, config, embeddings)

    print(f"Embedding model: {args.embedding_model or 'demo default (all-MiniLM-L6-v2)'}")
    print(f"Index: {chunk_count} chunks at {config.label}\n")

    rows = []
    for lang in ("en", "es", "fr"):
        questions = load_questions([lang])
        if not questions:
            continue
        as_asked = retrieval_mod.evaluate(index, questions, config, translate=False)
        translated = retrieval_mod.evaluate(index, questions, config, translate=True)
        rows.append((lang, as_asked, translated))
        print(
            f"{lang}: as asked hit@1 {as_asked.fraction(as_asked.hits_at_1)} "
            f"(recall {as_asked.fraction(as_asked.recalled)}) → "
            f"translated hit@1 {translated.fraction(translated.hits_at_1)} "
            f"(recall {translated.fraction(translated.recalled)})"
        )

    print("\n| Query language | hit@1 as asked | hit@1 translated to English |")
    print("|---|---|---|")
    for lang, as_asked, translated in rows:
        print(
            f"| {lang} | {as_asked.fraction(as_asked.hits_at_1)} "
            f"| {translated.fraction(translated.hits_at_1)} |"
        )

    if args.json:
        _write_json(
            args.json,
            [
                {
                    "lang": lang,
                    "as_asked": _report_dict(a),
                    "translated": _report_dict(t),
                }
                for lang, a, t in rows
            ],
        )
    return 0


def cmd_memory(args) -> int:
    """
    Peak RSS for each candidate embedding model, one subprocess each.

    Downloads up to ~1.5 GB of model weights the first time. That is why it is a
    separate command and not part of the default run.
    """
    models = args.models or EMBEDDING_CANDIDATES
    print(f"Free-tier budget: {FREE_TIER_MB} MB\n")

    rows = []
    for model in models:
        proc = subprocess.run(
            [sys.executable, "-m", "evals.memory_probe", model],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"{model}: FAILED\n{proc.stderr.strip()[:500]}", file=sys.stderr)
            continue
        measurement = json.loads(proc.stdout.strip().splitlines()[-1])
        peak = measurement["peak_rss_mb"]
        verdict = "fits" if peak < FREE_TIER_MB * 0.85 else (
            "no margin" if peak < FREE_TIER_MB else "OOM"
        )
        rows.append({**measurement, "verdict": verdict})
        print(f"{model}: {peak:.0f} MB peak RSS — {verdict}")

    print("\n| Embedding model | Peak RSS | Fits 1 GB |")
    print("|---|---:|---|")
    for row in rows:
        print(f"| `{row['model']}` | {row['peak_rss_mb']:.0f} MB | {row['verdict']} |")

    if args.json:
        _write_json(args.json, rows)
    return 0


def cmd_answers(args) -> int:
    from evals.answers import run as run_answers

    return run_answers(
        langs=args.langs,
        limit=args.limit,
        json_path=args.json,
        model=args.model,
        judge_model=args.judge_model,
    )


def cmd_tool_calls(args) -> int:
    from evals.tool_calls import run as run_tool_calls

    return run_tool_calls(
        models=args.models,
        attempts=args.attempts,
        json_path=args.json,
    )


# ---- helpers ----


def _demo_paths() -> Sequence[str]:
    import rag_core

    return rag_core.demo_pdf_paths()


def _embeddings(model_name: Optional[str]):
    from langchain_huggingface import HuggingFaceEmbeddings

    import rag_core

    # The eval always measures the demo path, so default to the demo model rather
    # than to whatever DEMO_MODE happens to be in the caller's shell.
    return HuggingFaceEmbeddings(model_name=model_name or rag_core.DEMO_EMBEDDING_MODEL)


def _report_dict(report: RetrievalReport) -> dict:
    return {
        "label": report.label,
        "total": report.total,
        "hit_at_1": report.hits_at_1,
        "recall_at_k": report.recalled,
        "mrr": round(report.mrr, 4),
        "results": [
            {"id": r.id, "lang": r.lang, "query": r.query, "rank": r.rank}
            for r in report.results
        ],
    }


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_eval",
        description="Evaluation harness for the RAG pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--json", metavar="PATH", help="also write results as JSON")
        return sub

    retrieval = add_common(
        subparsers.add_parser("retrieval", help="score retrieval, optionally sweeping chunk sizes")
    )
    retrieval.add_argument("--sweep", action="store_true", help="compare several chunk sizes")
    retrieval.add_argument("--langs", nargs="*", help="restrict to en/es/fr")
    retrieval.add_argument(
        "--translate",
        action="store_true",
        help="send the English form of non-English questions (what the agent does in demo mode)",
    )
    retrieval.add_argument("--embedding-model", help="override the embedding model")
    retrieval.set_defaults(func=cmd_retrieval)

    multilingual = add_common(
        subparsers.add_parser("multilingual", help="measure the query-translation fix")
    )
    multilingual.add_argument("--embedding-model", help="override the embedding model")
    multilingual.set_defaults(func=cmd_multilingual)

    memory = add_common(
        subparsers.add_parser("memory", help="peak RSS per embedding model (downloads weights)")
    )
    memory.add_argument("--models", nargs="*", help="embedding models to measure")
    memory.set_defaults(func=cmd_memory)

    answers = add_common(
        subparsers.add_parser("answers", help="RAGAS answer quality (needs GROQ_API_KEY)")
    )
    answers.add_argument("--langs", nargs="*", help="restrict to en/es/fr")
    answers.add_argument("--limit", type=int, help="only the first N questions")
    answers.add_argument("--model", help="override the LLM under test")
    answers.add_argument("--judge-model", help="override the judging LLM")
    answers.set_defaults(func=cmd_answers)

    tool_calls = add_common(
        subparsers.add_parser(
            "tool-calls", help="tool-call reliability per LLM (needs GROQ_API_KEY)"
        )
    )
    tool_calls.add_argument("--models", nargs="*", help="LLMs to compare")
    tool_calls.add_argument("--attempts", type=int, default=10, help="requests per model")
    tool_calls.set_defaults(func=cmd_tool_calls)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""
Peak-RSS probe for one embedding model. Runs as a subprocess, on purpose.

Loading a sentence-transformers model does not release its memory when the object
goes out of scope, so measuring several models inside one process reports the high
water mark of everything loaded so far. Each model therefore gets its own process,
and the parent reads the number back off stdout.

Usage (normally invoked by `python -m evals.run_eval memory`):

    python -m evals.memory_probe sentence-transformers/all-MiniLM-L6-v2
"""
import json
import resource
import sys


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: python -m evals.memory_probe <embedding-model>", file=sys.stderr)
        return 2

    model_name = argv[1]
    baseline = peak_rss_mb()

    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # Embed once: the model is lazy in places, and an unused model understates the
    # memory a request actually costs.
    embeddings.embed_query("How much is the churros and chocolate combo?")

    print(
        json.dumps(
            {
                "model": model_name,
                "peak_rss_mb": round(peak_rss_mb(), 1),
                "baseline_rss_mb": round(baseline, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

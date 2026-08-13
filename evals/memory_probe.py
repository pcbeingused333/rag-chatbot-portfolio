"""
Peak-RSS probe for one embedding model. Runs as a subprocess, on purpose.

Loading a sentence-transformers model does not release its memory when the object
goes out of scope, so measuring several models inside one process reports the high
water mark of everything loaded so far. Each model therefore gets its own process,
and the parent reads the number back off stdout.

With --with-index the probe also builds the demo FAISS index over the whole corpus,
which is what the container actually holds: the model on its own understates the
answer to the only question that matters, which is whether the app survives in 1 GB.

Usage (normally invoked by `python -m evals.run_eval memory`):

    python -m evals.memory_probe sentence-transformers/all-MiniLM-L6-v2
    python -m evals.memory_probe BAAI/bge-small-en-v1.5 --with-index
"""
import json
import resource
import sys


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv) -> int:
    if not 2 <= len(argv) <= 3:
        print(
            "usage: python -m evals.memory_probe <embedding-model> [--with-index]",
            file=sys.stderr,
        )
        return 2

    model_name = argv[1]
    with_index = len(argv) == 3 and argv[2] == "--with-index"
    baseline = peak_rss_mb()

    from langchain_huggingface import HuggingFaceEmbeddings

    import rag_core

    # Same encode_kwargs the app ships with. Building the probe's own embeddings
    # object without them measured a configuration that is not deployed — and since
    # batch size drives the transient activation memory of the index build, that is
    # exactly the part of the number this probe exists to capture.
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"batch_size": rag_core.EMBED_BATCH_SIZE},
    )
    # Embed once: the model is lazy in places, and an unused model understates the
    # memory a request actually costs.
    embeddings.embed_query("How quickly must a personal data breach be reported?")

    if with_index:
        import rag_core

        rag_core.build_demo_vectorstore(embeddings)

    print(
        json.dumps(
            {
                "model": model_name,
                "with_index": with_index,
                "peak_rss_mb": round(peak_rss_mb(), 1),
                "baseline_rss_mb": round(baseline, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

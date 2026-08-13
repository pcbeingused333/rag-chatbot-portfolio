"""
ingest.py — Batch-load a knowledge base into Postgres + pgvector.

Usage:
    python ingest.py            # ingests data/  (your own PDFs)
    python ingest.py gdpr       # ingests corpus/gdpr_en.jsonl (the legal corpus)
    python ingest.py demo       # ingests demo/  (the sample PDF)

Rebuilds the collection from scratch each run.

The `gdpr` target does not go through the PDF loader. The corpus is already parsed
into provisions carrying their article and paragraph, and rendering it to a PDF to
read it back would discard exactly the structure the citations depend on.
"""
import sys

from rag_core import ingest_directory, ingest_legal_corpus

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/"

    if target == "gdpr":
        print("⚖️  Cargando el corpus legal (corpus/gdpr_en.jsonl) ...")
        n_chunks = ingest_legal_corpus(pre_delete=True)
        print(f"✅ ¡Éxito! {n_chunks} chunks cargados en PostgreSQL con pgvector.")
    else:
        print(f"📄 Cargando PDFs desde {target} ...")
        n_chunks = ingest_directory(target, pre_delete=True)
        if n_chunks == 0:
            print(f"⚠️  No se encontraron PDFs en {target}. Añade documentos y vuelve a correr.")
        else:
            print(f"✅ ¡Éxito! {n_chunks} chunks cargados en PostgreSQL con pgvector.")

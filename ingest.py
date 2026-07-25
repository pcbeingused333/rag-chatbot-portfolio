"""
ingest.py — Batch-ingest every PDF in a folder into Postgres + pgvector.

Usage:
    python ingest.py            # ingests data/  (your own documents)
    python ingest.py demo       # ingests demo/  (the sample knowledge base)

Rebuilds the collection from scratch each run.
"""
import sys

from rag_core import ingest_directory

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/"
    print(f"📄 Cargando PDFs desde {directory} ...")
    n_chunks = ingest_directory(directory, pre_delete=True)
    if n_chunks == 0:
        print(f"⚠️  No se encontraron PDFs en {directory}. Añade documentos y vuelve a correr.")
    else:
        print(f"✅ ¡Éxito! {n_chunks} chunks cargados en PostgreSQL con pgvector.")

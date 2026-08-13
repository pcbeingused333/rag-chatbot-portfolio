"""
Retrieval evaluation: build the index under a given configuration, then score it.

This deliberately drives the *production* code path — `rag_core.split_documents`
under temporarily-set environment variables — rather than reimplementing the
splitter locally. An eval that measures its own copy of the pipeline measures
nothing.
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import rag_core
from evals.dataset import Question
from evals.metrics import QuestionResult, RetrievalReport, provision_rank


@dataclass(frozen=True)
class ChunkingConfig:
    """One point in the chunking sweep."""

    chunk_size: int
    chunk_overlap: int
    k: int

    @property
    def label(self) -> str:
        return f"`{self.chunk_size}/{self.chunk_overlap}`, k={self.k}"


# The corpus arrives pre-split into 414 provisions with a median length of ~370
# characters, so the question this sweep answers is how much of a provision the
# splitter is allowed to cut in half. 300 was measured as optimal for the previous
# corpus — a two-page business document — and is kept as the baseline it has to beat,
# because "the setting we already had" is the honest thing to compare against.
SWEEP: List[ChunkingConfig] = [
    ChunkingConfig(300, 50, 4),
    ChunkingConfig(600, 100, 4),
    ChunkingConfig(1000, 150, 4),
    ChunkingConfig(1500, 200, 4),
    ChunkingConfig(1500, 200, 8),
    ChunkingConfig(2000, 200, 8),
]

DEMO_DEFAULT = ChunkingConfig(
    rag_core.DEMO_CHUNK_SIZE, rag_core.DEMO_CHUNK_OVERLAP, rag_core.DEMO_RETRIEVAL_K
)


@contextmanager
def chunking(config: ChunkingConfig) -> Iterator[None]:
    """Apply a chunking configuration to the production code, then restore."""
    overrides: Dict[str, str] = {
        "CHUNK_SIZE": str(config.chunk_size),
        "CHUNK_OVERLAP": str(config.chunk_overlap),
        "RETRIEVAL_K": str(config.k),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_demo_documents() -> List:
    """The legal corpus, loaded once and reused across every configuration."""
    return rag_core.load_legal_corpus()


def build_index(docs: Sequence, config: ChunkingConfig, embeddings):
    """FAISS index over the corpus at this chunk size. Returns (index, chunk_count)."""
    from langchain_community.vectorstores import FAISS

    with chunking(config):
        chunks = rag_core.split_documents(list(docs))
    return FAISS.from_documents(chunks, embeddings), len(chunks)


def evaluate(
    index,
    questions: Sequence[Question],
    config: ChunkingConfig,
    translate: bool = False,
    label: Optional[str] = None,
) -> RetrievalReport:
    """Score every question against an already-built index."""
    results: List[QuestionResult] = []
    for question in questions:
        query = question.retrieval_query(translate=translate)
        retrieved = index.similarity_search(query, k=config.k)
        results.append(
            QuestionResult(
                id=question.id,
                lang=question.lang,
                query=query,
                rank=provision_rank(
                    [rag_core.cited_provision(doc) for doc in retrieved],
                    question.expect_citations,
                ),
                citations=[rag_core.format_citation(doc) for doc in retrieved],
            )
        )
    return RetrievalReport(label=label or config.label, results=results)

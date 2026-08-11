"""
Scoring for retrieval, kept free of LangChain and of any network call.

Everything here is a pure function over strings, so the metrics are unit-testable
and the numbers are deterministic: the same corpus and the same configuration
produce the same table every time.

Why these three metrics:

  hit@1     The demo shows one answer with one citation. If the right passage is
            not first, the citation shown to the user is the wrong one, even when
            the answer happens to read correctly.
  recall@k  Whether the passage was retrieved at all. This is the ceiling — the
            generator cannot ground an answer in a passage it never saw.
  MRR       Where in the top-k it landed, averaged. It separates "rank 2" from
            "rank 7", which the other two collapse together.

A gap between recall@k and hit@1 is a ranking problem; a low recall@k is a
chunking or embedding problem. They point at different fixes, which is the whole
reason for measuring both.
"""
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse whitespace and case-fold, so PDF line breaks do not cause misses."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def chunk_contains(chunk_text: str, expect: Iterable[str]) -> bool:
    """Whether a retrieved chunk holds any of the expected passage markers."""
    haystack = normalize(chunk_text)
    return any(normalize(needle) in haystack for needle in expect)


def hit_rank(chunk_texts: Sequence[str], expect: Iterable[str]) -> Optional[int]:
    """
    1-indexed position of the first chunk containing the expected passage.

    None when no retrieved chunk contains it — the passage was missed entirely.
    """
    expect = list(expect)
    for position, text in enumerate(chunk_texts, start=1):
        if chunk_contains(text, expect):
            return position
    return None


@dataclass(frozen=True)
class QuestionResult:
    """One question's outcome under one configuration."""

    id: str
    lang: str
    query: str
    rank: Optional[int]
    citations: List[str]

    @property
    def hit_at_1(self) -> bool:
        return self.rank == 1

    @property
    def retrieved(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.rank is None else 1.0 / self.rank


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate scores for a run, plus the per-question detail behind them."""

    label: str
    results: List[QuestionResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def hits_at_1(self) -> int:
        return sum(1 for r in self.results if r.hit_at_1)

    @property
    def recalled(self) -> int:
        return sum(1 for r in self.results if r.retrieved)

    @property
    def hit_at_1_rate(self) -> float:
        return self.hits_at_1 / self.total if self.total else 0.0

    @property
    def recall_at_k(self) -> float:
        return self.recalled / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        if not self.total:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / self.total

    @property
    def misses(self) -> List[QuestionResult]:
        """Questions whose passage never made the top-k — the ones worth reading."""
        return [r for r in self.results if not r.retrieved]

    def fraction(self, hits: int) -> str:
        return f"{hits}/{self.total}"


def markdown_table(reports: Sequence[RetrievalReport], label_header: str = "Configuration") -> str:
    """Render reports as a Markdown table, ready to paste into the README."""
    lines = [
        f"| {label_header} | hit@1 | recall@k | MRR |",
        "|---|---|---|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report.label} "
            f"| {report.fraction(report.hits_at_1)} ({report.hit_at_1_rate:.0%}) "
            f"| {report.fraction(report.recalled)} ({report.recall_at_k:.0%}) "
            f"| {report.mrr:.2f} |"
        )
    return "\n".join(lines)

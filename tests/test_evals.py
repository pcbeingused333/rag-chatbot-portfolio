"""
Tests for the evaluation harness itself.

An eval that is wrong is worse than no eval: it produces numbers that look
authoritative and are not. Everything scoreable without a network call is tested
here — the substring matching, the aggregates, the judge parsing.
"""
import pytest

from evals import judges
from evals.dataset import QUESTIONS, questions
from evals.metrics import (
    QuestionResult,
    RetrievalReport,
    chunk_contains,
    hit_rank,
    markdown_table,
    normalize,
)


# ---- passage matching ----

def test_normalize_collapses_the_line_breaks_pdfs_introduce():
    assert normalize("Monday:\n  closed.\tTuesday") == "monday: closed. tuesday"


def test_chunk_contains_survives_a_line_break_inside_the_expected_phrase():
    # PyPDFLoader wraps mid-phrase; without normalisation this is a false miss.
    chunk = "Opening Hours\nMonday: closed. Tuesday to Thursday: 9:00 AM"
    assert chunk_contains(chunk, ["Monday: closed"])


def test_chunk_contains_is_case_insensitive():
    assert chunk_contains("APPLE PAY accepted", ["Apple Pay"])


def test_chunk_contains_matches_any_of_several_markers():
    # A passage can appear in two places in the corpus; either counts.
    assert chunk_contains("we are closed on Mondays", ["Monday: closed", "closed on Mondays"])


def test_chunk_contains_rejects_an_unrelated_chunk():
    assert not chunk_contains("We accept cash and debit", ["Monday: closed"])


def test_hit_rank_is_one_indexed():
    chunks = ["irrelevant", "Monday: closed", "also irrelevant"]
    assert hit_rank(chunks, ["Monday: closed"]) == 2


def test_hit_rank_returns_none_when_the_passage_was_never_retrieved():
    assert hit_rank(["nothing", "here"], ["Monday: closed"]) is None


def test_hit_rank_reports_the_first_of_several_matches():
    chunks = ["Monday: closed", "closed on Mondays"]
    assert hit_rank(chunks, ["Monday: closed", "closed on Mondays"]) == 1


# ---- aggregates ----

def _result(id_, rank):
    return QuestionResult(id=id_, lang="en", query="q", rank=rank, citations=[])


def test_report_separates_ranking_failures_from_retrieval_failures():
    # Two of three retrieved but not first, one missed entirely: recall@k must not
    # be read as hit@1, because they point at different fixes.
    report = RetrievalReport("cfg", [_result("a", 1), _result("b", 3), _result("c", None)])
    assert report.hits_at_1 == 1
    assert report.recalled == 2
    assert report.hit_at_1_rate == pytest.approx(1 / 3)
    assert report.recall_at_k == pytest.approx(2 / 3)


def test_mrr_weights_by_position():
    report = RetrievalReport("cfg", [_result("a", 1), _result("b", 2)])
    assert report.mrr == pytest.approx((1.0 + 0.5) / 2)


def test_a_missed_question_contributes_zero_to_mrr():
    report = RetrievalReport("cfg", [_result("a", 1), _result("b", None)])
    assert report.mrr == pytest.approx(0.5)


def test_empty_report_scores_zero_rather_than_dividing_by_zero():
    report = RetrievalReport("cfg", [])
    assert (report.hit_at_1_rate, report.recall_at_k, report.mrr) == (0.0, 0.0, 0.0)


def test_misses_lists_only_what_was_never_retrieved():
    report = RetrievalReport("cfg", [_result("found", 4), _result("lost", None)])
    assert [r.id for r in report.misses] == ["lost"]


def test_markdown_table_has_a_row_per_report():
    table = markdown_table([RetrievalReport("A", [_result("a", 1)]),
                            RetrievalReport("B", [_result("b", None)])])
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, two rows
    assert "| A " in lines[2] and "| B " in lines[3]


# ---- the dataset ----

def test_question_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_question_carries_a_passage_marker_and_a_reference():
    for q in QUESTIONS:
        assert q.expect, f"{q.id} has no expected passage"
        assert q.reference, f"{q.id} has no reference answer"


def test_non_english_questions_have_an_english_form():
    # The translate-the-query eval is meaningless without it.
    for q in QUESTIONS:
        if q.lang != "en":
            assert q.question_en, f"{q.id} is {q.lang} but has no question_en"


def test_translate_swaps_the_query_only_for_non_english_questions():
    spanish = next(q for q in QUESTIONS if q.lang == "es")
    english = next(q for q in QUESTIONS if q.lang == "en")
    assert spanish.retrieval_query(translate=True) == spanish.question_en
    assert spanish.retrieval_query(translate=False) == spanish.question
    assert english.retrieval_query(translate=True) == english.question


def test_filtering_by_language_returns_only_that_language():
    assert {q.lang for q in questions(["es"])} == {"es"}


def test_unknown_language_fails_loudly_instead_of_returning_nothing():
    # Silently returning [] would report a perfect score over zero questions.
    with pytest.raises(ValueError):
        questions(["de"])


def test_every_expected_passage_actually_appears_in_the_demo_corpus():
    """
    Guards the dataset against the document changing underneath it.

    If demo/churreria_calderon.pdf is regenerated with different wording, every
    question silently becomes unanswerable and the eval reports 0/17 as though the
    retriever broke.
    """
    from langchain_community.document_loaders import PyPDFLoader

    import rag_core

    corpus = " ".join(
        doc.page_content for path in rag_core.demo_pdf_paths()
        for doc in PyPDFLoader(path).load()
    )
    for q in QUESTIONS:
        assert chunk_contains(corpus, q.expect), f"{q.id}: no expected passage in the corpus"


# ---- judge plumbing ----

def test_parse_json_object_reads_a_bare_object():
    assert judges.parse_json_object('{"claims": ["a"]}') == {"claims": ["a"]}


def test_parse_json_object_digs_through_fences_and_prose():
    # Models add these no matter what the prompt says.
    response = 'Sure! Here is the JSON:\n```json\n{"answer_relevancy": 0.5}\n```\nHope that helps.'
    assert judges.parse_json_object(response) == {"answer_relevancy": 0.5}


def test_parse_json_object_raises_rather_than_returning_an_empty_dict():
    # An empty dict would score as zero and be indistinguishable from a bad answer.
    with pytest.raises(ValueError):
        judges.parse_json_object("I could not answer that.")


def test_faithfulness_is_the_supported_fraction():
    verdicts = [{"supported": True}, {"supported": False}, {"supported": True}]
    assert judges.faithfulness_from_verdicts(verdicts) == pytest.approx(2 / 3)


def test_faithfulness_of_a_claimless_answer_is_none_not_zero():
    # A refusal makes no claims; scoring it 0.0 would punish correct behaviour.
    assert judges.faithfulness_from_verdicts([]) is None


def test_a_missing_supported_flag_counts_as_unsupported():
    assert judges.faithfulness_from_verdicts([{"why": "unclear"}]) == 0.0


def test_context_precision_is_relevant_over_retrieved():
    assert judges.context_precision([1, 3], retrieved=4) == pytest.approx(0.5)


def test_context_precision_ignores_passages_the_judge_invented():
    # A judge citing passage 9 out of 4 must not push precision above 1.0.
    assert judges.context_precision([1, 9, 9], retrieved=2) == pytest.approx(0.5)


def test_context_precision_is_none_when_nothing_was_retrieved():
    assert judges.context_precision([], retrieved=0) is None


def test_clamp_forces_a_judges_number_into_range():
    assert judges.clamp(1.7) == 1.0
    assert judges.clamp(-2) == 0.0
    assert judges.clamp("0.5") == 0.5


def test_clamp_returns_none_for_a_non_number():
    assert judges.clamp("very good") is None


def test_mean_skips_missing_scores():
    assert judges.mean([1.0, None, 0.0]) == pytest.approx(0.5)


def test_mean_of_nothing_is_none():
    assert judges.mean([None, None]) is None


def test_format_contexts_numbers_passages_from_one():
    assert judges.format_contexts(["first", "second"]) == "[1] first\n\n[2] second"


def test_judge_survives_an_unparseable_response_without_crashing_the_run():
    scores = judges.judge(
        lambda prompt: "the model is having a bad day",
        question="q", answer="a", reference="r", contexts=["c"],
    )
    assert scores.faithfulness is None
    assert scores.answer_relevancy is None


def test_judge_scores_a_well_formed_conversation():
    replies = iter([
        '{"claims": ["Closed on Mondays", "Open at 9"]}',
        '{"verdicts": [{"claim": "Closed on Mondays", "supported": true},'
        ' {"claim": "Open at 9", "supported": false}]}',
        '{"answer_relevancy": 1.0, "answer_correctness": 0.8,'
        ' "relevant_passages": [1], "note": "fine"}',
    ])
    scores = judges.judge(
        lambda prompt: next(replies),
        question="Are you open on Mondays?",
        answer="We are closed on Mondays and open at 9.",
        reference="Closed on Mondays.",
        contexts=["Monday: closed", "unrelated"],
    )
    assert scores.faithfulness == pytest.approx(0.5)
    assert scores.answer_relevancy == 1.0
    assert scores.answer_correctness == pytest.approx(0.8)
    assert scores.context_precision == pytest.approx(0.5)
    assert (scores.claims, scores.supported_claims) == (2, 1)


# ---- the agent the eval scores ----

def test_the_agent_is_told_to_search_before_refusing_as_off_topic():
    """
    Regression test for a failure the eval caught.

    Asked "Are you on Uber Eats or DoorDash?", the agent decided the question was
    off-topic and answered without ever calling the retrieval tool — a plain
    customer question, refused. Grounded retrieval went 16/17 to 17/17 once the
    prompt forbade refusing before searching.
    """
    import rag_core

    prompt = rag_core.build_system_prompt(demo_mode=False).lower()
    assert "off-topic" in prompt
    assert "search_documents" in prompt


def test_the_cross_lingual_rule_is_demo_only():
    # Production embeddings are multilingual and must query in the original
    # language; leaking this rule into production would degrade retrieval.
    import rag_core

    assert "ENGLISH" in rag_core.build_system_prompt(demo_mode=True)
    assert "ENGLISH" not in rag_core.build_system_prompt(demo_mode=False)


def test_search_tool_hands_retrieved_documents_to_its_caller():
    """The UI renders sources and the eval scores contexts through this callback."""
    from langchain_core.documents import Document

    import rag_core

    captured = []

    class FakeRetriever:
        def invoke(self, query):
            return [Document(page_content="Monday: closed", metadata={"source": "kb.pdf", "page": 0})]

    tool = rag_core.make_search_tool(FakeRetriever(), on_documents=captured.extend)
    output = tool.invoke({"query": "hours"})

    assert len(captured) == 1
    assert "kb.pdf — p. 1" in output
    assert "Monday: closed" in output


def test_search_tool_says_so_when_nothing_is_found():
    import rag_core

    class EmptyRetriever:
        def invoke(self, query):
            return []

    tool = rag_core.make_search_tool(EmptyRetriever())
    assert "No relevant information" in tool.invoke({"query": "anything"})

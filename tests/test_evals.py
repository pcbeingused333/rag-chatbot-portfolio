"""
Tests for the evaluation harness itself.

An eval that is wrong is worse than no eval: it produces numbers that look
authoritative and are not. Everything scoreable without a network call is tested
here — the substring matching, the aggregates, the judge parsing.
"""
import pytest

from evals import abstention, answers, judges
from evals.dataset import QUESTIONS, UNANSWERABLE, questions
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


def test_every_question_carries_an_expected_provision_and_a_reference():
    for q in QUESTIONS:
        assert q.expect_citations, f"{q.id} has no expected provision"
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


def test_every_expected_provision_exists_in_the_corpus():
    """
    Guards the dataset against the corpus being rebuilt underneath it.

    If build_gdpr_corpus.py changes how it numbers provisions, every question
    silently becomes unanswerable and the eval reports 0/25 as though the retriever
    broke. A citation that does not resolve is a dataset bug, not a retrieval score.
    """
    import rag_core

    available = {
        doc.metadata["citation"] for doc in rag_core.load_legal_corpus()
    }
    for q in QUESTIONS:
        for citation in q.expect_citations:
            assert citation in available, f"{q.id}: {citation} is not in the corpus"


def test_unanswerable_questions_name_only_real_adjacent_provisions():
    """
    The adjacent provisions are the ones the retriever is *expected* to return.

    They document why each question is a near miss rather than an unrelated one, so
    a typo there would quietly misdescribe what the abstention eval is testing.
    """
    import rag_core

    available = {
        doc.metadata["citation"] for doc in rag_core.load_legal_corpus()
    }
    for q in UNANSWERABLE:
        assert q.why, f"{q.id} does not say why it is unanswerable"
        for citation in q.adjacent:
            assert citation in available, f"{q.id}: {citation} is not in the corpus"


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


# The judge moved to Qwen to stop the shipped model grading its own answers. Qwen
# inlines its reasoning in the message content instead of returning it out of band
# the way the gpt-oss models do, so the reply shape changed and every judged metric
# silently came back n/a. These pin the shape, not just the parse.

# The reasoning restates the schema it was asked for, so it contains braces — which
# is why the old greedy `{.*}` ran from inside the reasoning to the real closing
# brace and could never parse.
JUDGE_REPLY_WITH_REASONING = (
    "<think>\n"
    'The user wants a verdict shaped like {"verdict": "abstained", "why": "..."}.\n'
    "The reply says the corpus does not contain the list, so it abstained.\n"
    "</think>\n"
    '{"verdict": "abstained", "why": "Says the corpus does not cover it."}'
)


def test_parse_json_object_ignores_braces_inside_a_think_block():
    assert judges.parse_json_object(JUDGE_REPLY_WITH_REASONING) == {
        "verdict": "abstained",
        "why": "Says the corpus does not cover it.",
    }


# The next two pass against the old parser too, and are kept anyway: they pin the
# failure modes the *new* balanced scanner introduces — truncation and braces inside
# strings — rather than the regression above. Said out loud so nobody counts them as
# evidence the bug is caught.

def test_parse_json_object_rejects_a_reply_cut_off_inside_its_reasoning():
    # Truncated before the answer: there is no verdict, and the deliberations are
    # not one. A parse failure here is the correct, loud outcome.
    with pytest.raises(ValueError):
        judges.parse_json_object('<think>The shape is {"verdict": ')


def test_parse_json_object_takes_the_last_object_not_the_first():
    # Prose around a JSON reply is a preamble far more often than a postscript.
    response = 'For reference the shape is {"verdict": "x"}.\n{"verdict": "hedged"}'
    assert judges.parse_json_object(response) == {"verdict": "hedged"}


def test_parse_json_object_is_not_fooled_by_braces_inside_strings():
    assert judges.parse_json_object('{"why": "it printed { and never closed it"}') == {
        "why": "it printed { and never closed it"
    }


def test_strip_reasoning_leaves_an_ordinary_reply_alone():
    assert judges.strip_reasoning('  {"verdict": "hedged"}  ') == '{"verdict": "hedged"}'


def test_abstention_reads_a_verdict_wrapped_in_reasoning():
    # The regression that mattered: a correct abstention scored `unparsed`, so the
    # published table read 0/6 abstained when the system had abstained every time.
    assert abstention._parse_verdict(JUDGE_REPLY_WITH_REASONING)["verdict"] == "abstained"


def test_a_run_where_the_judge_scored_nothing_does_not_exit_clean():
    # A table of n/a under exit 0 reads as a finished run. It is a harness fault.
    nothing = [dict.fromkeys(answers._JUDGED_METRICS)]
    assert answers._judge_produced_nothing(nothing)


def test_one_unscored_question_is_not_treated_as_a_broken_judge():
    # A refusal has no claims to be unfaithful about; None there is ordinary.
    rows = [dict.fromkeys(answers._JUDGED_METRICS), {**dict.fromkeys(answers._JUDGED_METRICS), "faithfulness": 1.0}]
    assert not answers._judge_produced_nothing(rows)


def test_a_metric_scored_on_a_handful_of_questions_is_not_publishable():
    # The shape that shipped on 28-Aug: the judge ran out of tokens mid-thought on
    # the claim-extraction prompt, so `faithfulness` scored 2 of 25 while the other
    # three scored all 25. The table printed "Faithfulness 1.00" - a mean over two
    # answers - in the same column as means over twenty-five.
    rows = [
        {**dict.fromkeys(answers._JUDGED_METRICS, 1.0), "faithfulness": None}
        for _ in range(25)
    ]
    rows[0]["faithfulness"] = 1.0
    rows[1]["faithfulness"] = 1.0

    thin = answers._thinly_scored_metrics(rows)

    assert [m for m, _ in thin] == ["faithfulness"]
    assert thin[0][1] == 2
    # Not the all-empty case: that has its own message, and this one hid behind it.
    assert not answers._judge_produced_nothing(rows)


def test_a_few_unscored_questions_do_not_trip_the_coverage_guard():
    # None is ordinary when an answer makes no claims. Only a collapse should trip.
    rows = [dict.fromkeys(answers._JUDGED_METRICS, 1.0) for _ in range(10)]
    rows[0]["faithfulness"] = None
    rows[1]["faithfulness"] = None

    assert answers._thinly_scored_metrics(rows) == []


def test_the_summary_says_how_many_questions_each_metric_scored(capsys):
    rows = [
        {
            **dict.fromkeys(answers._JUDGED_METRICS, 1.0),
            "faithfulness": 1.0 if i < 2 else None,
            "grounded": True,
            "error": None,
            "id": f"q{i}",
            "lang": "en",
            "note": "",
            "answer": "",
        }
        for i in range(25)
    ]

    answers._summarise(rows)

    table = capsys.readouterr().out
    # The count has to travel with the mean, or the mean is read as the set's.
    assert "2/25" in table
    assert "Scored" in table


def test_the_judge_is_given_room_to_finish_its_reasoning(monkeypatch):
    # A judge that inlines reasoning spends the token budget thinking before it
    # answers. On the default budget generation was cut off mid-thought, the reply
    # had an unclosed `<think>`, and the caller recorded "no claims" for an answer
    # that was full of them.
    captured = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def invoke(self, _messages):
            raise AssertionError("not called")

    import langchain_groq

    monkeypatch.setattr(langchain_groq, "ChatGroq", FakeChatGroq)
    answers._make_judge_callable("qwen/qwen3.6-27b")

    assert captured["max_tokens"] == answers.JUDGE_MAX_TOKENS
    assert answers.JUDGE_MAX_TOKENS >= 4096
    # The budget alone was not enough: on the longer answers the judge still spent
    # all of it reasoning, so only the shortest questions reached their JSON. The
    # extraction it is asked for does not need deliberation.
    assert captured["reasoning_effort"] == "none"


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


# --- tool-call probe: a retired model is not a failing model ------------------


def test_a_retired_model_is_reported_as_unavailable_not_as_a_100_percent_failure():
    """Groq retired `llama-3.3-70b-versatile` and the probe scored it 0/10 answered,
    100% failure rate — which reads as a model that cannot emit a tool call, when no
    request ever reached a model. The distinction is the whole point of the table."""
    from evals import tool_calls

    class MissingModelAgent:
        def invoke(self, _state):
            raise RuntimeError(
                "Error code: 404 - {'error': {'message': 'The model "
                "`llama-3.3-70b-versatile` does not exist or you do not have access "
                "to it.', 'code': 'model_not_found'}}"
            )

    report = tool_calls._probe_with_agent(MissingModelAgent(), "llama-3.3-70b-versatile", attempts=10)

    assert report.unavailable == "model_not_found"
    assert report.failure_rate is None
    assert report.tool_call_failures == 0
    assert report.other_failures == 0
    # Stopped on the first refusal instead of spending the other nine requests.
    assert report.attempts == 0


def test_a_model_that_cannot_emit_a_tool_call_is_still_scored():
    """The guard above must not swallow the failure the probe exists to measure."""
    from evals import tool_calls

    class MalformedToolCallAgent:
        def invoke(self, _state):
            raise RuntimeError("Error code: 400 - {'error': {'code': 'tool_use_failed'}}")

    report = tool_calls._probe_with_agent(MalformedToolCallAgent(), "some-model", attempts=3)

    assert report.unavailable is None
    assert report.attempts == 3
    assert report.tool_call_failures == 3
    assert report.failure_rate == 1.0


# --- the judge is as rate-limited as the model it judges ----------------------


def test_a_rate_limited_judge_waits_and_retries_instead_of_killing_the_run():
    """The `answers` run died at question 7 of 25 on a tokens-per-minute 429 raised
    by a judge call, throwing away the six questions already paid for. The agent path
    had backoff; the judge path called the API directly."""
    import rag_core

    waits = []
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError(
                "Error code: 429 - rate limit reached ... on tokens per minute (TPM)"
            )
        return "verdict"

    assert rag_core.call_llm_with_retry(flaky, sleep=waits.append) == "verdict"
    assert attempts["n"] == 3
    # It waited, and waited longer the second time, rather than spending the retry
    # immediately on a budget that only refills with the clock.
    assert len(waits) == 2 and waits[1] > waits[0]


def test_a_judge_failure_a_retry_cannot_fix_is_raised_at_once():
    import rag_core

    attempts = {"n": 0}

    def bad_key():
        attempts["n"] += 1
        raise RuntimeError("Error code: 401 - invalid api key")

    with pytest.raises(RuntimeError):
        rag_core.call_llm_with_retry(bad_key, sleep=lambda _s: None)
    assert attempts["n"] == 1


# --- a daily budget is not a transient failure -------------------------------


def test_a_daily_budget_exhaustion_is_not_retried():
    """Groq reports the per-day and the per-minute budget through the same 429. The
    per-minute one refills while the run waits; the per-day one says "try again in
    13m54s", so backing off spends attempts to be told the same thing."""
    import rag_core

    waits = []
    attempts = {"n": 0}

    def spent():
        attempts["n"] += 1
        raise RuntimeError(
            "Error code: 429 - Rate limit reached ... on tokens per day (TPD): "
            "Limit 200000, Used 199927. Please try again in 13m54.624s"
        )

    with pytest.raises(rag_core.DailyBudgetExhausted):
        rag_core.call_llm_with_retry(spent, sleep=waits.append)
    assert attempts["n"] == 1
    assert waits == []


def test_a_per_minute_limit_is_still_retried():
    """The guard above must not turn every 429 into a stopped run."""
    import rag_core

    exc = RuntimeError("429 rate limit ... on tokens per minute (TPM). try again in 1.6s")
    assert not rag_core.is_daily_budget_exhausted(exc)
    assert rag_core.is_transient_llm_error(exc)


# --- the judge has to be a different model than the one it judges -------------


def test_the_default_judge_is_not_the_shipped_model():
    """The judge started as a separate model and silently stopped being one when the
    shipped model changed under it: both defaults read `openai/gpt-oss-120b`, so every
    judged number was self-assessed. Nothing failed — which is why this is a test."""
    import rag_core
    from evals import abstention, answers, judges

    assert judges.DEFAULT_JUDGE_MODEL != rag_core.DEFAULT_LLM_MODEL
    assert answers.DEFAULT_JUDGE_MODEL != rag_core.DEFAULT_LLM_MODEL
    assert abstention.DEFAULT_JUDGE_MODEL != rag_core.DEFAULT_LLM_MODEL


def test_a_run_that_would_grade_its_own_homework_is_refused():
    from evals.judges import SelfJudgingError, check_judge_independence

    with pytest.raises(SelfJudgingError):
        check_judge_independence("openai/gpt-oss-120b", "openai/gpt-oss-120b")
    # Case and stray whitespace are the same model, not a different one.
    with pytest.raises(SelfJudgingError):
        check_judge_independence("openai/gpt-oss-120b", " OpenAI/GPT-OSS-120B ")


def test_two_different_models_are_allowed():
    from evals.judges import check_judge_independence

    check_judge_independence("openai/gpt-oss-120b", "qwen/qwen3.6-27b")


def test_the_agent_path_stops_the_run_too_when_the_daily_budget_goes():
    """The judge path stopped cleanly and the agent path did not: an exhausted budget
    was caught by the per-question handler and recorded as an agent failure, which
    turns one sentence about a spent budget into twenty-five bogus failures."""
    import rag_core
    from evals import answers

    class BrokeAgent:
        def invoke(self, _state):
            raise rag_core.DailyBudgetExhausted("tokens per day (TPD)")

    question = QUESTIONS[0]
    with pytest.raises(rag_core.DailyBudgetExhausted):
        answers._answer(BrokeAgent(), question)


def test_an_ordinary_agent_failure_is_still_scored_as_a_miss():
    """The re-raise above must not turn every agent error into a stopped run."""
    from evals import answers

    class FlakyAgent:
        def invoke(self, _state):
            raise RuntimeError("500 internal server error")

    result = answers._answer(FlakyAgent(), QUESTIONS[0])
    assert result["answer"] == ""
    assert "RuntimeError" in result["error"]

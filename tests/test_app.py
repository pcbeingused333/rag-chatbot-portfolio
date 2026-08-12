"""
Smoke tests for the Streamlit app, run headlessly with Streamlit's AppTest.

"The server starts" is not the same as "the script runs". A Streamlit app boots
fine and then throws on the first browser connection — which is exactly when a
visitor opens it. AppTest executes the script body, so an import error, a bad
session-state access or a missing demo file fails here instead.

This matters more here than in a repo with a deploy step. Streamlit Community
Cloud redeploys from GitHub on every push, so there is nothing between a bad
commit and the public demo except this suite.
"""
import os

import pytest

pytest.importorskip("streamlit.testing.v1", reason="streamlit not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture
def demo_app(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    # A developer's .env would otherwise decide what these tests are running
    # against, and the answer would differ on CI.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return AppTest.from_file(APP, default_timeout=90).run()


def test_the_script_runs_without_raising(demo_app):
    assert list(demo_app.exception) == []
    assert [e.value for e in demo_app.error] == []


def test_a_first_time_visitor_is_given_something_to_click(demo_app):
    """
    The original demo opened to an empty box with nothing indexed, so a visitor
    saw nothing and left. Demo mode exists to make the first screen answerable.
    """
    assert demo_app.title[0].value == "🧠 RAG Assistant with Intelligent Agent"
    assert len(demo_app.button) >= 4


def test_the_first_render_does_not_load_the_embedding_model(monkeypatch):
    """
    The reason DEMO_MODE exists at all.

    The production embedding model is ~2 GB and does not fit the free tier's 1 GB,
    and even the small one costs a download. Everything heavy sits behind
    st.cache_resource and must stay behind it: the moment a render touches the
    model, the first visitor waits for it — or the container is killed and they
    get nothing. Making the call explode is the only way to prove no render path
    reaches it.
    """
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    import rag_core

    def forbidden():
        raise AssertionError("the first render must not load the embedding model")

    monkeypatch.setattr(rag_core, "get_embeddings", forbidden)

    result = AppTest.from_file(APP, default_timeout=90).run()
    assert list(result.exception) == []


def test_demo_mode_says_which_store_it_is_using(demo_app):
    """The caption is how a reader tells the two runtime modes apart."""
    captions = [c.value for c in demo_app.caption]
    assert any("FAISS (in-memory)" in c for c in captions)


def test_the_demo_corpus_is_committed(demo_app):
    """
    Demo mode auto-indexes this file at startup. It is the whole demo, and a
    missing one turns the public app into an assistant that knows nothing.
    """
    corpus = os.path.join(os.path.dirname(APP), "demo", "churreria_calderon.pdf")
    assert os.path.exists(corpus)

"""The dashboard script itself, executed headless through streamlit.testing.

AppTest runs dashboard/app.py with the real rendering pipeline but no
browser and no websocket server: anything the page code raises shows up in
`app.exception`, and rendered widgets can be inspected. These tests assert
only what is true with or without a live backend on localhost:8000 - no
page may crash, and every navigation entry must be reachable - because the
suite has to stay green offline.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

PAGES = ("Overview", "AI Assistant", "Business Analytics",
         "Evaluation & Performance", "Documents", "History")


def launch() -> AppTest:
    return AppTest.from_file(str(DASHBOARD), default_timeout=120)


def test_the_app_boots_to_the_overview_with_the_full_navigation():
    app = launch()
    app.run()
    assert not app.exception, repr(app.exception)
    radio = app.sidebar.radio[0]
    assert radio.value == "Overview"
    assert set(radio.options) == set(PAGES)
    assert not any(element.value == "None" for element in app.code), \
        "page dispatch must not render None as a page element"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_crashing(page):
    app = launch()
    app.run()
    app.sidebar.radio[0].set_value(page)
    app.run()
    assert not app.exception, f"{page} crashed with {app.exception!r}"


def test_the_assistant_page_offers_quick_queries_and_a_chat_input():
    app = launch()
    app.run()
    app.sidebar.radio[0].set_value("AI Assistant")
    app.run()
    assert not app.exception
    buttons = [button.label for button in app.button]
    assert any("total revenue" in label.lower() for label in buttons)
    assert app.chat_input


def test_the_start_new_conversation_control_rotates_the_session():
    app = launch()
    app.run()
    buttons = [button for button in app.sidebar.button
               if button.label == "Start a new conversation"]
    assert buttons, "the sidebar must offer a fresh conversation"
    buttons[0].click()
    app.run()
    assert not app.exception, repr(app.exception)
    captions = [caption.value for caption in app.sidebar.caption]
    assert any(caption.startswith("Conversation:") for caption in captions), \
        "after rotation the current conversation id must be visible"


def test_overview_uses_the_exact_live_order_total(monkeypatch):
    def fake_request(method, url, timeout, **kwargs):
        if url.endswith("/health"):
            payload = {
                "version": "1.1.0",
                "components": {
                    "database": {"available": True, "detail": "connected"},
                    "llm": {"available": True, "model": "test"},
                    "documents": {"available": True, "chunks": 1,
                                  "documents": 1},
                },
            }
        else:
            payload = {
                "queries": [{
                    "rows": [{"order_total_due": "123216786.1159"}],
                }],
            }
        return SimpleNamespace(status_code=200, json=lambda: payload)

    monkeypatch.setattr(requests, "request", fake_request)

    def missing_local_snapshot(*args, **kwargs):
        raise FileNotFoundError("local snapshot is absent")

    monkeypatch.setattr("pandas.read_csv", missing_local_snapshot)
    app = launch()
    app.run()
    revenue = next(metric for metric in app.metric
                   if metric.label == "Revenue — order-level TotalDue (USD)")
    assert revenue.value == "123,216,786.1159"
    assert revenue.delta == "live PostgreSQL"


def test_pilot_tester_selection_survives_page_navigation():
    app = launch()
    app.run()
    app.sidebar.radio[0].set_value("Evaluation & Performance")
    app.run()
    app.selectbox[0].set_value("Tester 01")
    app.run()

    app.sidebar.radio[0].set_value("AI Assistant")
    app.run()
    app.sidebar.radio[0].set_value("Evaluation & Performance")
    app.run()
    assert app.selectbox[0].value == "Tester 01"

    app.selectbox[0].set_value("Tester 02")
    app.run()
    app.sidebar.radio[0].set_value("Overview")
    app.run()
    app.sidebar.radio[0].set_value("Evaluation & Performance")
    app.run()
    assert app.selectbox[0].value == "Tester 02"


def test_assistant_shows_the_selected_anonymous_feedback_tester():
    app = launch()
    app.run()
    app.sidebar.radio[0].set_value("AI Assistant")
    app.session_state["pilot_tester"] = "Tester 01"
    app.run()
    assert any(caption.value == "Anonymous tester for feedback: Tester 01"
               for caption in app.caption)

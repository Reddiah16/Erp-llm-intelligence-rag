"""Evaluation & Performance dashboard page: source-level contract tests.

Streamlit cannot render inside pytest, so - exactly like
test_dashboard_phase3.py - the dashboard contract is text: the page lists
itself in navigation, sends the anonymous tester id with /chat, offers the
feedback controls against /feedback, reads /evaluation/metrics and
/evaluation/export, and shows 'Not enough data' instead of inventing a
percentage when the sample is empty.
"""
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_the_evaluation_page_is_in_the_navigation():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert '"Evaluation & Performance"' in source, \
        "the page must be listed in PAGES so the sidebar shows it"
    assert "page_evaluation()" in source, "the page must be dispatched"


def test_the_page_drives_the_pilot_endpoints():
    source = DASHBOARD.read_text(encoding="utf-8")
    for path in ('"/evaluation/metrics"', '"/evaluation/suggested-questions"',
                 "/evaluation/export", '"/feedback"'):
        assert path in source, f"the dashboard never calls {path}"


def test_feedback_controls_ask_every_question_and_send_the_turn_id():
    source = DASHBOARD.read_text(encoding="utf-8")
    for needle, why in (
            ("Is this answer helpful?", "the helpful rating"),
            ("Is the answer factually correct?", "the correctness rating"),
            ("Are the citations relevant?", "the citation relevance rating"),
            ("Optional written feedback", "the optional comment"),
            ('"turn_id": turn_id', "feedback attaches to the exact exchange"),
            ("not_sure", "the not-sure vocabulary is sent as its stored value"),
            ("not_applicable", "the not-applicable vocabulary is sent verbatim")):
        assert needle in source, why


def test_empty_samples_say_not_enough_data_instead_of_inventing_a_percent():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert source.count("Not enough data") >= 1, \
        "empty metric samples must show 'Not enough data'"
    assert 'payload.get("value")' in source and "value is None" in source, \
        "a None metric value (zero sample) renders the no-data text"


def test_evaluation_metrics_disclose_unverified_provenance():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert "Provenance is not consistently " in source
    assert "cannot be attributed to internal " in source
    assert "manual testing, or an external pilot" in source
    assert "do not verify participant type" in source


def test_the_anonymous_tester_travelles_from_selector_to_api():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert "pilot_tester" in source, \
        "the tester selector must persist its choice in session state"
    assert '"tester_id"' in source, \
        "both /chat and /feedback must receive the tester id"
    assert "Tester" in source, "the selector offers the Tester 01..10 labels"


def test_every_headline_metric_is_rendered():
    source = DASHBOARD.read_text(encoding="utf-8")
    for needle in ("Total questions evaluated", "Answer accuracy",
                   "SQL execution success", "Avg response latency",
                   "Retrieval accuracy", "Data grounding", "Cost per query",
                   "Task completion"):
        assert needle in source, f"the headline metric '{needle}' is missing"


def test_the_accuracy_by_route_chart_frame_builds_from_real_metrics():
    """Runtime regression: the page's route chart crashed with NameError on
    2026-11-07 because the dict comprehension's clause sat inside the list
    comprehension. The transformation now lives in a testable module."""
    from src.evaluation.dashboard_views import route_accuracy_frame
    routes = {
        "sql": {"value": 100.0, "sample": 2},
        "rag": {"value": None, "sample": 0},
        "hybrid": {"value": 50.0, "sample": 1},
    }
    frame = route_accuracy_frame(routes)
    assert list(frame.index) == ["accuracy %"]
    assert list(frame.columns) == ["sql", "rag", "hybrid"]
    assert frame.loc["accuracy %", "sql"] == 100.0
    assert frame.loc["accuracy %", "rag"] == 0.0, "no-data routes chart as 0.0"
    assert frame.loc["accuracy %", "hybrid"] == 50.0


def test_the_accuracy_by_route_chart_frame_is_empty_shape_safe():
    from src.evaluation.dashboard_views import route_accuracy_frame
    frame = route_accuracy_frame({"sql": {"value": None, "sample": 0}})
    assert not frame.empty and frame.loc["accuracy %", "sql"] == 0.0

"""The pilot evaluation store: metric math, persistence and the no-fabrication
contract, against deterministic records only.

The store is deliberately file-based (src/evaluation/pilot.py), so these tests
need no database, no Ollama and no network - each test seeds a temporary
directory, drives PilotStore directly, and asserts exact numbers or absence
('not enough data') where the sample size is zero.
"""
import csv as csv_module

import pytest

from src.evaluation import pilot

CATALOG_QUESTION = "How many purchase orders are pending?"


def turn(turn_id, route="sql", question_id="q123", answered=True,
         sql_executed=False, sql_success=False, citations=0,
         duration=None, tester="", asked="2026-11-07T10:00:00.000+00:00",
         cost=0.0, kind="hash", session="sess"):
    return pilot.TurnRecord(
        turn_id=turn_id,
        session_id=session,
        question_id=question_id,
        question_kind=kind,
        tester_id=tester,
        route=route,
        classifier="model",
        asked_at=asked,
        duration_ms=duration,
        sql_executed=sql_executed,
        sql_success=sql_success,
        citations_count=citations,
        data_available=sql_success,
        documents_available=citations > 0,
        generated=answered,
        unverified_figures=0,
        warnings=0,
        api_cost=cost,
        cost_note="local Ollama inference - API cost only, "
                  "infrastructure not included" if cost is not None else "",
    )


def grade(turn_id, helpful=None, correct=None, citations=None, comment=None,
          tester="", session="sess", question_id="q123"):
    return pilot.FeedbackRecord(
        turn_id=turn_id,
        session_id=session,
        question_id=question_id,
        tester_id=tester,
        helpful=helpful,
        correct=correct,
        citations_relevant=citations,
        comment=comment,
        at="2026-11-07T10:05:00.000+00:00",
    )


# ---------------------------------------------------------------- identifiers

def test_catalog_questions_get_their_own_stable_identifier():
    session = "session-01"
    assert pilot.question_id_for(session, CATALOG_QUESTION) == ("PQ-SQL-01",
                                                                "catalog")
    assert pilot.question_id_for(session, f"  {CATALOG_QUESTION}  ") == \
        ("PQ-SQL-01", "catalog")


def test_free_form_questions_hash_to_a_stable_non_readable_id():
    first = pilot.question_id_for("session-01", "an exact wording")
    second = pilot.question_id_for("session-01", "an exact wording")
    other = pilot.question_id_for("session-02", "an exact wording")
    assert first == second, "repeats of the same question in one session are stable"
    assert first != other, "the same wording in another session is a distinct id"
    value, kind = first
    assert kind == "hash"
    assert len(value) == 16
    assert "an exact wording" not in value, "the text must never be embedded"


# --------------------------------------------------------------- persistence

def test_records_survive_a_reconstructed_store(tmp_path):
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t1", route="sql", question_id="PQ-SQL-01",
                           sql_executed=True, sql_success=True, duration=1200))
    store.record_feedback(grade("t1", helpful=True, correct="yes",
                                question_id="PQ-SQL-01"))

    reopened = pilot.PilotStore(tmp_path)   # a restarted process reads the same files
    assert reopened.summary_metrics()["total_questions"] == 1
    recorded = reopened.turn("t1")
    assert recorded["route"] == "sql"
    assert reopened.feedback_for_turn("t1")["correct"] == "yes"
    assert reopened.summary_metrics()["answer_accuracy"]["value"] == 100.0


def test_an_empty_store_reports_no_data_rather_than_zero_percent(tmp_path):
    report = pilot.PilotStore(tmp_path).summary_metrics()
    assert report["total_questions"] == 0
    for metric in ("answer_accuracy", "sql_execution_success",
                   "retrieval_accuracy", "data_grounding", "task_completion"):
        assert report[metric]["value"] is None
        assert report[metric]["sample"] == 0
    assert report["average_latency_ms"]["value"] is None
    assert report["cost_per_query"]["value"] is None
    assert report["over_time"] == [] and report["latency_series"] == []
    assert report["accuracy_series"] == []
    # every route bucket also says 'not enough data'
    assert all(bucket["value"] is None and bucket["sample"] == 0
               for bucket in report["accuracy_by_route"].values())


def test_feedback_fold_keeps_the_latest_event_per_turn(tmp_path):
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t1"))
    store.record_feedback(grade("t1", helpful=True, correct="yes"))
    store.record_feedback(grade("t1", helpful=False, correct="no"))
    held = store.feedback_for_turn("t1")
    assert held["helpful"] is False and held["correct"] == "no", \
        "the fold shows the latest submitted grades"


# ----------------------------------------------------------- metric arithmetic

@pytest.fixture
def seeded(tmp_path):
    """Three deterministic turns, two graded."""
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t1", route="sql", sql_executed=True,
                           sql_success=True, duration=1000))
    store.record_turn(turn("t2", route="rag", citations=2, duration=2000,
                           asked="2026-11-07T11:00:00.000+00:00"))
    store.record_turn(turn("t3", route="sql", sql_executed=True,
                           sql_success=False, duration=3000,
                           asked="2026-11-07T12:00:00.000+00:00"))
    store.record_feedback(grade("t1", helpful=True, correct="yes"))
    store.record_feedback(grade("t2", helpful=False, correct="no",
                                citations="yes", comment="citations good"))
    return store


def test_the_seven_metrics_match_the_deterministic_seed(seeded):
    report = seeded.summary_metrics()
    assert report["answer_accuracy"] == {"value": 50.0,
                                         "sample": 2}          # 1 yes / 2 graded
    assert report["sql_execution_success"] == {"value": 50.0,
                                               "sample": 2}        # 1 of 2 executed
    assert report["average_latency_ms"] == {"value": 2000.0, "sample": 3}
    assert report["retrieval_accuracy"] == {"value": 100.0,
                                            "sample": 1}           # 1 relevant / 1 judged
    assert report["data_grounding"] == {"value": round(100 * 2 / 3, 1),
                                        "sample": 3}              # t3 has no evidence
    assert report["task_completion"] == {"value": 50.0, "sample": 2}  # 1 helpful / 2


def test_route_breakdown_only_counts_that_routes_turns(seeded):
    routes = seeded.summary_metrics()["accuracy_by_route"]
    assert routes["sql"] == {"value": 100.0, "sample": 1}   # t1 graded yes; t3 ungraded
    assert routes["rag"] == {"value": 0.0, "sample": 1}     # t2 graded no
    assert routes["hybrid"] == {"value": None, "sample": 0}
    assert routes["unsupported"] == {"value": None, "sample": 0}


def test_grounding_requires_evidence_not_just_a_generated_answer(tmp_path):
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t1", route="rag", citations=1, duration=100))
    store.record_turn(turn("t2", route="sql", sql_executed=True,
                           sql_success=False, asked="2026-11-07T11:00:00.000+00:00"))
    store.record_turn(turn("t3", route="unsupported",
                           asked="2026-11-07T12:00:00.000+00:00"))
    report = store.summary_metrics()
    # t3 is refused (unsupported) and is never counted as an evaluated answer
    assert report["data_grounding"]["sample"] == 2
    assert report["data_grounding"]["value"] == 50.0


def test_cost_is_zero_for_local_inference_and_unknown_for_remote():
    assert pilot.is_local_inference("http://localhost:11434")
    assert pilot.is_local_inference("http://host.docker.internal:11434")
    assert not pilot.is_local_inference("https://api.example.com/v1")

    store = pilot.PilotStore()   # directory comes from the isolated test env
    store.wipe()
    store.record_turn(turn("t1", cost=0.0))
    store.record_turn(turn("t2", cost=None, asked="2026-11-07T11:00:00.000+00:00"))
    cost = store.summary_metrics()["cost_per_query"]
    assert cost["value"] == 0.0
    assert cost["sample"] == 1, "only turns with a recorded cost count"
    assert cost["unknown_count"] == 1
    assert "not included" in cost["note"]


def test_remote_only_costs_show_unknown_not_zero(tmp_path):
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t1", cost=None))
    cost = store.summary_metrics()["cost_per_query"]
    assert cost["value"] is None, "no invented figure"
    assert cost["sample"] == 0
    assert "not instrumented" in cost["note"]


def test_latency_series_follows_the_recorded_order(tmp_path):
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t-a", duration=700,
                           asked="2026-11-07T10:00:00.000+00:00"))
    store.record_turn(turn("t-b", duration=1500,
                           asked="2026-11-07T10:05:00.000+00:00"))
    report = store.summary_metrics()
    assert [point["ms"] for point in report["latency_series"]] == [700, 1500]
    # both turns fall on one evaluation day, so that day averages both
    assert report["over_time"][0]["avg_latency_ms"] == 1100


def test_accuracy_series_builds_a_running_value_and_ignores_unsure(tmp_path):
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t1", asked="2026-11-07T10:00:00.000+00:00"))
    store.record_turn(turn("t2", asked="2026-11-07T10:01:00.000+00:00"))
    store.record_feedback(grade("t1", correct="yes"))
    store.record_feedback(grade("t2", correct="not_sure"))
    series = store.summary_metrics()["accuracy_series"]
    assert len(series) == 1, "'not sure' is not a graded correctness mark"
    assert series[0]["value"] == 100.0


# --------------------------------------------------------------------- pilot

def test_the_tester_summary_tracks_anonymous_ids(tmp_path):
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t1", tester="Tester 01"))
    store.record_turn(turn("t2", tester="Tester 02",
                           asked="2026-11-07T11:00:00.000+00:00"))
    store.record_turn(turn("t3", asked="2026-11-07T12:00:00.000+00:00"))
    store.record_feedback(grade("t1", helpful=True, correct="yes",
                                tester="Tester 01"))
    store.record_feedback(grade("t2", helpful=False,
                                tester="Tester 02"))
    testers = {entry["tester_id"]: entry for entry in
               store.summary_metrics()["testers"]}
    assert set(testers) == {"Tester 01", "Tester 02", "(not set)"}
    assert testers["Tester 01"]["accuracy"] == 100.0
    assert testers["Tester 02"]["helpful"] == 0
    assert testers["(not set)"]["questions"] == 1


def test_the_suggested_catalog_covers_the_three_routes():
    hints = {entry["route_hint"] for entry in pilot.SUGGESTED_QUESTIONS}
    assert {"SQL", "RAG", "Hybrid"} <= hints
    assert all(entry["id"].startswith("PQ-") for entry in
               pilot.SUGGESTED_QUESTIONS)
    assert len({entry["id"] for entry in pilot.SUGGESTED_QUESTIONS}) == \
        len(pilot.SUGGESTED_QUESTIONS)


# ------------------------------------------------------------------ feedback rules

def test_invalid_feedback_values_are_refused():
    with pytest.raises(ValueError, match="helpful must be"):
        pilot.validate_feedback_fields(3, None, None, None)
    with pytest.raises(ValueError, match="correct must be"):
        pilot.validate_feedback_fields(None, "sure", None, None)
    with pytest.raises(ValueError, match="citations_relevant must be"):
        pilot.validate_feedback_fields(None, None, "irrelevant", None)
    with pytest.raises(ValueError, match="1000"):
        pilot.validate_feedback_fields(None, None, None, "x" * 1001)
    with pytest.raises(ValueError, match="feedback is empty"):
        pilot.validate_feedback_fields(None, None, None, None)


# ------------------------------------------------------------------------ CSV

def test_the_csv_export_has_a_stable_header_and_one_row_per_turn(seeded):
    path = seeded.export_csv()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv_module.reader(handle))
    assert rows[0] == list(pilot.CSV_FIELDS)
    assert len(rows) == 4, "header plus the three seeded turns"
    index_route = list(pilot.CSV_FIELDS).index("route")
    assert {row[index_route] for row in rows[1:]} == {"sql", "rag"}


def test_the_csv_export_folds_feedback_into_the_rows(tmp_path):
    store = pilot.PilotStore(tmp_path)
    store.record_turn(turn("t1", route="rag"))
    store.record_feedback(grade("t1", helpful=True, correct="yes",
                                citations="yes", comment="useful"))
    path = store.export_csv(path=tmp_path / "custom_report.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["helpful"] == "True"
    assert row["correct"] == "yes"
    assert row["citations_relevant"] == "yes"
    assert row["feedback_comment"] == "useful"
    assert row["timestamp"] == "2026-11-07T10:00:00.000+00:00"


def test_an_empty_store_exports_only_the_header(tmp_path):
    path = pilot.PilotStore(tmp_path).export_csv()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

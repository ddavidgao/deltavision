"""
Results store tests. The SQLite DB is the single source of truth for paper
figures — untested code here means paper numbers could silently wrong if
the schema ever shifts or save() starts dropping fields.
"""

import json
import tempfile
from pathlib import Path

import pytest

from results.store import ResultStore


@pytest.fixture
def store():
    """Fresh in-memory-equivalent store per test (uses tmp file, not shared DB)."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.db"
        s = ResultStore(path=path)
        yield s
        s.close()


class TestSave:
    def test_save_returns_id(self, store):
        rid = store.save("reaction", "cv", {"best_ms": 412, "avg_ms": 755})
        assert isinstance(rid, int)
        assert rid > 0

    def test_sequential_ids(self, store):
        r1 = store.save("reaction", "cv", {"best_ms": 412})
        r2 = store.save("reaction", "cv", {"best_ms": 500})
        r3 = store.save("ablation", "claude", {"steps": 3})
        assert r2 == r1 + 1
        assert r3 == r2 + 1

    def test_flattened_metrics_stored(self, store):
        rid = store.save("reaction", "cv", {
            "best_ms": 74,
            "avg_ms": 100,
            "delta_ratio": 0.8,
            "steps": 3,
            "token_cost": 4000,
        })
        row = store.query("SELECT * FROM runs WHERE id=?", (rid,))[0]
        assert row["best_ms"] == 74
        assert row["avg_ms"] == 100
        assert row["delta_ratio"] == 0.8
        assert row["steps"] == 3
        assert row["token_cost"] == 4000

    def test_metrics_json_preserves_all(self, store):
        metrics = {
            "best_ms": 74,
            "custom_field": "extra",
            "nested": {"a": 1, "b": [1, 2, 3]},
        }
        rid = store.save("test", "backend", metrics)
        row = store.query("SELECT metrics_json FROM runs WHERE id=?", (rid,))[0]
        loaded = json.loads(row["metrics_json"])
        assert loaded == metrics

    def test_config_stored(self, store):
        cfg = {"PHASH_DISTANCE_THRESHOLD": 20, "MAX_STEPS": 50}
        rid = store.save("test", "backend", {"steps": 3}, config=cfg)
        row = store.query("SELECT config_json FROM runs WHERE id=?", (rid,))[0]
        assert json.loads(row["config_json"]) == cfg

    def test_transition_log_stored(self, store):
        log = [
            {"step": 0, "transition": "NEW_PAGE"},
            {"step": 1, "transition": "DELTA"},
        ]
        rid = store.save("test", "backend", {"steps": 2}, transition_log=log)
        row = store.query("SELECT transition_log_json FROM runs WHERE id=?", (rid,))[0]
        assert json.loads(row["transition_log_json"]) == log

    def test_legacy_metric_names_accepted(self, store):
        """Older benchmarks used best_reaction_ms / avg_reaction_ms keys."""
        rid = store.save("reaction", "cv", {
            "best_reaction_ms": 412,
            "avg_reaction_ms": 755,
        })
        row = store.query("SELECT best_ms, avg_ms FROM runs WHERE id=?", (rid,))[0]
        assert row["best_ms"] == 412
        assert row["avg_ms"] == 755

    def test_missing_metrics_stored_as_null(self, store):
        rid = store.save("test", "backend", {"some_other_field": "x"})
        row = store.query("SELECT best_ms, avg_ms, steps FROM runs WHERE id=?", (rid,))[0]
        assert row["best_ms"] is None
        assert row["avg_ms"] is None
        assert row["steps"] is None

    def test_notes_stored(self, store):
        rid = store.save("test", "backend", {}, notes="special experiment")
        row = store.query("SELECT notes FROM runs WHERE id=?", (rid,))[0]
        assert row["notes"] == "special experiment"


class TestQuery:
    def test_query_returns_list_of_dicts(self, store):
        store.save("a", "x", {"steps": 1})
        store.save("b", "y", {"steps": 2})
        rows = store.query("SELECT benchmark, backend FROM runs")
        assert len(rows) == 2
        assert all(isinstance(r, dict) for r in rows)
        assert {r["benchmark"] for r in rows} == {"a", "b"}

    def test_query_with_params(self, store):
        store.save("a", "x", {"steps": 1})
        store.save("b", "y", {"steps": 2})
        rows = store.query("SELECT * FROM runs WHERE benchmark=?", ("a",))
        assert len(rows) == 1
        assert rows[0]["benchmark"] == "a"

    def test_empty_query(self, store):
        rows = store.query("SELECT * FROM runs")
        assert rows == []


class TestBest:
    def test_best_returns_lowest_time(self, store):
        store.save("reaction", "cv", {"avg_ms": 500})
        store.save("reaction", "cv", {"avg_ms": 200})
        store.save("reaction", "cv", {"avg_ms": 300})
        best = store.best("reaction")
        assert best is not None
        assert best["avg_ms"] == 200

    def test_best_prefers_best_ms_over_avg(self, store):
        """best_ms is coalesced first — a run with best_ms=100, avg_ms=1000
        should beat a run with only avg_ms=500."""
        store.save("reaction", "cv", {"avg_ms": 500})
        store.save("reaction", "cv", {"best_ms": 100, "avg_ms": 1000})
        best = store.best("reaction")
        assert best["best_ms"] == 100

    def test_best_none_when_no_runs(self, store):
        assert store.best("never-saved") is None

    def test_best_includes_metrics_dict(self, store):
        store.save("x", "cv", {"best_ms": 50, "custom": "field"})
        best = store.best("x")
        assert "metrics" in best
        assert best["metrics"]["custom"] == "field"


class TestSchema:
    def test_tables_exist(self, store):
        tables = store.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {t["name"] for t in tables}
        assert "runs" in names
        assert "comparisons" in names

    def test_indexes_exist(self, store):
        idx = store.query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        names = {i["name"] for i in idx}
        assert "idx_benchmark" in names
        assert "idx_timestamp" in names

    def test_reopen_preserves_data(self, store):
        """Closing and reopening the same file should not lose rows."""
        rid = store.save("persist", "cv", {"best_ms": 123})
        path = store.path
        store.close()
        s2 = ResultStore(path=path)
        rows = s2.query("SELECT * FROM runs WHERE id=?", (rid,))
        s2.close()
        assert len(rows) == 1
        assert rows[0]["best_ms"] == 123

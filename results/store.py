"""
SQLite-backed result store. Single file, SQL-queryable, zero dependencies.

Usage:
    from results.store import ResultStore
    db = ResultStore()
    db.save("reaction", "deltavision_cv", {"best_ms": 412, "avg_ms": 755})
    db.summary()
    db.query("SELECT * FROM runs WHERE benchmark='reaction' ORDER BY timestamp DESC")
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


DB_PATH = Path(__file__).parent / "deltavision.db"


class ResultStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark TEXT NOT NULL,
                backend TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                notes TEXT DEFAULT '',

                -- Flattened key metrics for fast queries
                best_ms REAL,
                avg_ms REAL,
                delta_ratio REAL,
                steps INTEGER,
                token_cost REAL,

                -- Full metrics blob
                metrics_json TEXT NOT NULL,
                config_json TEXT DEFAULT '{}',
                transition_log_json TEXT DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_benchmark ON runs(benchmark);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp);

            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_a INTEGER REFERENCES runs(id),
                run_b INTEGER REFERENCES runs(id),
                improvement_pct REAL,
                notes TEXT
            );
        """)
        self.conn.commit()

    def save(
        self,
        benchmark: str,
        backend: str,
        metrics: dict,
        config: dict = None,
        transition_log: list = None,
        notes: str = "",
    ) -> int:
        """Save a run. Returns the run ID."""
        now = datetime.now().isoformat()

        self.conn.execute(
            """INSERT INTO runs
               (benchmark, backend, timestamp, notes,
                best_ms, avg_ms, delta_ratio, steps, token_cost,
                metrics_json, config_json, transition_log_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                benchmark,
                backend,
                now,
                notes,
                metrics.get("best_ms") or metrics.get("best_reaction_ms"),
                metrics.get("avg_ms") or metrics.get("avg_reaction_ms"),
                metrics.get("delta_ratio"),
                metrics.get("steps"),
                metrics.get("token_cost"),
                json.dumps(metrics),
                json.dumps(config or {}),
                json.dumps(transition_log or []),
            ),
        )
        self.conn.commit()
        rid = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        print(f"  Saved run #{rid}: {benchmark} / {backend} @ {now[:19]}")
        return rid

    def summary(self, benchmark: str = None, limit: int = 20):
        """Print a summary table."""
        where = "WHERE benchmark = ?" if benchmark else ""
        params = (benchmark,) if benchmark else ()

        rows = self.conn.execute(
            f"""SELECT benchmark, backend, timestamp, best_ms, avg_ms,
                       delta_ratio, steps, notes
                FROM runs {where}
                ORDER BY timestamp DESC LIMIT ?""",
            (*params, limit),
        ).fetchall()

        if not rows:
            print("No results yet.")
            return

        print(f"\n{'ID':>3} {'Benchmark':<25} {'Backend':<22} {'Best':>8} {'Avg':>8} {'Delta%':>7} {'Date'}")
        print("-" * 95)
        for r in rows:
            best = f"{r['best_ms']:.0f}ms" if r["best_ms"] else "-"
            avg = f"{r['avg_ms']:.0f}ms" if r["avg_ms"] else "-"
            dr = f"{r['delta_ratio']:.1%}" if r["delta_ratio"] else "-"
            date = r["timestamp"][:10]
            print(f"{'':>3} {r['benchmark']:<25} {r['backend']:<22} {best:>8} {avg:>8} {dr:>7} {date}")

    def best(self, benchmark: str) -> Optional[dict]:
        """Get the best run for a benchmark (lowest avg_ms or best_ms)."""
        row = self.conn.execute(
            """SELECT * FROM runs WHERE benchmark = ?
               ORDER BY COALESCE(best_ms, avg_ms, 999999) ASC LIMIT 1""",
            (benchmark,),
        ).fetchone()
        if row:
            return {**dict(row), "metrics": json.loads(row["metrics_json"])}
        return None

    def query(self, sql: str, params: tuple = ()) -> list:
        """Run arbitrary SQL. Returns list of dicts."""
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()

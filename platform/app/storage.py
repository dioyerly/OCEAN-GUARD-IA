"""Historial y archivos de la plataforma (SQLite de la biblioteca estándar + carpetas)."""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    filename TEXT,
    width INTEGER, height INTEGER,
    slick_pct REAL, n_components INTEGER, slick_area_km2 REAL,
    threshold REAL,
    bounds_json TEXT,
    location_kind TEXT,
    center_lat REAL, center_lon REAL,
    reference_json TEXT,
    result_json TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    params_json TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def data_dir() -> Path:
    return DATA_DIR


def analysis_dir(analysis_id: str) -> Path:
    return DATA_DIR / "analyses" / analysis_id


def _db_path() -> Path:
    return DATA_DIR / "platform.sqlite3"


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_db_path())
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def save_analysis(row: dict) -> None:
    with connect() as con:
        con.execute(
            """INSERT INTO analyses (id, created_at, source, filename, width, height,
               slick_pct, n_components, slick_area_km2, threshold, bounds_json,
               location_kind, center_lat, center_lon, reference_json, result_json, notes)
               VALUES (:id, :created_at, :source, :filename, :width, :height,
               :slick_pct, :n_components, :slick_area_km2, :threshold, :bounds_json,
               :location_kind, :center_lat, :center_lon, :reference_json, :result_json, :notes)""",
            row,
        )
    con.close()


def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["bounds"] = json.loads(d.pop("bounds_json")) if d.get("bounds_json") else None
    d["reference"] = json.loads(d.pop("reference_json")) if d.get("reference_json") else None
    d["result"] = json.loads(d.pop("result_json")) if d.get("result_json") else None
    return d


def list_analyses(limit: int = 200) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    con.close()
    return [_row_to_dict(r) for r in rows]


def get_analysis(analysis_id: str) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    con.close()
    return _row_to_dict(r) if r else None


def delete_analysis(analysis_id: str) -> bool:
    with connect() as con:
        cur = con.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        con.execute("DELETE FROM simulations WHERE analysis_id = ?", (analysis_id,))
        deleted = cur.rowcount > 0
    con.close()
    folder = analysis_dir(analysis_id)
    if folder.exists():
        shutil.rmtree(folder)
    return deleted


def save_simulation(sim_id: str, analysis_id: str, params: dict, result: dict) -> None:
    folder = DATA_DIR / "simulations"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{sim_id}.json").write_text(json.dumps(result), encoding="utf-8")
    with connect() as con:
        con.execute(
            "INSERT INTO simulations (id, analysis_id, created_at, params_json) VALUES (?,?,?,?)",
            (sim_id, analysis_id, _now(), json.dumps(params)),
        )
    con.close()


def dashboard_counts() -> dict:
    with connect() as con:
        total = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        with_slick = con.execute(
            "SELECT COUNT(*) FROM analyses WHERE slick_pct > 0"
        ).fetchone()[0]
        sims = con.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
        last = con.execute(
            "SELECT created_at FROM analyses ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    con.close()
    return {
        "analyses": total,
        "with_slick": with_slick,
        "simulations": sims,
        "last_analysis_at": last[0] if last else None,
    }


def now_iso() -> str:
    return _now()


def latest_simulation(analysis_id: str) -> dict | None:
    """Última simulación guardada de un análisis (o None)."""
    with connect() as con:
        r = con.execute(
            "SELECT id FROM simulations WHERE analysis_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (analysis_id,),
        ).fetchone()
    con.close()
    if not r:
        return None
    path = DATA_DIR / "simulations" / f"{r['id']}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def activity_last_days(days: int = 14) -> list[dict]:
    """Análisis por día (UTC) en los últimos `days` días, incluyendo días sin actividad."""
    from datetime import timedelta
    today = datetime.now(timezone.utc).date()
    with connect() as con:
        rows = con.execute("SELECT substr(created_at,1,10) AS d, COUNT(*) AS n FROM analyses GROUP BY d").fetchall()
    con.close()
    counts = {r["d"]: r["n"] for r in rows}
    out = []
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        out.append({"date": d, "count": counts.get(d, 0)})
    return out


def total_slick_area_km2() -> float:
    with connect() as con:
        v = con.execute("SELECT COALESCE(SUM(slick_area_km2),0) FROM analyses").fetchone()[0]
    con.close()
    return round(float(v), 2)


def update_location(analysis_id: str, bounds: list, kind: str, lat: float, lon: float,
                    area_km2: float | None, result: dict) -> None:
    with connect() as con:
        con.execute(
            "UPDATE analyses SET bounds_json=?, location_kind=?, center_lat=?, center_lon=?, slick_area_km2=?, result_json=? WHERE id=?",
            (json.dumps(bounds), kind, lat, lon, area_km2, json.dumps(result), analysis_id),
        )
    con.close()

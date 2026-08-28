"""Persistência em SQLite: log de predições, atualizações e avaliações.

SQLite foi escolhido em vez de Prometheus por proporção entre valor e
custo. Ele é um arquivo, já vem com o Python, aceita SQL para as
agregações do endpoint `/metrics` e é lido diretamente pelo dashboard.
Prometheus + Grafana resolveriam o mesmo problema com dois containers
a mais e uma linguagem de consulta nova.

Três tabelas:

    predictions   toda predição servida (para volume e distribuição)
    updates       toda atualização incremental (para rastrear versões)
    evaluations   desempenho medido quando os rótulos chegam
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

DB_PATH = Path("data/monitoring.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    model_version INTEGER NOT NULL,
    probability   REAL    NOT NULL,
    prediction    INTEGER NOT NULL,
    threshold     REAL    NOT NULL,
    features      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS updates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    from_version  INTEGER NOT NULL,
    to_version    INTEGER NOT NULL,
    n_samples     INTEGER NOT NULL,
    loss          REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    model_version INTEGER NOT NULL,
    n_samples     INTEGER NOT NULL,
    tp INTEGER NOT NULL, tn INTEGER NOT NULL,
    fp INTEGER NOT NULL, fn INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Abre a conexão, garante o schema e faz commit ao sair."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    with connect(db_path):
        pass


# --------------------------------------------------------------------------
# Escrita
# --------------------------------------------------------------------------

def log_prediction(
    model_version: int,
    probability: float,
    prediction: int,
    threshold: float,
    features: List[float],
    db_path: Path = DB_PATH,
) -> int:
    """Registra uma predição servida e devolve o id gerado."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO predictions "
            "(created_at, model_version, probability, prediction, threshold, features) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), model_version, float(probability), int(prediction),
             float(threshold), json.dumps(features)),
        )
        return int(cur.lastrowid)


def log_update(
    from_version: int, to_version: int, n_samples: int, loss: float,
    db_path: Path = DB_PATH,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO updates (created_at, from_version, to_version, n_samples, loss) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), from_version, to_version, n_samples, float(loss)),
        )


def log_evaluation(
    model_version: int, cm: Dict[str, int], db_path: Path = DB_PATH,
) -> None:
    """Registra o desempenho medido em um lote rotulado."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO evaluations "
            "(created_at, model_version, n_samples, tp, tn, fp, fn) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now(), model_version, sum(cm.values()),
             cm["tp"], cm["tn"], cm["fp"], cm["fn"]),
        )


# --------------------------------------------------------------------------
# Leitura / agregações
# --------------------------------------------------------------------------

def prediction_stats(db_path: Path = DB_PATH) -> Dict:
    """Volume, taxa de positivos e distribuição das probabilidades."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, AVG(probability) mean_prob, "
            "SUM(prediction) n_pos, MIN(created_at) first, MAX(created_at) last "
            "FROM predictions"
        ).fetchone()

        total = row["n"] or 0
        if total == 0:
            return {"total": 0, "positive_rate": None, "mean_probability": None,
                    "histogram": {}, "first_at": None, "last_at": None}

        # Histograma em 10 faixas, para ver se a distribuição das
        # probabilidades muda ao longo do tempo.
        bins = conn.execute(
            "SELECT CAST(probability * 10 AS INTEGER) b, COUNT(*) c "
            "FROM predictions GROUP BY b ORDER BY b"
        ).fetchall()

        return {
            "total": total,
            "positive_rate": round((row["n_pos"] or 0) / total, 4),
            "mean_probability": round(row["mean_prob"], 4),
            "histogram": {
                f"{min(r['b'], 9) / 10:.1f}-{(min(r['b'], 9) + 1) / 10:.1f}": r["c"]
                for r in bins
            },
            "first_at": row["first"],
            "last_at": row["last"],
        }


def update_stats(db_path: Path = DB_PATH) -> Dict:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, SUM(n_samples) samples, MAX(created_at) last "
            "FROM updates"
        ).fetchone()
        return {
            "total": row["n"] or 0,
            "samples_learned": row["samples"] or 0,
            "last_at": row["last"],
        }


def performance_stats(db_path: Path = DB_PATH) -> Optional[Dict]:
    """Desempenho acumulado, somando as matrizes de confusão registradas.

    Só existe se algum lote rotulado já tiver chegado via `/update`.
    """
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT SUM(tp) tp, SUM(tn) tn, SUM(fp) fp, SUM(fn) fn, "
            "SUM(n_samples) n, COUNT(*) batches FROM evaluations"
        ).fetchone()

    if not row["n"]:
        return None

    from app.metrics import compute_metrics

    cm = {k: int(row[k]) for k in ("tp", "tn", "fp", "fn")}
    return {**compute_metrics(cm), **cm,
            "n_labeled": int(row["n"]), "n_batches": int(row["batches"])}


def recent_predictions(limit: int = 200, db_path: Path = DB_PATH) -> List[Dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, created_at, model_version, probability, prediction "
            "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

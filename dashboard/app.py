"""Dashboard de monitoramento.

Lê o mesmo SQLite que a API escreve. Não há acoplamento entre os dois:
o dashboard só consulta, e pode subir ou cair sem afetar o serviço.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB_PATH = Path(os.getenv("DB_PATH", "data/monitoring.db"))
REGISTRY_PATH = Path(os.getenv("REGISTRY_PATH", "models/registry.json"))

st.set_page_config(page_title="Rain Model — Monitoramento", layout="wide")


@st.cache_data(ttl=10)
def load(table: str) -> pd.DataFrame:
    """Carrega uma tabela do banco de monitoramento."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
        except pd.errors.DatabaseError:
            return pd.DataFrame()
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], format="mixed")
    return df


st.title("🌧️ Previsão de chuva — monitoramento")

predictions = load("predictions")
updates = load("updates")
evaluations = load("evaluations")

if predictions.empty:
    st.warning(
        "Nenhuma predição registrada ainda.\n\n"
        "Suba a API e gere tráfego:\n"
        "```\n"
        "uvicorn app.main:app\n"
        "python -m training.simulate_production --in-process\n"
        "```"
    )
    st.stop()

# --------------------------------------------------------------- topo

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predições servidas", f"{len(predictions):,}")
col2.metric("Versão do modelo", int(predictions["model_version"].max()))
col3.metric("Atualizações", len(updates))
col4.metric(
    "Probabilidade média",
    f"{predictions['probability'].mean():.3f}",
)

# --------------------------------------------------- desempenho real

st.subheader("Desempenho medido")

if evaluations.empty:
    st.info(
        "Ainda não chegaram rótulos verdadeiros. O desempenho só pode ser "
        "medido depois que lotes rotulados forem enviados para `/update`."
    )
else:
    total = evaluations[["tp", "tn", "fp", "fn"]].sum()
    tp, tn, fp, fn = (int(total[k]) for k in ("tp", "tn", "fp", "fn"))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1", f"{f1:.3f}")
    c2.metric("Precisão", f"{precision:.3f}")
    c3.metric("Recall", f"{recall:.3f}")
    c4.metric("Rótulos recebidos", f"{int(total.sum()):,}")

    # F1 por lote: mostra se o modelo está degradando com o tempo.
    evaluations = evaluations.sort_values("created_at").copy()
    p = evaluations["tp"] / (evaluations["tp"] + evaluations["fp"]).replace(0, pd.NA)
    r = evaluations["tp"] / (evaluations["tp"] + evaluations["fn"]).replace(0, pd.NA)
    evaluations["f1"] = (2 * p * r / (p + r)).fillna(0.0)

    st.caption("F1 por lote avaliado — medido antes de cada atualização")
    st.line_chart(
        evaluations.set_index("model_version")["f1"],
        height=220,
    )

    with st.expander("Matriz de confusão acumulada"):
        st.dataframe(
            pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["real: sem chuva", "real: chuva"],
                columns=["previsto: sem chuva", "previsto: chuva"],
            ),
            use_container_width=True,
        )

# --------------------------------------------- distribuição das saídas

st.subheader("Distribuição das probabilidades")
st.caption(
    "É o sinal de alerta mais rápido: um deslocamento nesta distribuição "
    "indica mudança nos dados de entrada ou no modelo, sem precisar "
    "esperar pelos rótulos verdadeiros."
)

left, right = st.columns([2, 1])

with left:
    hist = pd.cut(
        predictions["probability"],
        bins=[i / 10 for i in range(11)],
        include_lowest=True,
    ).value_counts().sort_index()
    hist.index = [f"{i.left:.1f}–{i.right:.1f}" for i in hist.index]
    st.bar_chart(hist, height=280)

with right:
    st.metric("Taxa de positivos", f"{predictions['prediction'].mean():.1%}")
    st.metric("Limiar em uso", f"{predictions['threshold'].iloc[-1]:.2f}")
    st.metric(
        "Mediana",
        f"{predictions['probability'].median():.3f}",
    )

# ------------------------------------------------ volume ao longo do tempo

st.subheader("Volume de predições")
por_dia = (
    predictions.set_index("created_at")
    .resample("D")
    .size()
    .rename("predições")
)
st.bar_chart(por_dia, height=200)

# ------------------------------------------------------- versões

st.subheader("Histórico de versões")

if REGISTRY_PATH.exists():
    entries = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_df = pd.DataFrame([
        {
            "versão": e["version"],
            "estágio": e["stage"],
            "amostras": e["n_samples"],
            "F1": e["metrics"].get("f1"),
            "criado em": e["created_at"][:19].replace("T", " "),
            "atual": "✓" if e.get("is_current") else "",
        }
        for e in entries
    ])
    st.dataframe(
        registry_df.sort_values("versão", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Nenhum registro de versões encontrado.")

if not updates.empty:
    with st.expander("Detalhe das atualizações"):
        st.dataframe(
            updates[["created_at", "from_version", "to_version", "n_samples", "loss"]]
            .sort_values("created_at", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

st.caption(
    f"Fonte: {DB_PATH} · atualiza a cada 10s · "
    "dados gravados pela API a cada /predict e /update"
)

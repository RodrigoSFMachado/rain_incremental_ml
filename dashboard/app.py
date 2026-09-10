"""Dashboard de monitoramento (somente leitura).

O que ele lê:

    data/monitoring.db      tabelas predictions, updates e evaluations
    models/registry.json    histórico de versões do modelo

O que ele NÃO faz, de propósito:

    - não carrega models/model.pt nem instancia PyTorch;
    - não chama a API;
    - não escreve nada (no Docker, os volumes são montados read-only).

Por isso não há acoplamento entre os dois serviços: o dashboard pode
subir ou cair sem afetar a API, e vice-versa. O único contrato entre
eles é o arquivo SQLite.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(os.getenv("DB_PATH", "data/monitoring.db"))
REGISTRY_PATH = Path(os.getenv("REGISTRY_PATH", "models/registry.json"))

# Nomes de tabela permitidos. A consulta abaixo interpola o nome
# diretamente porque SQLite não aceita placeholder para identificador;
# restringir a esta lista garante que o valor interpolado nunca venha
# de entrada externa.
TABELAS = ("predictions", "updates", "evaluations")

st.set_page_config(page_title="Rain Model — Monitoramento", layout="wide")


@st.cache_data(ttl=10)
def load(table: str) -> pd.DataFrame:
    """Carrega uma tabela do banco de monitoramento.

    O `ttl=10` é o tempo de vida do cache: depois de 10 segundos, a
    próxima execução do script vai ao banco de novo. Não é atualização
    automática — o Streamlit só reexecuta o script quando há interação
    ou quando a página é recarregada. Deixada a tela parada, os números
    ficam parados junto.
    """
    if table not in TABELAS:
        raise ValueError(f"Tabela não reconhecida: {table}")
    if not DB_PATH.exists():
        return pd.DataFrame()

    with closing(sqlite3.connect(DB_PATH)) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
        except pd.errors.DatabaseError:
            # Banco existe mas a tabela ainda não foi criada: acontece
            # se o dashboard sobe antes do primeiro start da API.
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
# Quatro números de volume. "Versão do modelo" é a maior versão que já
# serviu alguma predição — logo após um /update ela fica uma atrás da
# versão atual do registry, até chegar a próxima predição.

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predições servidas", f"{len(predictions):,}")
col2.metric("Versão que serviu predições", int(predictions["model_version"].max()))
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
        "medido depois que lotes rotulados forem enviados para `/update`. "
        "Predições servidas e rótulos recebidos são coisas diferentes: o "
        "rótulo de uma previsão só existe uma hora depois dela."
    )
else:
    # Soma dos quadrantes e derivação do F1 no fim — não média de F1s
    # por lote, que trataria um lote com 3 chuvas igual a um com 90.
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
    # Cada ponto é medido ANTES da atualização daquele lote, então
    # descreve o desempenho da versão indicada no eixo x sobre dados
    # que ela ainda não tinha visto.
    evaluations = evaluations.sort_values("created_at").copy()
    p = evaluations["tp"] / (evaluations["tp"] + evaluations["fp"]).replace(0, pd.NA)
    r = evaluations["tp"] / (evaluations["tp"] + evaluations["fn"]).replace(0, pd.NA)
    evaluations["f1"] = (2 * p * r / (p + r)).fillna(0.0)

    st.caption(
        "F1 por lote avaliado — cada ponto é medido antes da atualização "
        "correspondente. Lotes pequenos oscilam muito: duas semanas sem "
        "chuva produzem F1 igual a zero por ausência de positivos, não "
        "por falha do modelo."
    )
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
    "esperar pelos rótulos verdadeiros — que chegam em lotes, com atraso."
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
st.caption(
    "Lido de registry.json. Para versões de atualização incremental, o F1 "
    "exibido foi medido no lote recebido ANTES do treino — ou seja, "
    "descreve a versão anterior sobre aqueles dados."
)

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
    f"Fonte: {DB_PATH} · leitura em cache por 10s · "
    "recarregue a página para buscar dados novos · "
    "gravado pela API a cada /predict e /update"
)

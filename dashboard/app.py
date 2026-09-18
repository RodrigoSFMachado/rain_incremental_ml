"""Dashboard de monitoramento somente leitura.

Fontes de dados:

    data/monitoring.db
        Tabelas `predictions`, `updates` e `evaluations`.

    models/registry.json
        Histórico de versões do modelo.

O dashboard não:

    - carrega `models/model.pt` nem instancia o PyTorch;
    - chama a API;
    - grava dados.

No Docker, os volumes são montados como somente leitura. Essa separação
evita o acoplamento entre os serviços: o dashboard pode iniciar ou parar
sem afetar a API, e a API pode fazer o mesmo sem afetar o dashboard.

O único contrato compartilhado entre os dois serviços é o arquivo
SQLite.

Para iniciar:

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

    `ttl=10` define o tempo de vida do cache. Após 10 segundos, a próxima
    execução do script consulta o banco novamente.

    Isso não representa atualização automática: o Streamlit só reexecuta o
    script quando há interação ou quando a página é recarregada. Portanto,
    se a tela permanecer parada, os números também permanecerão inalterados.
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
# Exibe quatro indicadores de volume.
# "Versão do modelo" corresponde à maior versão que já serviu alguma
# predição. Após um `/update`, ela pode permanecer uma versão atrás da
# versão atual no registry até que uma nova predição seja servida.

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
    # Soma os quadrantes das matrizes de confusão e calcula o F1 apenas ao
    # final. Não faz média dos F1 por lote, pois isso daria o mesmo peso a um
    # lote com 3 amostras e a outro com 90.
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

    # F1 por lote: mostra se o desempenho do modelo está se degradando ao
    # longo do tempo.
    # Cada ponto é calculado antes da atualização correspondente. Assim, ele
    # representa o desempenho da versão indicada no eixo x em dados que essa
    # versão ainda não havia utilizado para aprendizado.
    evaluations = evaluations.sort_values("created_at").copy()

    for column in ("tp", "tn", "fp", "fn"):
        evaluations[column] = pd.to_numeric(
            evaluations[column],
            errors="coerce",
        ).fillna(0)

    evaluations["actual_positives"] = (
        evaluations["tp"] + evaluations["fn"]
    )

    evaluations["predicted_positives"] = (
        evaluations["tp"] + evaluations["fp"]
    )

    p_denominator = evaluations["predicted_positives"]
    r_denominator = evaluations["actual_positives"]

    precision_by_batch = evaluations["tp"].div(
        p_denominator.replace(0, pd.NA)
    )

    recall_by_batch = evaluations["tp"].div(
        r_denominator.replace(0, pd.NA)
    )

    f1_denominator = precision_by_batch + recall_by_batch

    evaluations["f1"] = (
        2 * precision_by_batch * recall_by_batch
    ).div(
        f1_denominator.where(f1_denominator.ne(0))
    )

    # O F1 só é avaliável quando há chuva observada ou uma previsão positiva.
    # Se ambos forem zero, não houve evento positivo para avaliar. Mantemos
    # `NaN` para criar uma lacuna no gráfico, em vez de exibir F1 = 0 e
    # sugerir um desempenho nulo.
    evaluations["f1_evaluable"] = (
        evaluations["actual_positives"].gt(0)
        | evaluations["predicted_positives"].gt(0)
    )

    evaluations.loc[
        ~evaluations["f1_evaluable"],
        "f1",
    ] = pd.NA

    st.caption(
    "F1 por lote avaliado — cada ponto é medido antes da atualização "
    "correspondente. Lacunas indicam lotes sem chuva observada e sem "
    "previsões positivas; nesses casos, F1 não é aplicável."
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

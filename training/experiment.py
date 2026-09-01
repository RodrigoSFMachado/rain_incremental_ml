"""Experimento: vale a pena atualizar o modelo ao longo do tempo?

Compara três políticas de atualização sobre a mesma sequência de
janelas temporais, adaptando a metodologia do notebook original:

    Static       treina uma vez na primeira janela e nunca mais muda.
    Retrain      a cada passo, cria um modelo novo e treina do zero
                 com os 180 dias mais recentes.
    Incremental  treina uma vez e depois continua o treinamento com
                 cada lote novo de dados rotulados.

Além dos três, a baseline de persistência ("se está chovendo agora,
vai chover na próxima hora") é avaliada nas mesmas janelas.

Protocolo (mesmo do notebook original):
    - 180 dias de treino
    - 14 dias de teste
    - passo de 14 dias
    - avaliação prequencial: prevê primeiro, aprende depois

Nenhuma janela usa dados do futuro. O rótulo verdadeiro só é
incorporado ao modelo depois que a previsão daquela janela já foi
registrada.

Uso:
    python -m training.experiment
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FEATURE_NAMES, TARGET
from app.metrics import compute_metrics, confusion_summary, roc_auc
from app.model import EPOCHS_INITIAL, EPOCHS_UPDATE, LR_INITIAL, LR_UPDATE, RainModel

TRAIN_DAYS = 180
TEST_DAYS = 14
STEP_DAYS = 14
CALIB_DAYS = 30          # fatia final do treino usada para calibrar o limiar
SEED = 42

SCENARIOS = ("static", "retrain", "incremental")


# --------------------------------------------------------------------------
# Janelas
# --------------------------------------------------------------------------

def create_windows(df: pd.DataFrame) -> List[Dict]:
    """Gera janelas deslizantes de treino/teste sobre o eixo temporal.

    Args:
        df: Dataset com a coluna "valid" ordenada.

    Returns:
        Lista de dicionários com os limites de cada janela.

    Raises:
        ValueError: Se o período não comporta nenhuma janela completa.
    """
    start = df["valid"].min().normalize()
    total_days = (df["valid"].max().normalize() - start).days + 1

    windows: List[Dict] = []
    cursor = 0
    while cursor + TRAIN_DAYS + TEST_DAYS <= total_days:
        train_start = start + pd.Timedelta(days=cursor)
        train_end = train_start + pd.Timedelta(days=TRAIN_DAYS)
        test_end = train_end + pd.Timedelta(days=TEST_DAYS)
        windows.append({
            "window": len(windows),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
        })
        cursor += STEP_DAYS

    if not windows:
        raise ValueError(
            f"Período de {total_days} dias é curto demais para janelas de "
            f"{TRAIN_DAYS}+{TEST_DAYS} dias."
        )
    return windows


def slice_period(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Recorta o intervalo [start, end)."""
    return df[(df["valid"] >= start) & (df["valid"] < end)]


def as_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return (
        df[FEATURE_NAMES].to_numpy(dtype=np.float32),
        df[TARGET].to_numpy(dtype=np.float32),
    )


# --------------------------------------------------------------------------
# Treino de uma janela
# --------------------------------------------------------------------------

def train_fresh(train_df: pd.DataFrame, epochs: int = EPOCHS_INITIAL) -> RainModel:
    """Cria e treina um modelo do zero, calibrando o limiar.

    A calibração usa os últimos `CALIB_DAYS` dias do próprio período de
    treino. É a única informação disponível no momento — usar a janela
    de teste para escolher o limiar seria vazamento.
    """
    model = RainModel(seed=SEED)

    calib_start = train_df["valid"].max() - pd.Timedelta(days=CALIB_DAYS)
    fit_df = train_df[train_df["valid"] < calib_start]
    calib_df = train_df[train_df["valid"] >= calib_start]

    # Se a fatia de calibração ficar degenerada, treina em tudo.
    if len(fit_df) < 100 or calib_df[TARGET].sum() < 5:
        fit_df, calib_df = train_df, train_df

    X, y = as_arrays(fit_df)
    model.fit(X, y, epochs=epochs, lr=LR_INITIAL)

    Xc, yc = as_arrays(calib_df)
    model.calibrate_threshold(Xc, yc)
    return model


# --------------------------------------------------------------------------
# Experimento
# --------------------------------------------------------------------------

def run_experiment(df: pd.DataFrame, epochs: int = EPOCHS_INITIAL) -> pd.DataFrame:
    """Executa os três cenários sobre as mesmas janelas.

    Returns:
        DataFrame com uma linha por (janela, cenário), contendo a matriz
        de confusão e a AUC daquela janela.
    """
    windows = create_windows(df)
    print(f"{len(windows)} janelas de {TEST_DAYS} dias "
          f"({windows[0]['test_start'].date()} -> {windows[-1]['test_end'].date()})\n")

    # Todos os cenários partem do mesmo modelo inicial, treinado na
    # primeira janela. Assim a comparação isola a política de
    # atualização, e não o ponto de partida.
    first = windows[0]
    base_train = slice_period(df, first["train_start"], first["train_end"])

    print("treinando modelo inicial (comum aos três cenários)...")
    t0 = time.time()
    models = {
        "static": train_fresh(base_train, epochs),
        "incremental": train_fresh(base_train, epochs),
    }
    print(f"  pronto em {time.time() - t0:.1f}s  "
          f"(threshold={models['static'].threshold:.2f})\n")

    rows: List[Dict] = []
    t_start = time.time()

    for w in windows:
        test_df = slice_period(df, w["test_start"], w["test_end"])
        if len(test_df) < 24 or test_df[TARGET].sum() == 0:
            continue  # janela sem sinal útil

        X_test, y_test = as_arrays(test_df)

        # ---- Retrain: modelo novo com os 180 dias mais recentes ----
        # Diferente do notebook original, aqui o retrain roda o mesmo
        # número de épocas do treino inicial. Sem isso ele ficaria
        # subtreinado e a vantagem do incremental seria um artefato
        # do número de passos de gradiente, não da política.
        retrain_df = slice_period(df, w["train_start"], w["train_end"])
        models["retrain"] = train_fresh(retrain_df, epochs)

        # ---- avaliação: os três preveem antes de qualquer aprendizado ----
        for name in SCENARIOS:
            model = models[name]
            prob = model.predict_proba(X_test)
            pred = (prob >= model.threshold).astype(int)

            rows.append({
                "window": w["window"],
                "scenario": name,
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "year": w["test_start"].year,
                "n_test": len(test_df),
                "n_positives": int(y_test.sum()),
                "threshold": round(model.threshold, 3),
                "version": model.version,
                "roc_auc": round(roc_auc(y_test, prob), 4),
                **confusion_summary(y_test, pred),
            })

        # ---- baseline de persistência, nas mesmas janelas ----
        rows.append({
            "window": w["window"],
            "scenario": "persistence",
            "test_start": w["test_start"],
            "test_end": w["test_end"],
            "year": w["test_start"].year,
            "n_test": len(test_df),
            "n_positives": int(y_test.sum()),
            "threshold": 0.5,
            "version": 0,
            "roc_auc": round(roc_auc(y_test, test_df["rain_now"].to_numpy(float)), 4),
            **confusion_summary(y_test, test_df["rain_now"].to_numpy(int)),
        })

        # ---- só agora os rótulos "chegam" e o incremental aprende ----
        # O limiar NÃO é recalibrado aqui. O incremental usa exatamente o
        # mesmo limiar do static, e a única diferença entre os dois passa
        # a ser a atualização dos pesos. Sem isso, estaríamos comparando
        # duas coisas ao mesmo tempo e não saberíamos qual delas explica
        # a diferença.
        inc = models["incremental"]
        inc.incremental_fit(X_test, y_test, epochs=EPOCHS_UPDATE, lr=LR_UPDATE)

        if w["window"] % 20 == 0:
            print(f"  janela {w['window']:>3}  {w['test_start'].date()}  "
                  f"versão incremental = {inc.version}")

    print(f"\nconcluído em {time.time() - t_start:.1f}s")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Agregação
# --------------------------------------------------------------------------

def aggregate(results: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Soma as matrizes de confusão e recalcula as métricas.

    Somar os quadrantes e derivar o F1 no fim é diferente de tirar a
    média dos F1 de cada janela: a média trata uma janela com 3 chuvas
    e outra com 90 como se pesassem igual.
    """
    grouped = results.groupby(by, as_index=False)[["tp", "tn", "fp", "fn"]].sum()
    metrics = grouped.apply(
        lambda r: pd.Series(compute_metrics(r.to_dict())), axis=1
    )
    return pd.concat([grouped, metrics], axis=1)


def print_table(df: pd.DataFrame, label: str, key: str = "scenario") -> None:
    print(f"\n{label}")
    print("-" * 60)
    print(f"{key:<14}{'F1':>8}{'Precisão':>10}{'Recall':>9}{'Acurácia':>10}")
    order = ["persistence", "static", "retrain", "incremental"]
    df = df.copy()
    if key == "scenario":
        df["_o"] = df[key].map({s: i for i, s in enumerate(order)})
        df = df.sort_values("_o")
    for _, r in df.iterrows():
        print(f"{str(r[key]):<14}{r['f1']:>8.3f}{r['precision']:>10.3f}"
              f"{r['recall']:>9.3f}{r['accuracy']:>10.3f}")


def make_plots(results: pd.DataFrame, out_dir: Path) -> None:
    """Gera os dois gráficos do README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "static": "#d62728", "retrain": "#ff7f0e",
        "incremental": "#2ca02c", "persistence": "#7f7f7f",
    }

    # --- F1 por janela, suavizado ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, g in results.groupby("scenario"):
        g = g.sort_values("window")
        f1 = g.apply(lambda r: compute_metrics(r.to_dict())["f1"], axis=1)
        ax.plot(g["test_start"], f1.rolling(6, min_periods=1).mean(),
                label=name, color=colors.get(name), linewidth=1.8,
                linestyle="--" if name == "persistence" else "-")
    ax.set_title("F1 por janela de teste (média móvel de 6 janelas ≈ 3 meses)")
    ax.set_ylabel("F1")
    ax.legend(ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "f1_por_janela.png", dpi=130)
    plt.close(fig)

    # --- F1 agregado por ano ---
    by_year = aggregate(results, ["year", "scenario"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    years = sorted(by_year["year"].unique())
    width = 0.2
    for i, name in enumerate(["persistence", "static", "retrain", "incremental"]):
        sub = by_year[by_year["scenario"] == name].set_index("year")
        vals = [sub["f1"].get(y, 0) for y in years]
        ax.bar(np.arange(len(years)) + i * width, vals, width,
               label=name, color=colors.get(name))
    ax.set_xticks(np.arange(len(years)) + 1.5 * width)
    ax.set_xticklabels(years)
    ax.set_title("F1 agregado por ano")
    ax.set_ylabel("F1")
    ax.legend(ncol=4)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "f1_por_ano.png", dpi=130)
    plt.close(fig)

    print(f"gráficos salvos em {out_dir}/")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Static x Retrain x Incremental.")
    parser.add_argument("--data", default="data/dataset.parquet")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--epochs", type=int, default=EPOCHS_INITIAL)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.data).sort_values("valid").reset_index(drop=True)
    results = run_experiment(df, epochs=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)

    overall = aggregate(results, ["scenario"])
    print_table(overall, "RESULTADO AGREGADO (todas as janelas)")

    by_year = aggregate(results, ["scenario", "year"])
    for name in ["static", "retrain", "incremental"]:
        sub = by_year[by_year["scenario"] == name]
        print_table(sub, f"\n{name.upper()} por ano", key="year")

    make_plots(results, out_dir)

    inc_f1 = float(overall.loc[overall.scenario == "incremental", "f1"].iloc[0])
    sta_f1 = float(overall.loc[overall.scenario == "static", "f1"].iloc[0])
    ret_f1 = float(overall.loc[overall.scenario == "retrain", "f1"].iloc[0])
    print("\n" + "=" * 60)
    print(f"incremental vs static : {inc_f1 - sta_f1:+.4f} F1")
    print(f"incremental vs retrain: {inc_f1 - ret_f1:+.4f} F1")
    print("=" * 60)

    if not args.no_mlflow:
        try:
            import mlflow
        except ImportError:
            return
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("rain-prediction")
        with mlflow.start_run(run_name="comparacao_cenarios"):
            mlflow.log_params({
                "train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                "step_days": STEP_DAYS, "epochs": args.epochs,
                "epochs_update": EPOCHS_UPDATE, "lr_update": LR_UPDATE,
                "n_windows": int(results["window"].nunique()),
            })
            for _, r in overall.iterrows():
                for m in ("f1", "precision", "recall", "accuracy"):
                    mlflow.log_metric(f"{r['scenario']}_{m}", r[m])
            mlflow.log_artifact(str(out_dir / "results.csv"))
            for png in out_dir.glob("*.png"):
                mlflow.log_artifact(str(png))
        print("logado no MLflow.")


if __name__ == "__main__":
    main()

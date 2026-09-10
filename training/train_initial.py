"""Treino inicial do modelo, com tracking no MLflow.

Este é o único ponto do projeto onde o modelo é criado do zero. Todas
as atualizações posteriores partem do checkpoint gerado aqui.

Divisão temporal (sem embaralhar — os dados são uma série no tempo):

    treino     : 2021 - 2022   -> aprende os pesos
    validação  : 2023          -> calibra o limiar de decisão
    teste      : 2024 - 2025   -> avaliação final, usada uma única vez

Por que não um split aleatório: embaralhar colocaria observações de
julho de 2025 no treino e de junho de 2025 no teste. Como horas
vizinhas são altamente correlacionadas, o modelo estaria praticamente
consultando a resposta, e a métrica de teste seria otimista demais.

Uso:
    python -m training.train_initial
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # para ver os resultados
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import registry
from app.features import FEATURE_NAMES, TARGET
from app.metrics import evaluate
from app.model import (
    BATCH_SIZE, EPOCHS_INITIAL, HIDDEN_SIZE, LR_INITIAL, SEED, RainModel,
)

MLFLOW_URI = "sqlite:///mlflow.db"

TRAIN_YEARS = (2021, 2022)
VAL_YEAR = 2023
TEST_YEARS = (2024, 2025)


def split(df: pd.DataFrame):
    """Separa o DataFrame em treino, validação e teste por ano."""
    year = df["valid"].dt.year
    train = df[year.isin(TRAIN_YEARS)]
    val = df[year == VAL_YEAR]
    test = df[year.isin(TEST_YEARS)]
    return train, val, test


def as_arrays(df: pd.DataFrame):
    """Extrai X e y no formato esperado pelo modelo.

    A seleção por `FEATURE_NAMES` garante a ordem canônica, mesmo que o
    parquet tenha as colunas em outra sequência.
    """
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y


def persistence_baseline(df: pd.DataFrame) -> dict:
    """Baseline trivial: se está chovendo agora, vai chover na próxima hora.

    Serve para responder à pergunta que todo entrevistador faz: o modelo
    é melhor do que o palpite óbvio? Sem baseline, um F1 de 0,53 não
    significa nada — pode ser excelente ou pior que adivinhar.
    """
    _, y = as_arrays(df)
    return evaluate(y, df["rain_now"].to_numpy(dtype=float), threshold=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino inicial do modelo.")
    parser.add_argument("--data", default="data/dataset.parquet")
    parser.add_argument("--model-out", default="models/model.pt")
    parser.add_argument("--epochs", type=int, default=EPOCHS_INITIAL)
    parser.add_argument("--lr", type=float, default=LR_INITIAL)
    parser.add_argument("--hidden", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Desliga o tracking (útil em CI ou em ambiente sem MLflow).",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset não encontrado: {args.data}\n"
            f"Rode antes: python -m training.prepare_data"
        )

    df = pd.read_parquet(args.data)
    train, val, test = split(df)

    print(f"treino    : {len(train):>6,} obs  ({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]})")
    print(f"validação : {len(val):>6,} obs  ({VAL_YEAR})")
    print(f"teste     : {len(test):>6,} obs  ({TEST_YEARS[0]}-{TEST_YEARS[1]})")

    X_train, y_train = as_arrays(train)
    X_val, y_val = as_arrays(val)
    X_test, y_test = as_arrays(test)

    # ---------------------------------------------------------- treino
    model = RainModel(hidden=args.hidden, lr=args.lr, seed=args.seed)
    loss = model.fit(
        X_train, y_train,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    print(f"\nloss final : {loss:.4f}   pos_weight: {model.pos_weight:.2f}")

    # ------------------------------------------- calibração do limiar
    # Só validação. Calibrar no teste transformaria o teste em treino.
    threshold = model.calibrate_threshold(X_val, y_val)
    print(f"threshold  : {threshold:.2f}  (calibrado em {VAL_YEAR})")

    # ------------------------------------------------------ avaliação
    val_metrics = evaluate(y_val, model.predict_proba(X_val), threshold)
    test_metrics = evaluate(y_test, model.predict_proba(X_test), threshold)
    baseline = persistence_baseline(test)

    print("\n" + "=" * 64)
    print(f"{'':<16}{'F1':>8}{'Precisão':>10}{'Recall':>9}{'AUC':>8}")
    print("-" * 64)
    for name, m in [
        (f"validação {VAL_YEAR}", val_metrics),
        ("teste", test_metrics),
        ("persistência", baseline),
    ]:
        print(f"{name:<16}{m['f1']:>8.3f}{m['precision']:>10.3f}"
              f"{m['recall']:>9.3f}{m['roc_auc']:>8.3f}")
    print("=" * 64)

    ganho = test_metrics["f1"] - baseline["f1"]
    print(f"ganho de F1 sobre a baseline: {ganho:+.3f}")

    # -------------------------------------------------------- artefatos
    path = model.save(args.model_out)
    print(f"\nmodelo salvo em: {path}")

    registry.register(
        version=model.version,
        model_path=str(path),
        stage="initial_training",
        n_samples=len(train),
        metrics={k: test_metrics[k] for k in ("f1", "precision", "recall", "roc_auc")},
        notes=(
            f"Treino inicial {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}. "
            f"Métricas medidas no conjunto de teste "
            f"{TEST_YEARS[0]}-{TEST_YEARS[1]}, com limiar calibrado em {VAL_YEAR}."
        ),
    )
    print(f"registrado em  : {registry.REGISTRY_PATH}")

    # ---------------------------------------------------------- MLflow
    # MLflow entra só aqui, no offline. A API não depende dele em
    # runtime, e por isso ele não está no requirements.txt.
    if not args.no_mlflow:
        try:
            import mlflow
        except ImportError:
            print("\n[aviso] MLflow não instalado; tracking ignorado.")
            return

        # MLflow 3.x colocou o file store (./mlruns) em modo de manutenção.
        # SQLite continua sendo um arquivo local, sem servidor nem infra.
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("rain-prediction")

        with mlflow.start_run(run_name="initial_training"):
            mlflow.log_params({
                "model": "MLP",
                "hidden_size": args.hidden,
                "n_features": len(FEATURE_NAMES),
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "optimizer": "Adam",
                "loss": "BCEWithLogitsLoss",
                "pos_weight": round(model.pos_weight, 2),
                "threshold": threshold,
                "seed": args.seed,
                "train_years": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}",
                "test_years": f"{TEST_YEARS[0]}-{TEST_YEARS[1]}",
                "n_train": len(train),
            })
            mlflow.log_metrics({
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "baseline_f1": baseline["f1"],
                "f1_gain_over_baseline": round(ganho, 4),
                "final_loss": round(loss, 5),
            })
            mlflow.log_artifact(str(path), artifact_path="model")

        print(f"logado no MLflow: {MLFLOW_URI}")
        print("  visualize com: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()

"""Treinamento inicial do modelo com acompanhamento no MLflow.

Este é o único ponto do projeto em que o modelo é criado do zero. Todas
as atualizações posteriores partem do checkpoint gerado por este
treinamento.

A divisão é temporal, sem embaralhamento, porque os dados formam uma
série cronológica. Os anos são configuráveis; os padrões são:

treino
    2012 - 2013 — aprende os pesos.

validação
    2014 — calibra o limiar de decisão.

teste
    2015 - 2025 — avaliação final, utilizada uma única vez.

Os intervalos são inclusivos e precisam estar em ordem, sem sobreposição:
treino < validação < teste. O script recusa configurações que violem isso,
porque qualquer sobreposição vazaria informação do futuro para o treino.

Um split aleatório não seria adequado. Ele poderia colocar observações
de julho de 2025 no treino e de junho de 2025 no teste. Como horas
vizinhas são altamente correlacionadas, o modelo teria acesso indireto à
resposta, produzindo uma métrica de teste otimista demais.

Uso:

# Split padrão (2012-2013 / 2014 / 2015-2025):
python -m training.train_initial

# Outro split temporal:
python -m training.train_initial \
    --train-years 2012 2019 \
    --val-year 2020 \
    --test-years 2021 2025

Para visualizar os resultados no MLflow:

mlflow ui --backend-store-uri sqlite:///mlflow.db
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
    BATCH_SIZE,
    EPOCHS_INITIAL,
    HIDDEN_SIZE,
    LR_INITIAL,
    SEED,
    RainModel,
)

MLFLOW_URI = "sqlite:///mlflow.db"

# Split padrão. Pode ser sobrescrito por --train-years, --val-year
# e --test-years. Os intervalos são inclusivos: (2012, 2013) = 2012 e 2013.
TRAIN_YEARS = (2012, 2013)
VAL_YEAR = 2014
TEST_YEARS = (2015, 2025)


def validate_years(train_years, val_year, test_years) -> None:
    """Garante que treino < validação < teste, sem sobreposição.

    Levanta SystemExit com uma mensagem clara em vez de treinar com um
    split que vazaria informação do futuro.
    """
    t0, t1 = train_years
    s0, s1 = test_years

    if t0 > t1 or s0 > s1:
        raise SystemExit(
            "Intervalo inválido: o ano inicial deve ser menor ou igual ao final."
        )

    if not (t1 < val_year < s0):
        raise SystemExit(
            f"Split inválido: treino {t0}-{t1}, validação {val_year}, "
            f"teste {s0}-{s1}. É preciso que treino < validação < teste."
        )


def split(
    df: pd.DataFrame,
    train_years=TRAIN_YEARS,
    val_year: int = VAL_YEAR,
    test_years=TEST_YEARS,
):
    """Separa o DataFrame em treino, validação e teste por ano."""
    year = df["valid"].dt.year

    train = df[year.between(*train_years)]
    val = df[year == val_year]
    test = df[year.between(*test_years)]

    return train, val, test


def as_arrays(df: pd.DataFrame):
    """Extrai X e y no formato esperado pelo modelo.

    A seleção por FEATURE_NAMES garante a ordem canônica das features,
    mesmo que as colunas do arquivo Parquet estejam organizadas em outra
    sequência.
    """
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)

    return X, y


def persistence_baseline(df: pd.DataFrame) -> dict:
    """Baseline trivial: se está chovendo agora, prevê chuva na próxima hora.

    Serve para responder à pergunta comum em entrevistas: o modelo é melhor
    do que o palpite óbvio?

    Sem um baseline, um F1 de 0,53 não é interpretável: pode representar
    um resultado excelente ou ser pior do que simplesmente repetir o estado
    atual.
    """
    _, y = as_arrays(df)

    return evaluate(
        y,
        df["rain_now"].to_numpy(dtype=float),
        threshold=0.5,
    )


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
        "--train-years",
        type=int,
        nargs=2,
        default=list(TRAIN_YEARS),
        metavar=("INICIO", "FIM"),
        help=(
            "Anos de treino, inclusivos. "
            f"Padrão: {TRAIN_YEARS[0]} {TRAIN_YEARS[1]}."
        ),
    )

    parser.add_argument(
        "--val-year",
        type=int,
        default=VAL_YEAR,
        help=(
            "Ano de validação usado para calibrar o limiar. "
            f"Padrão: {VAL_YEAR}."
        ),
    )

    parser.add_argument(
        "--test-years",
        type=int,
        nargs=2,
        default=list(TEST_YEARS),
        metavar=("INICIO", "FIM"),
        help=(
            "Anos de teste, inclusivos. "
            f"Padrão: {TEST_YEARS[0]} {TEST_YEARS[1]}."
        ),
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Desliga o tracking; útil em CI ou em ambiente sem MLflow.",
    )

    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset não encontrado: {args.data}\n"
            "Rode antes: python -m training.prepare_data"
        )

    train_years = tuple(args.train_years)
    val_year = args.val_year
    test_years = tuple(args.test_years)

    validate_years(train_years, val_year, test_years)

    df = pd.read_parquet(args.data)
    train, val, test = split(df, train_years, val_year, test_years)

    print(
        f"treino    : {len(train):>6,} obs  "
        f"({train_years[0]}-{train_years[1]})"
    )
    print(f"validação : {len(val):>6,} obs  ({val_year})")
    print(
        f"teste     : {len(test):>6,} obs  "
        f"({test_years[0]}-{test_years[1]})"
    )

    empty = [
        name
        for name, subset in (
            ("treino", train),
            ("validação", val),
            ("teste", test),
        )
        if subset.empty
    ]

    if empty:
        raise SystemExit(
            f"Sem dados em: {', '.join(empty)}. Confira os anos informados e "
            "o --start-year usado no prepare_data "
            f"(dataset vai de {df['valid'].dt.year.min()} "
            f"a {df['valid'].dt.year.max()})."
        )

    X_train, y_train = as_arrays(train)
    X_val, y_val = as_arrays(val)
    X_test, y_test = as_arrays(test)

    # ---------------------------------------------------------- treino
    model = RainModel(hidden=args.hidden, lr=args.lr, seed=args.seed)

    loss = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print(f"\nloss final : {loss:.4f}   pos_weight: {model.pos_weight:.2f}")

    # ------------------------------------------- calibração do limiar
    # Só validação. Calibrar no teste transformaria o teste em treino.
    threshold = model.calibrate_threshold(X_val, y_val)

    print(f"threshold  : {threshold:.2f}  (calibrado em {val_year})")

    # ------------------------------------------------------ avaliação
    val_metrics = evaluate(y_val, model.predict_proba(X_val), threshold)
    test_metrics = evaluate(y_test, model.predict_proba(X_test), threshold)
    baseline = persistence_baseline(test)

    print("\n" + "=" * 64)
    print(f"{'':<16}{'F1':>8}{'Precisão':>10}{'Recall':>9}{'AUC':>8}")
    print("-" * 64)

    for name, metrics in [
        (f"validação {val_year}", val_metrics),
        ("teste", test_metrics),
        ("persistência", baseline),
    ]:
        print(
            f"{name:<16}{metrics['f1']:>8.3f}"
            f"{metrics['precision']:>10.3f}"
            f"{metrics['recall']:>9.3f}"
            f"{metrics['roc_auc']:>8.3f}"
        )

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
        metrics={
            key: test_metrics[key]
            for key in ("f1", "precision", "recall", "roc_auc")
        },
        notes=(
            f"Treino inicial {train_years[0]}-{train_years[1]}. "
            f"Métricas medidas no conjunto de teste "
            f"{test_years[0]}-{test_years[1]}, "
            f"com limiar calibrado em {val_year}."
        ),
    )

    print(f"registrado em  : {registry.REGISTRY_PATH}")

    # ---------------------------------------------------------- MLflow
    # MLflow entra só aqui, no offline. A API não depende dele em runtime,
    # e por isso ele não está no requirements.txt.
    if args.no_mlflow:
        return

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
        mlflow.log_params(
            {
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
                "train_years": f"{train_years[0]}-{train_years[1]}",
                "val_year": val_year,
                "test_years": f"{test_years[0]}-{test_years[1]}",
                "n_train": len(train),
            }
        )

        mlflow.log_metrics(
            {
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "baseline_f1": baseline["f1"],
                "f1_gain_over_baseline": round(ganho, 4),
                "final_loss": round(loss, 5),
            }
        )

        mlflow.log_artifact(str(path), artifact_path="model")

    print(f"logado no MLflow: {MLFLOW_URI}")
    print("  visualize com: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
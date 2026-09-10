"""Simula o uso do serviço em produção, replayando dados reais.

Percorre um período do dataset hora a hora, chamando `/predict` como um
cliente faria, e a cada N dias envia o lote rotulado para `/update` —
reproduzindo o atraso real com que os rótulos ficam disponíveis: só se
sabe se choveu depois que a hora passou.

Serve para dois propósitos: popular o banco de monitoramento com dados
realistas para o dashboard, e demonstrar o ciclo completo do sistema.

ATENÇÃO — este script MODIFICA artefatos do projeto:

    models/model.pt        sobrescrito a cada /update (versão avança)
    models/registry.json   ganha uma entrada por atualização
    data/monitoring.db     ganha predições, updates e avaliações

Se você pretende commitar o modelo do treino inicial, rode
`python -m training.train_initial` depois da simulação para restaurar o
checkpoint v1, ou use `--no-update` para apenas gerar predições.

Uso:
    # contra uma API já rodando:
    uvicorn app.main:app &
    python -m training.simulate_production --days 120

    # ou sem subir servidor nenhum:
    python -m training.simulate_production --days 120 --in-process
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.constants import FEATURE_NAMES, TARGET

UPDATE_EVERY_DAYS = 14


def to_payload(row: pd.Series) -> dict:
    """Converte uma linha do dataset no corpo esperado por `/predict`.

    Usa `FEATURE_NAMES` de `app/constants.py`, a mesma lista que o
    schema da API usa para montar o vetor. Uma lista só, um contrato só.
    """
    return {name: float(row[name]) for name in FEATURE_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula tráfego de produção.")
    parser.add_argument("--data", default="data/dataset.parquet")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument(
        "--no-update", action="store_true",
        help="Só faz predições, sem atualizar o modelo.",
    )
    parser.add_argument(
        "--in-process", action="store_true",
        help="Roda a API no mesmo processo, sem precisar de servidor.",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset não encontrado: {args.data}\n"
            f"Rode antes: python -m training.prepare_data"
        )

    df = pd.read_parquet(args.data).sort_values("valid")
    start = pd.Timestamp(args.start)
    end = start + pd.Timedelta(days=args.days)
    period = df[(df["valid"] >= start) & (df["valid"] < end)]

    if period.empty:
        raise SystemExit(f"Nenhum dado entre {start.date()} e {end.date()}.")

    if not args.no_update:
        print("[aviso] esta execução vai sobrescrever models/model.pt "
              "e adicionar entradas em models/registry.json.\n")

    if args.in_process:
        # TestClient sobe a app em memória: útil para demonstrar o ciclo
        # sem depender de um servidor rodando em outro terminal.
        from fastapi.testclient import TestClient
        from app.main import app as fastapi_app
        context = TestClient(fastapi_app)
    else:
        context = httpx.Client(base_url=args.api_url, timeout=30)

    with context as client:
        run(client, period, start, args)


def run(client, period: pd.DataFrame, start: pd.Timestamp, args) -> None:
    """Executa o replay contra o cliente HTTP fornecido."""
    try:
        health = client.get("/health").json()
    except httpx.ConnectError:
        raise SystemExit(
            f"API não respondeu em {args.api_url}.\n"
            f"Suba o serviço primeiro:  uvicorn app.main:app\n"
            f"Ou rode com --in-process."
        )

    if not health.get("model_loaded"):
        raise SystemExit(
            "A API subiu, mas sem modelo carregado. "
            "Rode antes: python -m training.train_initial"
        )

    print(f"API ok — modelo versão {health['model_version']}")
    print(f"Simulando {len(period):,} horas "
          f"({period['valid'].min().date()} -> {period['valid'].max().date()})\n")

    buffer: list[dict] = []
    next_update = start + pd.Timedelta(days=UPDATE_EVERY_DAYS)
    n_pred = 0

    for _, row in period.iterrows():
        features = to_payload(row)

        response = client.post("/predict", json=features)
        response.raise_for_status()
        n_pred += 1

        # O rótulo verdadeiro só é conhecido depois; guarda para o lote.
        # É o que torna a simulação realista: em produção nenhum sistema
        # sabe se choveu no instante em que faz a previsão.
        buffer.append({"features": features, "target": int(row[TARGET])})

        if not args.no_update and row["valid"] >= next_update and buffer:
            result = client.post("/update", json={"observations": buffer}).json()
            before = result["metrics_before_update"]
            print(
                f"  {row['valid'].date()}  update com {len(buffer):>3} obs  "
                f"v{result['previous_version']} -> v{result['new_version']}  "
                f"F1 no lote: {before['f1']:.3f}"
            )
            buffer.clear()
            next_update += pd.Timedelta(days=UPDATE_EVERY_DAYS)

    metrics = client.get("/metrics").json()
    print(f"\n{n_pred:,} predições servidas")
    print(f"versão final do modelo: {metrics['model']['version']}")
    print(f"probabilidade média   : {metrics['predictions']['mean_probability']}")

    if metrics["performance"]:
        p = metrics["performance"]
        print(f"desempenho acumulado  : F1={p['f1']:.3f}  "
              f"precisão={p['precision']:.3f}  recall={p['recall']:.3f}  "
              f"({p['n_labeled']:,} rótulos)")


if __name__ == "__main__":
    main()

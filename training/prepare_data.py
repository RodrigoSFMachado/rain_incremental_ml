"""Transforma o CSV bruto do ASOS no dataset pronto para modelagem.

Uso:
    python -m training.prepare_data \
        --input data/MIA_2012_2025.csv \
        --output data/dataset.parquet \
        --start-year 2021
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FEATURE_NAMES, TARGET, clean_asos, build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o dataset ASOS.")
    parser.add_argument("--input", default="data/MIA_2012_2025.csv")
    parser.add_argument("--output", default="data/dataset.parquet")
    parser.add_argument(
        "--start-year", type=int, default=2021,
        help="Mantém apenas observações a partir deste ano.",
    )
    args = parser.parse_args()

    print(f"Lendo {args.input} ...")
    raw = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(raw):,} linhas brutas")

    clean = clean_asos(raw)
    print(f"  {len(clean):,} linhas após filtrar METAR de rotina")

    df = build_features(clean)
    print(f"  {len(df):,} linhas com features completas")

    df = df[df["valid"].dt.year >= args.start_year].reset_index(drop=True)
    print(f"  {len(df):,} linhas a partir de {args.start_year}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    # ---------------- relatório de qualidade ----------------
    print("\n" + "=" * 62)
    print("RELATÓRIO DO DATASET")
    print("=" * 62)
    print(f"Período      : {df['valid'].min()}  ->  {df['valid'].max()}")
    print(f"Observações  : {len(df):,}")
    print(f"Features     : {len(FEATURE_NAMES)}")
    print(f"Taxa positiva: {df[TARGET].mean():.2%} (chove na próxima hora)")
    print(f"Desbalanço   : 1 positivo para cada "
          f"{(1 - df[TARGET].mean()) / df[TARGET].mean():.1f} negativos")

    print("\nObservações por ano:")
    by_year = df.groupby(df["valid"].dt.year).agg(
        n=(TARGET, "size"), taxa_chuva=(TARGET, "mean")
    )
    for year, row in by_year.iterrows():
        print(f"  {year}:  {int(row['n']):>5,} obs   {row['taxa_chuva']:.2%}")

    print(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()

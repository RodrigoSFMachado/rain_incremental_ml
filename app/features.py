"""Preparação de dados e engenharia de features.

Este módulo é compartilhado entre o treino (offline) e a API (online).
Isso garante que a mesma transformação aplicada no treino seja aplicada
na predição, evitando training/serving skew.

Fluxo:
    CSV bruto ASOS -> clean_asos() -> build_features() -> DataFrame pronto
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constantes do problema
# --------------------------------------------------------------------------

STATION: str = "MIA"
TARGET_RAW: str = "p01i"
TARGET: str = "rain_next_hour"

# Colunas brutas lidas do CSV ASOS.
RAW_COLUMNS: List[str] = [
    "station", "valid", "tmpf", "dwpf", "relh",
    "drct", "sknt", "p01i", "mslp", "vsby",
]

# Ordem canônica das features do modelo.
# A API valida a entrada contra esta lista; a ordem importa porque o
# scaler e a primeira camada da rede dependem dela.
FEATURE_NAMES: List[str] = [
    "tmpf",            # temperatura (F)
    "dwpf",            # ponto de orvalho (F)
    "relh",            # umidade relativa (%)
    "mslp",            # pressão ao nível do mar (hPa)
    "vsby",            # visibilidade (milhas)
    "wind_speed",      # velocidade do vento (nós)
    "wind_u",          # componente leste-oeste do vento
    "wind_v",          # componente norte-sul do vento
    "dew_spread",      # tmpf - dwpf (proximidade da saturação)
    "mslp_delta_3h",   # tendência barométrica nas últimas 3h
    "relh_delta_1h",   # variação da umidade na última hora
    "rain_now",        # está chovendo na hora atual (0/1)
    "hour_sin",        # ciclo diurno
    "hour_cos",
    "month_sin",       # sazonalidade anual
    "month_cos",
]

N_FEATURES: int = len(FEATURE_NAMES)

# Limiar em polegadas para considerar que houve chuva na hora.
RAIN_THRESHOLD_INCHES: float = 0.01


# --------------------------------------------------------------------------
# Limpeza
# --------------------------------------------------------------------------

def clean_asos(raw: pd.DataFrame, station: str = STATION) -> pd.DataFrame:
    """Limpa o CSV bruto do ASOS e devolve uma série horária regular.

    Args:
        raw: DataFrame lido diretamente do CSV do IEM/ASOS.
        station: Código da estação de interesse.

    Returns:
        DataFrame ordenado no tempo, uma linha por hora, com as colunas
        numéricas já convertidas.

    Raises:
        ValueError: Se nada sobrar após a limpeza.

    Notes:
        Mantém apenas os METAR de rotina (minuto 53). Os relatórios
        especiais (SPECI) são emitidos justamente quando o tempo muda,
        o que tornaria a frequência das observações correlacionada com
        o alvo. Eles também não trazem a pressão ao nível do mar.
    """
    df = raw.copy()
    df = df[df["station"] == station].copy()

    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df = df.dropna(subset=["valid"])

    # Mantém somente os relatórios horários de rotina.
    df = df[df["valid"].dt.minute == 53].copy()

    # "M" = ausente. "T" = traço de chuva (mensurável, mas < 0.01").
    numeric_cols = ["tmpf", "dwpf", "relh", "drct", "sknt", "mslp", "vsby"]
    for col in numeric_cols:
        df[col] = (
            df[col].astype(str).str.strip()
            .replace({"M": np.nan, "": np.nan, "VRB": np.nan, "999": np.nan})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_RAW] = (
        df[TARGET_RAW].astype(str).str.strip()
        .replace({"M": np.nan, "": np.nan, "T": "0.00"})
    )
    df[TARGET_RAW] = pd.to_numeric(df[TARGET_RAW], errors="coerce")

    df = df.sort_values("valid").reset_index(drop=True)
    df = df[RAW_COLUMNS].copy()

    if df.empty:
        raise ValueError(f"Nenhum registro válido para a estação '{station}'.")

    return df


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def build_features(df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """Constrói as features do modelo a partir dos dados limpos.

    Args:
        df: Saída de `clean_asos`, ordenada no tempo.
        with_target: Se True, cria a coluna alvo `rain_next_hour`.

    Returns:
        DataFrame com "valid", as colunas de FEATURE_NAMES e (opcionalmente)
        o alvo, sem valores ausentes.

    Notes:
        O alvo é deslocado em uma hora: as features do instante t são
        usadas para prever a chuva no intervalo t -> t+1. Isso torna o
        problema uma previsão de fato, e não um diagnóstico da hora
        corrente.
    """
    out = df.copy().sort_values("valid").reset_index(drop=True)

    # --- vento: direção é circular, então decompomos em componentes ---
    # Quando a direção é variável (ausente no CSV), zeramos as componentes
    # mas preservamos a velocidade, que continua informativa.
    rad = np.deg2rad(out["drct"])
    out["wind_speed"] = out["sknt"]
    out["wind_u"] = (-out["sknt"] * np.sin(rad)).fillna(0.0)
    out["wind_v"] = (-out["sknt"] * np.cos(rad)).fillna(0.0)

    # --- estado da atmosfera ---
    out["dew_spread"] = out["tmpf"] - out["dwpf"]

    # --- tendências de curto prazo ---
    out["mslp_delta_3h"] = out["mslp"] - out["mslp"].shift(3)
    out["relh_delta_1h"] = out["relh"] - out["relh"].shift(1)

    # --- persistência: já está chovendo agora? ---
    out["rain_now"] = (out[TARGET_RAW] >= RAIN_THRESHOLD_INCHES).astype(float)

    # --- tempo (cíclico) ---
    hour = out["valid"].dt.hour
    month = out["valid"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    keep = ["valid", *FEATURE_NAMES]

    if with_target:
        # p01i no instante t+1 mede a chuva acumulada entre t e t+1.
        out[TARGET] = (
            out[TARGET_RAW].shift(-1) >= RAIN_THRESHOLD_INCHES
        ).astype(float)
        # Só é válido se a próxima observação for de fato 1 hora depois.
        gap_ok = out["valid"].shift(-1) - out["valid"] == pd.Timedelta(hours=1)
        out.loc[~gap_ok, TARGET] = np.nan
        keep.append(TARGET)

    out = out[keep].dropna().reset_index(drop=True)
    return out


def prepare(raw: pd.DataFrame, station: str = STATION) -> pd.DataFrame:
    """Atalho: limpeza + features em uma chamada."""
    return build_features(clean_asos(raw, station=station))

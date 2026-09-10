"""Limpeza dos dados brutos e engenharia de features.

Este módulo é compartilhado entre o treino (offline) e a API (online).
Isso garante que a mesma transformação aplicada no treino seja aplicada
na predição, evitando training/serving skew.

Fluxo:
    CSV bruto ASOS -> clean_asos() -> build_features() -> DataFrame pronto

As constantes do domínio (nomes e ordem das features, alvo, limiar de
chuva) vivem em `app/constants.py` e são apenas reexportadas aqui, para
que `from app.features import FEATURE_NAMES` continue funcionando nos
arquivos que já usavam esse caminho. A lista existe em um lugar só.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.constants import (
    FEATURE_NAMES,
    N_FEATURES,
    RAIN_THRESHOLD_INCHES,
    RAW_COLUMNS,
    STATION,
    TARGET,
    TARGET_RAW,
)

__all__ = [
    "FEATURE_NAMES", "N_FEATURES", "RAIN_THRESHOLD_INCHES", "RAW_COLUMNS",
    "STATION", "TARGET", "TARGET_RAW",
    "clean_asos", "build_features", "prepare",
]


# --------------------------------------------------------------------------
# Limpeza
# --------------------------------------------------------------------------

def clean_asos(raw: pd.DataFrame, station: str = STATION) -> pd.DataFrame:
    """Limpa o CSV bruto do ASOS e devolve as observações horárias de rotina.

    Args:
        raw: DataFrame lido diretamente do CSV do IEM/ASOS.
        station: Código da estação de interesse.

    Returns:
        DataFrame ordenado no tempo, com as colunas de `RAW_COLUMNS` já
        convertidas para numérico.

    Raises:
        ValueError: Se nada sobrar após a limpeza.

    Notes:
        Mantém apenas os METAR de rotina (minuto 53). Os relatórios
        especiais (SPECI) são emitidos justamente quando o tempo muda,
        o que tornaria a *frequência* das observações correlacionada com
        o alvo. Eles também não trazem a pressão ao nível do mar.

        O filtro `minute == 53` é específico da estação MIA. Outras
        estações publicam o relatório de rotina em outro minuto; trocar
        de estação exige conferir esse valor, ou o resultado será um
        DataFrame vazio.

        Atenção ao que esta função NÃO faz: ela não reindexa a série
        nem preenche horas ausentes. Se a estação não reportou às
        14:53, aquela linha simplesmente não existe. Depois do filtro,
        cerca de 99,9% dos intervalos consecutivos são de exatamente 1
        hora — o restante é tratado explicitamente no alvo (ver
        `build_features`), mas não nos lags.
    """
    df = raw.copy()
    df = df[df["station"] == station].copy()

    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df = df.dropna(subset=["valid"])

    # Mantém somente os relatórios horários de rotina.
    df = df[df["valid"].dt.minute == 53].copy()

    # "M" = ausente. "VRB" = direção variável. "T" = traço de chuva
    # (mensurável, mas abaixo de 0.01").
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
        raise ValueError(
            f"Nenhum registro válido para a estação '{station}'. "
            f"Verifique o código da estação e o filtro de minuto 53."
        )

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

        Limitação conhecida: os lags (`mslp_delta_3h`, `relh_delta_1h`)
        usam `shift`, que conta *linhas*, não horas. Nos poucos pontos
        em que há buraco na série, "3 linhas atrás" não é "3 horas
        atrás". O alvo é protegido contra isso pela checagem de gap
        abaixo; os lags não são. Como os buracos são cerca de 0,1% das
        observações, o efeito foi considerado desprezível e o
        comportamento mantido — mas ele existe.
    """
    out = df.copy().sort_values("valid").reset_index(drop=True)

    # --- vento: direção é circular, então decompomos em componentes ---
    # 359 graus e 1 grau são vizinhos no céu, mas ficam nos extremos
    # opostos de uma escala numérica. Seno e cosseno resolvem isso.
    # Quando a direção é variável (ausente no CSV), zeramos as componentes
    # mas preservamos a velocidade, que continua informativa.
    rad = np.deg2rad(out["drct"])
    out["wind_speed"] = out["sknt"]
    out["wind_u"] = (-out["sknt"] * np.sin(rad)).fillna(0.0)
    out["wind_v"] = (-out["sknt"] * np.cos(rad)).fillna(0.0)

    # --- estado da atmosfera ---
    # Quanto menor o dew_spread, mais próximo o ar está da saturação.
    out["dew_spread"] = out["tmpf"] - out["dwpf"]

    # --- tendências de curto prazo ---
    # O nível de pressão importa menos que a direção em que ela se move:
    # queda barométrica costuma anteceder chuva.
    out["mslp_delta_3h"] = out["mslp"] - out["mslp"].shift(3)
    out["relh_delta_1h"] = out["relh"] - out["relh"].shift(1)

    # --- persistência: já está chovendo agora? ---
    # É também a feature usada pela baseline de persistência.
    out["rain_now"] = (out[TARGET_RAW] >= RAIN_THRESHOLD_INCHES).astype(float)

    # --- tempo (cíclico) ---
    # Hora 23 e hora 0 são adjacentes; mês 12 e mês 1 também. Sem
    # seno/cosseno, a rede veria uma descontinuidade artificial.
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
        # Sem esta checagem, um buraco de 6 horas na série viraria um
        # rótulo que descreve outro momento do dia.
        gap_ok = out["valid"].shift(-1) - out["valid"] == pd.Timedelta(hours=1)
        out.loc[~gap_ok, TARGET] = np.nan
        keep.append(TARGET)

    out = out[keep].dropna().reset_index(drop=True)
    return out


def prepare(raw: pd.DataFrame, station: str = STATION) -> pd.DataFrame:
    """Atalho: limpeza + features em uma chamada."""
    return build_features(clean_asos(raw, station=station))

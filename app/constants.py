"""Constantes do domínio, compartilhadas por treino e serving.

Este módulo existe separado de `features.py` por um motivo prático: a
API precisa conhecer o contrato das features (nomes e ordem), mas não
precisa do pandas, que só é usado na preparação dos dados.

Mantendo as constantes aqui, a imagem Docker da API não carrega pandas
nem pyarrow. São cerca de 50 MB de RAM a menos no processo — o que
importa quando o alvo é um contêiner de 512 MB.
"""

from __future__ import annotations

from typing import List

STATION: str = "MIA"
TARGET_RAW: str = "p01i"
TARGET: str = "rain_next_hour"

# Limiar em polegadas para considerar que houve chuva na hora.
RAIN_THRESHOLD_INCHES: float = 0.01

# Ordem canônica das features do modelo.
#
# A ordem importa: o scaler e a primeira camada da rede dependem dela.
# Tanto o treino quanto a API derivam a ordem desta lista, então as
# duas nunca divergem.
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

# Colunas brutas lidas do CSV ASOS.
RAW_COLUMNS: List[str] = [
    "station", "valid", "tmpf", "dwpf", "relh",
    "drct", "sknt", "p01i", "mslp", "vsby",
]

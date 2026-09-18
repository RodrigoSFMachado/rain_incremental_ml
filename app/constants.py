"""Constantes de domínio compartilhadas pelo treino, pela API e pela simulação.

Define a fonte única para os nomes e a ordem das 16 features do modelo,
garantindo que scaler, rede neural, API e scripts de treino usem o mesmo contrato.
"""

from __future__ import annotations

from typing import List

STATION: str = "MIA"

# Coluna bruta do CSV ASOS com a precipitação acumulada na hora, em polegadas.
TARGET_RAW: str = "p01i"

# Variável-alvo: indica se choveu na hora seguinte (0/1).
TARGET: str = "rain_next_hour"

# Limiar em polegadas para considerar que houve chuva na hora.
# 0.01" é o menor valor mensurável do ASOS; abaixo disso o relatório
# traz "T" (trace), tratado como 0.00 na limpeza.
RAIN_THRESHOLD_INCHES: float = 0.01

# Ordem das features do modelo. NÃO reordene nem insira no
# meio: um checkpoint já treinado deixaria de ser compatível, porque o
# scaler e os pesos da primeira camada foram aprendidos nesta ordem.
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

# Colunas brutas lidas do CSV ASOS, na ordem em que o pipeline as usa.
# Correspondem às variáveis selecionadas no download do IEM (ver README).
RAW_COLUMNS: List[str] = [
    "station", "valid", "tmpf", "dwpf", "relh",
    "drct", "sknt", "p01i", "mslp", "vsby",
]

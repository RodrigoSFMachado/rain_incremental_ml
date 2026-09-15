"""Constantes de domínio compartilhadas por treino, API e simulação.

Este módulo é a **fonte única** dos nomes e da ordem das 16 features.
`app/features.py`, `app/model.py`, `app/schemas.py` e os scripts de
`training/` derivam tudo daqui — nenhum deles redeclara a lista.

Por que uma fonte única importa aqui mais do que de costume:

A ordem das features é um contrato implícito entre três lugares — o
scaler congelado, a primeira camada da rede e o corpo JSON do
`/predict`. Se duas cópias da lista divergirem, a API continua
aceitando a requisição (os campos são nomeados e todos existem),
monta o vetor na ordem errada, normaliza cada valor pela média e pelo
desvio de outra coluna e devolve uma probabilidade plausível.

Não há exceção, não há log, o teste continua verde. Manter uma lista
só elimina a classe inteira de bug.

Nota sobre dependências: este módulo não importa pandas, então quem
precisa apenas do contrato das features paga menos por isso aqui. Mas
isso não torna a API livre de pandas — `app/features.py` o importa e é
carregado em runtime através do `app/schemas.py`. Por esse motivo
pandas está declarado em `requirements.txt`, e não apenas no
`requirements-train.txt`.
"""

from __future__ import annotations

from typing import List

STATION: str = "MIA"

# Coluna bruta do CSV ASOS com a precipitação acumulada na hora, em polegadas.
TARGET_RAW: str = "p01i"

# Coluna derivada pelo pipeline: choveu na hora SEGUINTE (0/1).
TARGET: str = "rain_next_hour"

# Limiar em polegadas para considerar que houve chuva na hora.
# 0.01" é o menor valor mensurável do ASOS; abaixo disso o relatório
# traz "T" (trace), tratado como 0.00 na limpeza.
RAIN_THRESHOLD_INCHES: float = 0.01

# Ordem canônica das features do modelo. NÃO reordene nem insira no
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

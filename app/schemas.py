"""Define os contratos de entrada e saída da API.

Os schemas validam os dados na borda. Campos ausentes ou fora dos limites
são rejeitados com `422` antes de chegarem ao modelo, evitando que entradas
inválidas sejam processadas e produzam uma probabilidade aparentemente
válida.

O cliente envia o vetor de features já preparado, incluindo variáveis
derivadas como `dew_spread`, `hour_sin` e `month_cos`. Esse vetor é igual
ao consumido pelo modelo durante o treino, permitindo reproduzir na API
as mesmas entradas usadas no dataset.

A alternativa seria receber o horário e as leituras brutas para executar
as transformações no servidor. O formato atual foi mantido por ser mais
simples para a simulação de produção. Como contrapartida, o cliente precisa
calcular as variáveis derivadas, incluindo as codificações seno e cosseno.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.features import FEATURE_NAMES


class WeatherFeatures(BaseModel):
    """Representa uma observação meteorológica no instante `t`.

    Os campos de variação `mslp_delta_3h` e `relh_delta_1h` usam zero como
    valor padrão, permitindo predições sem histórico. Quando disponíveis,
    esses valores devem ser enviados, pois informam tendências recentes da
    pressão e da umidade.
    """

    tmpf: float = Field(..., description="Temperatura (°F)", examples=[82.0])
    dwpf: float = Field(..., description="Ponto de orvalho (°F)", examples=[75.0])
    relh: float = Field(..., ge=0, le=100, description="Umidade relativa (%)", examples=[79.5])
    mslp: float = Field(..., description="Pressão ao nível do mar (hPa)", examples=[1012.5])
    vsby: float = Field(..., ge=0, description="Visibilidade (milhas)", examples=[10.0])
    wind_speed: float = Field(..., ge=0, description="Velocidade do vento (nós)", examples=[8.0])
    wind_u: float = Field(0.0, description="Componente leste-oeste do vento", examples=[-5.6])
    wind_v: float = Field(0.0, description="Componente norte-sul do vento", examples=[-5.6])
    dew_spread: float = Field(..., description="tmpf - dwpf", examples=[7.0])
    mslp_delta_3h: float = Field(0.0, description="Variação da pressão em 3h", examples=[-1.2])
    relh_delta_1h: float = Field(0.0, description="Variação da umidade em 1h", examples=[3.5])
    rain_now: float = Field(0.0, ge=0, le=1, description="Está chovendo agora (0 ou 1)", examples=[0.0])
    hour_sin: float = Field(..., ge=-1, le=1, examples=[0.5])
    hour_cos: float = Field(..., ge=-1, le=1, examples=[-0.866])
    month_sin: float = Field(..., ge=-1, le=1, examples=[-0.5])
    month_cos: float = Field(..., ge=-1, le=1, examples=[-0.866])

    def to_list(self) -> List[float]:
        """Converte a observação em uma lista na ordem canônica de `FEATURE_NAMES`.

        A ordem é necessária porque o scaler e a primeira camada da rede foram
        treinados nessa sequência. Usar `FEATURE_NAMES`, definido em
        `app/constants.py`, evita duplicar a lista e mantém o treino e a API
        consistentes.
        """
        return [getattr(self, name) for name in FEATURE_NAMES]


class PredictResponse(BaseModel):
    """Representa a resposta do endpoint `/predict`.

    Retorna duas informações diferentes:

    - `probability`: saída contínua do modelo, entre `0` e `1`;
    - `prediction`: decisão binária após aplicar o limiar configurado.

    Por exemplo, com limiar de `0.84`, uma probabilidade de `0.79` resulta
    em `prediction = 0`.

    O campo `threshold` também é retornado para que o cliente conheça o
    limiar usado. Assim, pode aplicar uma regra mais ou menos conservadora
    à probabilidade recebida, se necessário.
    """

    prediction: int = Field(..., description="1 = vai chover na próxima hora")
    probability: float = Field(..., description="Probabilidade da classe positiva")
    threshold: float = Field(..., description="Limiar aplicado para gerar `prediction`")
    model_version: int
    prediction_id: int = Field(..., description="Identificador do registro salvo")


class LabeledObservation(BaseModel):
    """Representa uma observação cujo rótulo verdadeiro já está disponível."""

    features: WeatherFeatures
    target: int = Field(..., ge=0, le=1, description="1 se choveu na hora seguinte")


class UpdateRequest(BaseModel):
    """Representa um lote de observações rotuladas para atualização incremental.

    Aceita entre 1 e 5.000 observações. O uso de lotes é preferível ao envio
    de uma observação por vez porque, com apenas 5% de casos positivos, um
    lote unitário quase sempre contém somente exemplos negativos, produzindo
    um gradiente pouco informativo.

    `epochs` e `learning_rate` são opcionais. Quando omitidos, a API utiliza
    os valores padrão definidos em `app/model.py`:

    - `EPOCHS_UPDATE = 2`;
    - `LR_UPDATE = 1e-4`.

    A taxa de aprendizado da atualização é dez vezes menor que a usada no
    treinamento inicial, reduzindo o risco de o lote recente sobrescrever o
    conhecimento já aprendido.
    """

    observations: List[LabeledObservation] = Field(..., min_length=1, max_length=5000)
    epochs: Optional[int] = Field(
        None, ge=1, le=10,
        description="Épocas sobre o lote. Padrão: EPOCHS_UPDATE (2).",
        examples=[2],
    )
    # O exemplo explícito evita que o Swagger UI sugira um valor inválido.
    # Sem ele, a interface poderia usar 1 como exemplo para `learning_rate`,
    # mas esse valor ultrapassa o limite `le=1e-2` e causaria erro 422.
    learning_rate: Optional[float] = Field(
        None, gt=0, le=1e-2,
        description="Taxa de aprendizado do update. Padrão: LR_UPDATE (1e-4).",
        examples=[1e-4],
    )


class UpdateResponse(BaseModel):
    status: str
    n_samples: int
    previous_version: int
    new_version: int
    loss: float
    metrics_before_update: Dict[str, float] = Field(
        ..., description="Desempenho do modelo neste lote ANTES de aprender com ele"
    )


class ModelInfo(BaseModel):
    version: int
    n_updates: int
    n_samples_seen: int
    n_features: int
    hidden_size: int
    threshold: float
    n_parameters: int


class HealthResponse(BaseModel):
    """Representa a resposta do endpoint `/health`.

    `status` assume o valor `"ok"` somente quando o modelo está carregado.
    Se o checkpoint não for encontrado, o serviço continua respondendo com
    HTTP 200 e `status="degraded"`.

    Nesse caso, o processo está vivo, mas os endpoints que dependem do modelo
    retornam `503 Service Unavailable`. A distinção entre liveness e readiness
    é detalhada no README.
    """

    status: str
    model_loaded: bool
    model_version: Optional[int] = None

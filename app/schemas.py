"""Contratos de entrada e saída da API.

Os schemas fazem a validação na borda: se um campo estiver faltando ou
fora de faixa, a requisição é rejeitada com 422 antes de chegar ao
modelo. Isso evita a classe de bug mais chata em serviço de ML — o
modelo receber lixo silenciosamente e devolver uma probabilidade
plausível.

Decisão de contrato: o cliente envia o vetor de features já pronto,
incluindo as derivadas (`dew_spread`, `hour_sin`, `month_cos`, ...).
A alternativa seria aceitar um timestamp e as leituras cruas e derivar
tudo no servidor. Foi mantido o formato atual porque ele é o mesmo
vetor que o modelo consome no treino, o que torna a simulação de
produção um replay direto do dataset. O custo é que o cliente precisa
saber calcular sin/cos — algo listado em "próximos passos" no README.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.features import FEATURE_NAMES


class WeatherFeatures(BaseModel):
    """Observação meteorológica no instante t.

    Os campos de variação (`mslp_delta_3h`, `relh_delta_1h`) têm valor
    padrão zero para permitir chamadas sem histórico, mas o ideal é
    enviá-los: são o que informa ao modelo se a pressão está caindo.
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
        """Converte para lista na ordem canônica de `FEATURE_NAMES`.

        A ordem importa: o scaler e a primeira camada da rede dependem
        dela. Derivar a lista de `FEATURE_NAMES` (definida em
        `app/constants.py`) em vez de escrever à mão garante que treino
        e serving nunca divirjam.
        """
        return [getattr(self, name) for name in FEATURE_NAMES]


class PredictResponse(BaseModel):
    """Resposta do `/predict`.

    Devolve os dois níveis de informação de propósito:

        probability  a saída contínua do modelo, entre 0 e 1;
        prediction   a decisão binária, já com o limiar aplicado.

    Não são a mesma coisa. Com limiar calibrado em 0,84, uma
    probabilidade de 0,79 resulta em `prediction = 0`. Devolver os dois
    permite ao cliente aplicar o próprio corte se quiser ser mais ou
    menos conservador, e é por isso que `threshold` também vai na
    resposta.
    """

    prediction: int = Field(..., description="1 = vai chover na próxima hora")
    probability: float = Field(..., description="Probabilidade da classe positiva")
    threshold: float = Field(..., description="Limiar aplicado para gerar `prediction`")
    model_version: int
    prediction_id: int = Field(..., description="Identificador do registro salvo")


class LabeledObservation(BaseModel):
    """Uma observação com o rótulo verdadeiro já conhecido."""

    features: WeatherFeatures
    target: int = Field(..., ge=0, le=1, description="1 se choveu na hora seguinte")


class UpdateRequest(BaseModel):
    """Lote de observações rotuladas para atualizar o modelo.

    Aceita de 1 a 5.000 observações. O lote é preferível a uma
    observação por vez: com apenas 5% de positivos, um lote de tamanho
    1 quase nunca contém chuva, e o gradiente resultante é ruído.

    `epochs` e `learning_rate` são opcionais. Omitidos, a API usa os
    defaults de `app/model.py` (EPOCHS_UPDATE = 2, LR_UPDATE = 1e-4).
    O learning rate de update é dez vezes menor que o do treino inicial
    justamente para que um lote recente não sobrescreva o que o modelo
    aprendeu antes.
    """

    observations: List[LabeledObservation] = Field(..., min_length=1, max_length=5000)
    epochs: Optional[int] = Field(
        None, ge=1, le=10,
        description="Épocas sobre o lote. Padrão: EPOCHS_UPDATE (2).",
        examples=[2],
    )
    # O `examples` aqui não é decorativo: sem ele, o Swagger UI sugere
    # 1 como valor de exemplo, que viola o próprio `le=1e-2` e resulta
    # em 422 para quem apenas clica em "Try it out".
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
    """Resposta do `/health`.

    `status` é "ok" apenas quando o modelo está carregado. Quando o
    checkpoint não é encontrado, o serviço sobe assim mesmo e responde
    "degraded" com HTTP 200 — o processo está vivo, mas os endpoints
    que dependem do modelo devolvem 503. Ver a nota sobre liveness e
    readiness no README.
    """

    status: str
    model_loaded: bool
    model_version: Optional[int] = None

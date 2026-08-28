"""Contratos de entrada e saída da API.

Os schemas fazem a validação na borda: se um campo estiver faltando ou
fora de faixa, a requisição é rejeitada com 422 antes de chegar ao
modelo. Isso evita a classe de bug mais chata em serviço de ML — o
modelo receber lixo silenciosamente e devolver uma probabilidade
plausível.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

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
        dela. Derivar a lista de `FEATURE_NAMES` em vez de escrever à
        mão garante que treino e serving nunca divirjam.
        """
        return [getattr(self, name) for name in FEATURE_NAMES]


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="1 = vai chover na próxima hora")
    probability: float = Field(..., description="Probabilidade da classe positiva")
    threshold: float
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
    """

    observations: List[LabeledObservation] = Field(..., min_length=1, max_length=5000)
    epochs: Optional[int] = Field(None, ge=1, le=10)
    learning_rate: Optional[float] = Field(None, gt=0, le=1e-2)

    @field_validator("observations")
    @classmethod
    def _warn_on_tiny_batch(cls, v):
        return v


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
    status: str
    model_loaded: bool
    model_version: Optional[int] = None

"""API de previsão de chuva.

Expõe o modelo treinado como serviço e permite atualizá-lo com dados
rotulados novos, sem recriar a rede.

    GET  /health    verificação de vida
    GET  /model     metadados da versão em uso
    POST /predict   previsão para uma observação
    POST /update    atualização incremental com um lote rotulado
    GET  /metrics   monitoramento
    GET  /versions  histórico de versões do modelo

Rodar localmente:
    uvicorn app.main:app --reload
    http://localhost:8000/docs
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException

from app import registry, storage
from app.metrics import compute_metrics, confusion_summary
from app.model import EPOCHS_UPDATE, LR_UPDATE, RainModel
from app.schemas import (
    HealthResponse, ModelInfo, PredictResponse, UpdateRequest,
    UpdateResponse, WeatherFeatures,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pt"))
DB_PATH = Path(os.getenv("DB_PATH", "data/monitoring.db"))

# Estado do processo. O lock serializa as atualizações: `/update` altera
# os pesos in-place, e duas atualizações simultâneas corromperiam o
# estado do otimizador. Predições são leitura e não precisam do lock.
_model: RainModel | None = None
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma vez na subida do serviço."""
    global _model
    storage.init_db(DB_PATH)
    if MODEL_PATH.exists():
        _model = RainModel.load(MODEL_PATH)
        print(f"modelo carregado: {MODEL_PATH} (versão {_model.version})")
    else:
        print(f"[aviso] {MODEL_PATH} não encontrado. "
              f"Rode: python -m training.train_initial")
    yield


app = FastAPI(
    title="Rain Prediction API",
    description=(
        "Prevê chuva na próxima hora a partir de observações de superfície, "
        "com atualização incremental do modelo."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_model() -> RainModel:
    """Devolve o modelo carregado ou falha com 503."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treino inicial.",
        )
    return _model


# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    """Verificação de vida, usada pelo Docker e pelo serviço de deploy."""
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=_model.version if _model else None,
    )


@app.get("/model", response_model=ModelInfo, tags=["modelo"])
def model_info() -> ModelInfo:
    """Metadados da versão em uso."""
    return ModelInfo(**get_model().info())


@app.get("/versions", tags=["modelo"])
def versions() -> Dict:
    """Histórico de versões registradas."""
    return {"current": registry.current(), "history": registry.history()}


# --------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["inferência"])
def predict(features: WeatherFeatures) -> PredictResponse:
    """Prevê se vai chover na próxima hora.

    Toda predição é registrada no SQLite, o que alimenta o `/metrics`
    e o dashboard.
    """
    model = get_model()
    values = features.to_list()

    probability = float(model.predict_proba(np.array([values], dtype=np.float32))[0])
    prediction = int(probability >= model.threshold)

    prediction_id = storage.log_prediction(
        model_version=model.version,
        probability=probability,
        prediction=prediction,
        threshold=model.threshold,
        features=values,
        db_path=DB_PATH,
    )

    return PredictResponse(
        prediction=prediction,
        probability=round(probability, 4),
        threshold=round(model.threshold, 4),
        model_version=model.version,
        prediction_id=prediction_id,
    )


@app.post("/update", response_model=UpdateResponse, tags=["treinamento"])
def update(request: UpdateRequest) -> UpdateResponse:
    """Atualiza o modelo incrementalmente com um lote rotulado.

    A ordem das operações reproduz o protocolo do experimento offline:

        1. o modelo prevê sobre o lote e o desempenho é registrado;
        2. só depois ele aprende com esses dados;
        3. o checkpoint é salvo com a versão incrementada.

    Avaliar antes de aprender é o que torna a métrica honesta: medir
    depois do treino mostraria o modelo acertando dados que ele acabou
    de ver.

    Os pesos, o estado do otimizador e o scaler são preservados. A rede
    não é recriada.
    """
    model = get_model()

    X = np.array([o.features.to_list() for o in request.observations], dtype=np.float32)
    y = np.array([o.target for o in request.observations], dtype=np.float32)

    with _lock:
        previous_version = model.version

        # (1) avaliação antes do aprendizado
        probs = model.predict_proba(X)
        cm = confusion_summary(y, (probs >= model.threshold).astype(int))
        metrics_before = compute_metrics(cm)
        storage.log_evaluation(previous_version, cm, db_path=DB_PATH)

        # (2) aprendizado incremental
        loss = model.incremental_fit(
            X, y,
            epochs=request.epochs or EPOCHS_UPDATE,
            lr=request.learning_rate or LR_UPDATE,
        )

        # (3) persistência da nova versão
        model.save(MODEL_PATH)
        storage.log_update(previous_version, model.version, len(X), loss, db_path=DB_PATH)
        registry.register(
            version=model.version,
            model_path=str(MODEL_PATH),
            stage="incremental_update",
            n_samples=len(X),
            metrics=metrics_before,
            notes=f"Atualização via API com {len(X)} observações",
        )

    return UpdateResponse(
        status="ok",
        n_samples=len(X),
        previous_version=previous_version,
        new_version=model.version,
        loss=round(loss, 5),
        metrics_before_update=metrics_before,
    )


# --------------------------------------------------------------------------

@app.get("/metrics", tags=["monitoramento"])
def metrics() -> Dict:
    """Métricas operacionais do serviço.

    Reúne o que dá para observar sem infraestrutura extra:

        - volume e distribuição das predições servidas;
        - quantas atualizações o modelo já recebeu;
        - desempenho real, quando lotes rotulados chegaram via /update.

    A distribuição das probabilidades é o sinal mais útil aqui: se ela
    se deslocar em relação ao histórico, algo mudou nos dados de
    entrada ou no modelo — sem precisar esperar pelos rótulos.
    """
    model = get_model()
    return {
        "model": model.info(),
        "predictions": storage.prediction_stats(DB_PATH),
        "updates": storage.update_stats(DB_PATH),
        "performance": storage.performance_stats(DB_PATH),
    }

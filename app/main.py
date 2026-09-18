"""API de previsão de chuva.

Expõe o modelo treinado como serviço e permite atualizá-lo com novos
dados rotulados, sem recriar a rede.

Endpoints:

    GET  /health    Verificação de vida.
    GET  /model     Metadados da versão atual.
    POST /predict   Previsão para uma observação.
    POST /update    Atualização incremental com um lote rotulado.
    GET  /metrics   Métricas operacionais.
    GET  /versions  Histórico de versões do modelo.

Execução local:

    uvicorn app.main:app --reload

Documentação:

    http://localhost:8000/docs

O modelo é carregado uma vez durante o startup, permanece na memória do
processo e é atualizado in-place pelo endpoint `/update`.

Por isso, o serviço deve ser executado com `--workers 1`. Com múltiplos
workers, cada processo teria sua própria cópia do modelo. Uma atualização
feita em um worker não seria refletida nos demais, e as predições
poderiam usar versões diferentes dos pesos.
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

# O caminho do registro não é armazenado neste módulo. As funções de
# `app/registry.py` consultam `registry.REGISTRY_PATH` no momento da
# chamada, permitindo que os testes redirecionem o registro para um
# diretório temporário.

# ---------------------------------------------------------------- estado

# O modelo é global no processo e carregado uma única vez durante o startup.
# Carregá-lo a cada requisição exigiria ler o checkpoint e reconstruir o
# estado do otimizador em todo `/predict`. Além de ser mais caro, isso
# descartaria atualizações incrementais ainda mantidas em memória.
_model: RainModel | None = None

# O lock serializa as atualizações do modelo. O endpoint `/update`
# altera os pesos e o estado do otimizador in-place; sem o lock, duas
# atualizações concorrentes poderiam intercalar etapas de gradiente e
# corromper o estado do Adam.
#
# `/predict` não usa o lock intencionalmente. Como os endpoints são
# síncronos, o Uvicorn pode executá-los em threads diferentes. Assim,
# uma predição pode ocorrer durante um `optimizer.step()` e observar
# uma combinação de parâmetros parcialmente atualizada.
#
# Nesse caso, o resultado pode ser calculado sobre um estado intermediário,
# mas não há exceção nem corrupção permanente do modelo. A janela dura
# apenas alguns milissegundos, a rede não usa dropout nem batch normalization,
# e as atualizações são raras. Usar o lock em toda predição teria custo
# de serializar as leituras por causa de um evento pouco frequente.
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa os recursos necessários para o serviço.

    Cria o banco de dados e carrega o modelo uma única vez durante a
    inicialização da aplicação.

    Se o checkpoint não existir, o serviço continua disponível. Nesse caso,
    `/health` permanece acessível para facilitar o diagnóstico, enquanto os
    endpoints que dependem do modelo retornam `503 Service Unavailable` com
    uma mensagem indicando como corrigir o problema.

    Essa decisão evita que o container entre em um ciclo de reinicializações
    e facilita a identificação do erro nos logs.
    """
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
    """Retorna o modelo carregado ou informa que o serviço está indisponível.

    Usa HTTP 503 em vez de 500 porque a requisição é válida, mas o serviço
    ainda não possui o modelo necessário para processá-la. Isso também
    permite que um balanceador encaminhe a requisição para outra instância
    ou tente novamente mais tarde.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o treino inicial.",
        )
    return _model


# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    """Verifica se o processo HTTP está respondendo.

    Retorna 200 mesmo quando o modelo ainda não foi carregado.
    Nesse caso, informa `status="degraded"` e `model_loaded=false`.

    O endpoint verifica apenas a disponibilidade do processo (liveness).
    A confirmação de que o modelo está pronto para atender predições
    (readiness) é feita a partir do JSON retornado. Um endpoint separado
    `/ready` poderá ser adicionado posteriormente.
    """
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=_model.version if _model else None,
    )


@app.get("/model", response_model=ModelInfo, tags=["modelo"])
def model_info() -> ModelInfo:
    """Retorna os metadados da versão atual do modelo."""
    return ModelInfo(**get_model().info())


@app.get("/versions", tags=["modelo"])
def versions() -> Dict:
    """Retorna o histórico de versões registrado em `models/registry.json`."""
    return {"current": registry.current(), "history": registry.history()}


# --------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["inferência"])
def predict(features: WeatherFeatures) -> PredictResponse:
    """Prevê se vai chover na próxima hora.

    Cada predição é armazenada no SQLite para alimentar o endpoint `/metrics`
    e o dashboard com o volume de requisições e a distribuição das
    probabilidades geradas.

    Essa distribuição é o principal sinal disponível antes da chegada dos
    rótulos reais, que só podem ser conhecidos após a próxima hora.

    A resposta separa:

    - `probability`: probabilidade contínua estimada pelo modelo;
    - `prediction`: decisão binária obtida após aplicar o limiar.

    Ver `PredictResponse`.
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

        O fluxo segue o protocolo pré-sequencial: primeiro avaliar, depois aprender.

        1. O modelo faz previsões sobre o lote recebido.
        2. O desempenho é registrado antes da atualização.
        3. O modelo aprende com o mesmo lote.
        4. O checkpoint é salvo com a versão incrementada.

        Avaliar antes do treino evita uma métrica otimista, pois o modelo ainda
        não viu os dados usados na avaliação.

        A versão identifica o estado dos pesos que gerou cada predição. O
        checkpoint continua sendo salvo em `models/model.pt`, portanto o arquivo
        é sobrescrito a cada atualização e apenas a versão mais recente permanece
        em disco.

        O histórico das versões é mantido em `models/registry.json` e na tabela
        `updates` do SQLite. Essa escolha evita armazenar um arquivo `.pt` para
        cada versão sem prejudicar a demonstração.

        Os pesos, o estado do otimizador e o scaler são preservados durante a
        atualização. A rede não é recriada.
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
        # A avaliação é atribuída à versão ANTERIOR, que é a que de fato
        # produziu essas previsões.
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
            notes=(
                f"Atualização via API com {len(X)} observações. "
                f"As métricas desta entrada foram medidas no lote recebido "
                f"ANTES do treino, ou seja, descrevem o desempenho da versão "
                f"{previous_version} sobre esses dados."
            ),
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
    """Retorna as métricas operacionais disponíveis no serviço.

        Reúne informações observáveis sem infraestrutura adicional:

        - volume e distribuição das predições;
        - quantidade de atualizações incrementais;
        - desempenho do modelo após a chegada de lotes rotulados.

        `performance` permanece como `None` até o primeiro `/update`.
        Isso é esperado: sem rótulos verdadeiros, não é possível calcular
        acertos ou erros. Retornar `None` evita inventar uma métrica.

        Enquanto os rótulos não chegam, a distribuição das probabilidades é o
        principal sinal de monitoramento. Mudanças nessa distribuição podem
        indicar alterações nos dados de entrada ou no comportamento do modelo.
        """
    model = get_model()
    return {
        "model": model.info(),
        "predictions": storage.prediction_stats(DB_PATH),
        "updates": storage.update_stats(DB_PATH),
        "performance": storage.performance_stats(DB_PATH),
    }

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

Desenho do processo, em uma frase: o modelo é carregado uma vez, vive
na memória deste processo e é alterado in-place pelo /update. É por
isso que o serviço roda com --workers 1. Com vários workers, cada
processo teria a sua própria cópia do checkpoint; um /update atingiria
apenas um deles e os /predict seguintes responderiam com pesos
diferentes dependendo de qual worker atendesse a requisição.
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

# O caminho do registro NÃO é lido aqui de propósito: as funções de
# `app/registry.py` resolvem `registry.REGISTRY_PATH` no momento da
# chamada. Capturar o valor neste módulo impediria os testes de
# redirecionar o registro para um diretório temporário.

# ---------------------------------------------------------------- estado
# O modelo é global do processo, carregado uma única vez no startup.
# Carregar por requisição custaria a leitura do checkpoint e a
# reconstrução do otimizador a cada /predict — e, pior, jogaria fora
# qualquer atualização incremental feita em memória.
_model: RainModel | None = None

# O lock serializa as atualizações. `/update` altera os pesos e o estado
# do otimizador in-place; duas atualizações concorrentes intercalariam
# passos de gradiente e corromperiam as médias móveis do Adam.
#
# `/predict` NÃO adquire o lock, e isso é uma escolha, não um descuido.
# Como os endpoints são síncronos, o Uvicorn os executa em um
# threadpool, então uma predição pode ocorrer no meio de um
# `optimizer.step()` e ler uma camada já atualizada junto com outra
# ainda antiga. O resultado é uma probabilidade calculada sobre um
# estado intermediário — não uma exceção nem corrupção de dados. A
# janela é de milissegundos, a rede não tem dropout nem batchnorm, e
# pagar um lock em todo /predict serializaria a leitura por causa de um
# evento que acontece a cada duas semanas de dados simulados. O
# trade-off foi aceito conscientemente.
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepara o processo: cria o banco e carrega o modelo uma única vez.

    Se o checkpoint não existir, o serviço sobe assim mesmo. É
    deliberado: `/health` continua respondendo (útil para diagnosticar
    um container recém-criado) e os endpoints que dependem do modelo
    devolvem 503 com uma mensagem acionável, em vez de o container
    entrar em crash loop e não deixar rastro nos logs.
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
    """Devolve o modelo carregado ou falha com 503.

    503 (Service Unavailable) e não 500: não houve erro no
    processamento da requisição, o serviço é que ainda não tem o que
    precisa para atendê-la. É também o código que um balanceador
    interpreta como "tente outra instância", que é o comportamento
    desejado.
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
    """Verificação de vida, usada pelo Docker e pelo serviço de deploy.

    Responde 200 mesmo sem modelo carregado, com `status="degraded"` e
    `model_loaded=false`. O HEALTHCHECK do Docker confirma apenas que o
    processo HTTP responde (liveness); saber se o modelo está pronto
    (readiness) é responsabilidade de quem lê o JSON. Separar os dois
    em `/health` e `/ready` está listado em próximos passos no README.
    """
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
    """Histórico de versões registradas em models/registry.json."""
    return {"current": registry.current(), "history": registry.history()}


# --------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["inferência"])
def predict(features: WeatherFeatures) -> PredictResponse:
    """Prevê se vai chover na próxima hora.

    Toda predição é gravada no SQLite. Isso não é log por log: é o que
    permite ao `/metrics` e ao dashboard mostrarem volume e
    distribuição das probabilidades servidas. A distribuição é o único
    sinal de monitoramento disponível antes de os rótulos verdadeiros
    chegarem — e eles só chegam uma hora depois, no melhor caso.

    A resposta separa `probability` (saída contínua do modelo) de
    `prediction` (decisão binária depois do limiar). Ver PredictResponse.
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

    A ordem das operações segue o protocolo prequencial — prever
    primeiro, aprender depois:

        1. o modelo prevê sobre o lote e o desempenho é registrado;
        2. só depois ele aprende com esses dados;
        3. o checkpoint é salvo com a versão incrementada.

    Avaliar antes de aprender é o que torna a métrica honesta: medir
    depois do treino mostraria o modelo acertando dados que ele acabou
    de ver.

    Sobre a versão: ela é incrementada a cada `incremental_fit`, e serve
    para responder "qual estado dos pesos gerou esta predição". Como o
    checkpoint é salvo sempre no MESMO caminho (models/model.pt), o
    arquivo é sobrescrito e só o estado mais recente existe em disco. O
    histórico de como se chegou até ele vive em models/registry.json e
    na tabela `updates` do SQLite. É uma escolha de simplicidade:
    guardar um .pt por versão exigiria política de retenção e não
    acrescentaria nada à demonstração.

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
    """Métricas operacionais do serviço.

    Reúne o que dá para observar sem infraestrutura extra:

        - volume e distribuição das predições servidas;
        - quantas atualizações o modelo já recebeu;
        - desempenho real, quando lotes rotulados chegaram via /update.

    O campo `performance` é None até o primeiro /update. Isso não é
    falha: sem rótulo verdadeiro não existe acerto ou erro a medir, e o
    projeto prefere devolver None a inventar um número. É o problema
    central de monitorar ML em produção — o feedback chega atrasado, e
    aqui chega em lotes de duas semanas.

    A distribuição das probabilidades é o sinal mais útil enquanto isso:
    se ela se deslocar em relação ao histórico, algo mudou nos dados de
    entrada ou no modelo, sem precisar esperar pelos rótulos.
    """
    model = get_model()
    return {
        "model": model.info(),
        "predictions": storage.prediction_stats(DB_PATH),
        "updates": storage.update_stats(DB_PATH),
        "performance": storage.performance_stats(DB_PATH),
    }

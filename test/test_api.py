"""Testes da API.

    pytest tests/test_api.py -v

Cada teste roda contra um modelo, um banco e um registro de versões
temporários, para não tocar nos artefatos do projeto. Os três
isolamentos são necessários e cada um funciona de um jeito diferente:

    MODEL_PATH e DB_PATH   variáveis de ambiente lidas por app/main.py
                           no momento do import, por isso o reload.
    REGISTRY_PATH          atributo de módulo em app/registry.py,
                           resolvido a cada chamada.

O terceiro só funciona porque `registry.register` resolve o caminho
dentro da função. Se ele usasse `registry_path=REGISTRY_PATH` como
default na assinatura, o valor ficaria preso no momento da definição e
os testes gravariam no models/registry.json real do repositório — que
é versionado e faz parte do artefato entregue.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FEATURE_NAMES
from app.model import RainModel

SAMPLE = {
    "tmpf": 82.0, "dwpf": 75.0, "relh": 79.5, "mslp": 1012.5, "vsby": 10.0,
    "wind_speed": 8.0, "wind_u": -5.6, "wind_v": -5.6, "dew_spread": 7.0,
    "mslp_delta_3h": -1.2, "relh_delta_1h": 3.5, "rain_now": 0.0,
    "hour_sin": 0.5, "hour_cos": -0.866, "month_sin": -0.5, "month_cos": -0.866,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Sobe a API com modelo, banco e registro isolados.

    Os dados sintéticos são gerados na escala real de cada feature. Se
    fossem N(0,1), o scaler ficaria calibrado numa escala em que
    `mslp=1012.5` estaria a centenas de desvios da média, o sigmoid
    saturaria em 1.0 e os testes não conseguiriam observar mudança
    alguma na predição.
    """
    rng = np.random.default_rng(0)
    n = 500
    centro = np.array([SAMPLE[f] for f in FEATURE_NAMES], dtype=np.float32)
    escala = np.array(
        [6, 6, 12, 4, 3, 5, 6, 6, 4, 2, 8, 0.4, 0.7, 0.7, 0.7, 0.7],
        dtype=np.float32,
    )
    X = (centro + rng.normal(size=(n, len(FEATURE_NAMES))) * escala).astype(np.float32)

    # Alvo ruidoso: sem ruído o modelo fica confiante demais e satura.
    sinal = (X[:, 2] - SAMPLE["relh"]) / 12 - (X[:, 3] - SAMPLE["mslp"]) / 4
    y = (sinal + rng.normal(scale=1.0, size=n) > 0.3).astype(np.float32)

    model = RainModel()
    model.fit(X, y, epochs=15)
    model_path = tmp_path / "model.pt"
    model.save(model_path)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "monitoring.db"))
    monkeypatch.setattr("app.registry.REGISTRY_PATH", tmp_path / "registry.json")

    # O reload é o que faz app/main.py reler MODEL_PATH e DB_PATH do
    # ambiente. Sem ele, os valores capturados no primeiro import
    # continuariam valendo.
    import importlib
    from app import main
    importlib.reload(main)

    with TestClient(main.app) as c:
        yield c


# --------------------------------------------------------------------------

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_retorna_probabilidade_valida(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction_id"] > 0


def test_predict_aplica_o_limiar_do_modelo(client):
    """`prediction` precisa ser coerente com `probability` e `threshold`.

    Protege contra a confusão mais comum entre probabilidade e decisão
    binária: se algum dia o limiar deixasse de ser aplicado, este teste
    quebraria antes de o erro chegar a um cliente.
    """
    body = client.post("/predict", json=SAMPLE).json()
    esperado = int(body["probability"] >= body["threshold"])
    assert body["prediction"] == esperado


def test_predict_rejeita_entrada_invalida(client):
    """Umidade fora de 0-100 deve ser barrada pelo schema, não pelo modelo."""
    invalido = {**SAMPLE, "relh": 150.0}
    assert client.post("/predict", json=invalido).status_code == 422


def test_predict_rejeita_campo_faltando(client):
    incompleto = {k: v for k, v in SAMPLE.items() if k != "tmpf"}
    assert client.post("/predict", json=incompleto).status_code == 422


def test_update_incrementa_versao_e_avalia_antes(client):
    """`/update` deve avaliar o lote antes de aprender com ele."""
    versao_antes = client.get("/model").json()["version"]

    payload = {"observations": [{"features": SAMPLE, "target": 1} for _ in range(20)]}
    r = client.post("/update", json=payload)
    assert r.status_code == 200

    body = r.json()
    assert body["new_version"] == versao_antes + 1
    assert body["n_samples"] == 20
    assert "f1" in body["metrics_before_update"]
    assert client.get("/model").json()["version"] == versao_antes + 1


def test_update_muda_a_predicao(client):
    """Aprender com um lote precisa ter efeito observável na saída.

    O lote usa rótulo negativo: a probabilidade inicial para esta
    observação é alta, então há espaço para cair. Testar na direção
    oposta falharia por saturação em 1.0, não por falta de aprendizado.
    """
    antes = client.post("/predict", json=SAMPLE).json()["probability"]

    payload = {
        "observations": [{"features": SAMPLE, "target": 0} for _ in range(50)],
        "epochs": 5,
        "learning_rate": 1e-3,
    }
    client.post("/update", json=payload)

    depois = client.post("/predict", json=SAMPLE).json()["probability"]
    assert depois != antes, "a atualização deveria alterar a predição"
    assert depois < antes, "após 50 exemplos negativos a probabilidade deveria cair"


def test_update_rejeita_learning_rate_fora_da_faixa(client):
    """`learning_rate` acima de 1e-2 é barrado pelo schema.

    Cobre o valor que o Swagger sugeria por padrão antes de o campo
    ganhar um `examples` explícito.
    """
    payload = {
        "observations": [{"features": SAMPLE, "target": 0}],
        "learning_rate": 1.0,
    }
    assert client.post("/update", json=payload).status_code == 422


def test_update_exige_ao_menos_uma_observacao(client):
    assert client.post("/update", json={"observations": []}).status_code == 422


def test_metrics_sem_rotulos_nao_tem_performance(client):
    """Antes de qualquer /update, `performance` é None — não zero.

    Sem rótulo verdadeiro não existe acerto a medir. Devolver zeros
    pareceria um modelo ruim; None deixa claro que a medição não
    aconteceu.
    """
    client.post("/predict", json=SAMPLE)
    m = client.get("/metrics").json()
    assert m["predictions"]["total"] == 1
    assert m["performance"] is None


def test_metrics_reflete_o_uso(client):
    for _ in range(3):
        client.post("/predict", json=SAMPLE)
    client.post(
        "/update",
        json={"observations": [{"features": SAMPLE, "target": 0} for _ in range(10)]},
    )

    m = client.get("/metrics").json()
    assert m["predictions"]["total"] == 3
    assert m["updates"]["total"] == 1
    assert m["updates"]["samples_learned"] == 10
    assert m["performance"]["n_labeled"] == 10
    assert m["model"]["n_updates"] == 1


def test_versions_registra_historico(client):
    client.post(
        "/update",
        json={"observations": [{"features": SAMPLE, "target": 1} for _ in range(5)]},
    )
    body = client.get("/versions").json()
    assert body["current"]["stage"] == "incremental_update"
    assert body["current"]["is_current"] is True


def test_registry_de_teste_fica_isolado(tmp_path, client):
    """O registro escrito pelos testes não pode ser o do repositório.

    Este é o teste que protege contra a regressão do argumento default
    avaliado uma única vez. Se `registry.register` voltar a capturar o
    caminho na assinatura, o arquivo temporário nunca é criado e este
    teste falha — em vez de o repositório ser silenciosamente sujo.
    """
    import app.registry as registry

    client.post(
        "/update",
        json={"observations": [{"features": SAMPLE, "target": 1} for _ in range(5)]},
    )

    assert Path(registry.REGISTRY_PATH).exists()
    assert Path(registry.REGISTRY_PATH).parent == tmp_path

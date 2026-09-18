"""Testes da API.

Execute com:

    pytest tests/test_api.py -v

Cada teste usa um modelo, um banco de dados e um registro de versões
temporários, evitando alterações nos artefatos reais do projeto.

Os três isolamentos são necessários e funcionam de maneiras diferentes:

    MODEL_PATH e DB_PATH
        Variáveis de ambiente lidas por `app/main.py` no momento do
        import. Por isso, é necessário recarregar o módulo.

    REGISTRY_PATH
        Atributo de módulo em `app/registry.py`, resolvido a cada chamada.

O terceiro isolamento só funciona porque `registry.register` resolve o
caminho dentro da função. Se a assinatura usasse
`registry_path=REGISTRY_PATH` como valor padrão, o caminho seria capturado
no momento da definição da função. Nesse caso, os testes gravariam no
`models/registry.json` real do repositório, que é versionado e faz parte
do artefato entregue.
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
    """Inicia a API com modelo, banco de dados e registro de versões isolados.

    Os dados sintéticos são gerados na escala real de cada feature. Se fossem
    gerados a partir de `N(0, 1)`, o scaler seria calibrado em uma escala
    incompatível: um valor como `mslp=1012.5` ficaria a centenas de desvios
    padrão da média.

    Nesse cenário, o sigmoid saturaria em `1.0`, impedindo os testes de
    observar mudanças nas predições.
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
    """Verifica se `prediction` é coerente com `probability` e `threshold`.

    O teste protege contra a confusão comum entre probabilidade e decisão
    binária. Se o limiar deixasse de ser aplicado em algum momento, o teste
    falharia antes que o erro chegasse a um cliente.
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
    """Verifica se aprender com um lote produz efeito observável na saída.

    O lote usa um rótulo negativo. Como a probabilidade inicial dessa
    observação é alta, há espaço para que ela diminua após o aprendizado.

    Testar na direção oposta poderia falhar por saturação em `1.0`, e não
    por ausência de aprendizado.
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
    """Verifica que `learning_rate` acima de `1e-2` é rejeitado pelo schema.

    O teste cobre o valor que o Swagger sugeria como padrão antes de o campo
    receber um exemplo explícito (`examples`).
    """
    payload = {
        "observations": [{"features": SAMPLE, "target": 0}],
        "learning_rate": 1.0,
    }
    assert client.post("/update", json=payload).status_code == 422


def test_update_exige_ao_menos_uma_observacao(client):
    assert client.post("/update", json={"observations": []}).status_code == 422


def test_metrics_sem_rotulos_nao_tem_performance(client):
    """Verifica que `performance` é `None` antes de qualquer `/update`, e não zero.

    Sem rótulo verdadeiro, não há como medir o desempenho. Retornar zeros
    daria a impressão de que o modelo teve desempenho ruim; retornar `None`
    deixa explícito que ainda não houve medição.
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
    """Verifica que os testes não escrevem no registro do repositório.

    Este teste protege contra a regressão causada por um argumento padrão
    avaliado apenas uma vez. Se `registry.register` voltar a capturar o
    caminho na assinatura, o arquivo temporário não será criado e o teste
    falhará, evitando que o repositório seja alterado silenciosamente.
    """
    import app.registry as registry

    client.post(
        "/update",
        json={"observations": [{"features": SAMPLE, "target": 1} for _ in range(5)]},
    )

    assert Path(registry.REGISTRY_PATH).exists()
    assert Path(registry.REGISTRY_PATH).parent == tmp_path

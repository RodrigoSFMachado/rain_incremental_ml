"""Testes do modelo.

O teste mais importante aqui é `test_incremental_continues_from_saved_weights`:
ele prova, de forma verificável, que a atualização incremental continua
do estado salvo em vez de recomeçar do zero.

    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FEATURE_NAMES
from app.model import RainModel


@pytest.fixture
def data():
    """Dados sintéticos pequenos, só para exercitar a mecânica."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURE_NAMES)))
    y = (X[:, 0] + X[:, 3] + rng.normal(scale=0.5, size=400) > 0.8).astype(float)
    return X, y


@pytest.fixture
def trained(data):
    X, y = data
    model = RainModel()
    model.fit(X, y, epochs=5)
    return model


# --------------------------------------------------------------------------
# Mecânica básica
# --------------------------------------------------------------------------

def test_predict_proba_dentro_do_intervalo(trained, data):
    probs = trained.predict_proba(data[0])
    assert probs.shape == (len(data[0]),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_incremental_fit_exige_fit_antes(data):
    X, y = data
    model = RainModel()
    with pytest.raises(RuntimeError):
        model.incremental_fit(X[:10], y[:10])


def test_scaler_nao_muda_no_update(trained, data):
    """O scaler é congelado: atualizar o modelo não pode alterá-lo."""
    X, y = data
    mean_antes = trained.scaler.mean.copy()
    std_antes = trained.scaler.std.copy()

    trained.incremental_fit(X[:64], y[:64])

    np.testing.assert_array_equal(mean_antes, trained.scaler.mean)
    np.testing.assert_array_equal(std_antes, trained.scaler.std)


def test_versao_incrementa_a_cada_update(trained, data):
    X, y = data
    v0, u0 = trained.version, trained.n_updates
    trained.incremental_fit(X[:64], y[:64])
    assert trained.version == v0 + 1
    assert trained.n_updates == u0 + 1


# --------------------------------------------------------------------------
# Persistência
# --------------------------------------------------------------------------

def test_save_load_preserva_predicoes(trained, data, tmp_path):
    """Salvar e recarregar não pode mudar nenhuma previsão."""
    X, _ = data
    antes = trained.predict_proba(X)

    trained.save(tmp_path / "m.pt")
    depois = RainModel.load(tmp_path / "m.pt").predict_proba(X)

    np.testing.assert_allclose(antes, depois, rtol=1e-6)


def test_save_load_preserva_estado_do_otimizador(trained, tmp_path):
    """O estado do Adam precisa sobreviver ao round-trip."""
    trained.save(tmp_path / "m.pt")
    carregado = RainModel.load(tmp_path / "m.pt")

    orig = trained.optimizer.state_dict()["state"]
    novo = carregado.optimizer.state_dict()["state"]

    assert set(orig.keys()) == set(novo.keys())
    assert len(novo) > 0, "o otimizador deveria ter estado após o treino"
    for k in orig:
        torch.testing.assert_close(orig[k]["exp_avg"], novo[k]["exp_avg"])
        torch.testing.assert_close(orig[k]["exp_avg_sq"], novo[k]["exp_avg_sq"])


# --------------------------------------------------------------------------
# O teste central
# --------------------------------------------------------------------------

def test_incremental_continues_from_saved_weights(trained, data, tmp_path):
    """Prova que `incremental_fit` continua o treino em vez de recomeçar.

    A verificação tem duas partes:

    1. Ao recarregar o checkpoint, os pesos são idênticos aos salvos.
    2. Após uma atualização com taxa de aprendizado baixa, os pesos
       mudam pouco — porque partiram do estado anterior. Um modelo
       recriado do zero teria pesos completamente diferentes.
    """
    X, y = data
    trained.save(tmp_path / "v1.pt")

    recarregado = RainModel.load(tmp_path / "v1.pt")

    # (1) continuidade exata no reload
    for p_orig, p_novo in zip(trained.net.parameters(), recarregado.net.parameters()):
        assert torch.equal(p_orig, p_novo)

    pesos_antes = [p.detach().clone() for p in recarregado.net.parameters()]

    recarregado.incremental_fit(X[:128], y[:128], epochs=2, lr=1e-4)

    # (2) mudou (aprendeu algo) mas continua próximo (não recomeçou)
    deltas = [
        (p_dep - p_ant).abs().max().item()
        for p_ant, p_dep in zip(pesos_antes, recarregado.net.parameters())
    ]
    assert max(deltas) > 0, "os pesos deveriam ter mudado após a atualização"
    assert max(deltas) < 0.1, "mudança grande demais: parece treino do zero"

    # Comparação com o cenário errado: um modelo novo em folha.
    do_zero = RainModel()
    do_zero.scaler = recarregado.scaler
    distancia_do_zero = max(
        (p_zero - p_ant).abs().max().item()
        for p_ant, p_zero in zip(pesos_antes, do_zero.net.parameters())
    )
    assert distancia_do_zero > max(deltas), (
        "um modelo recriado deveria estar muito mais distante do estado "
        "anterior do que uma atualização incremental"
    )

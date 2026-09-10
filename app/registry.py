"""Registro operacional de versões do modelo.

Substitui o MLflow Model Registry por um arquivo JSON.

A justificativa é de escopo: o Model Registry do MLflow exige um
backend store, um servidor rodando e a API passando a depender dele em
tempo de execução — se o MLflow cai, a API cai junto. Neste projeto o
"modelo em produção" é um arquivo `.pt` no disco, e o que realmente
precisamos saber é: qual é a versão atual, quando foi criada, com
quantas amostras e com que desempenho.

Um JSON responde a isso em poucas linhas e continua legível por
humanos. O MLflow segue sendo usado para *tracking de experimentos*
offline, que é onde ele agrega de verdade.

Divisão de responsabilidades no projeto:

    models/registry.json   histórico operacional local (este módulo)
    data/monitoring.db     eventos de runtime: predições e avaliações
    mlflow.db              experimentos offline (params, métricas, artefatos)

Semântica do campo `metrics` (importante ao ler o JSON direto):

    stage="initial_training"
        métricas do conjunto de teste, medidas por
        training/train_initial.py.

    stage="incremental_update"
        métricas do lote rotulado recebido, medidas ANTES da
        atualização. Descrevem o desempenho dos pesos ANTERIORES sobre
        aqueles dados, não o desempenho do modelo depois de treinar.
        O campo `notes` repete isso em texto.

O campo `is_current` marca a versão operacional ativa, isto é, a que
corresponde ao arquivo em models/model.pt.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Caminho padrão do registro. Configurável por variável de ambiente,
# igual a MODEL_PATH e DB_PATH, para que os três se movam juntos.
REGISTRY_PATH = Path(os.getenv("REGISTRY_PATH", "models/registry.json"))


def _resolve(registry_path: Optional[Path]) -> Path:
    """Resolve o caminho do registro no momento da chamada.

    Por que não usar `registry_path: Path = REGISTRY_PATH` na
    assinatura: em Python, o valor default de um parâmetro é avaliado
    uma única vez, quando a função é definida. O default ficaria preso
    ao objeto Path original, e substituir `registry.REGISTRY_PATH`
    depois — o que os testes fazem via monkeypatch — não teria efeito
    nenhum. Na prática, os testes gravariam no models/registry.json
    real do repositório, que é um arquivo versionado e faz parte do
    artefato entregue.

    Lendo a variável de módulo aqui dentro, a substituição funciona.
    """
    if registry_path is not None:
        return Path(registry_path)
    return Path(REGISTRY_PATH)


def _load(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def register(
    version: int,
    model_path: str,
    stage: str,
    n_samples: int,
    metrics: Optional[Dict] = None,
    notes: str = "",
    registry_path: Optional[Path] = None,
) -> Dict:
    """Adiciona uma entrada ao registro e marca a versão como atual.

    Args:
        version: Número da versão do modelo.
        model_path: Caminho do arquivo `.pt`.
        stage: "initial_training" ou "incremental_update".
        n_samples: Amostras usadas nesta etapa.
        metrics: Objeto plano de métricas (f1, precision, recall, ...).
            Ver a nota sobre semântica no topo do módulo: em
            atualizações incrementais são as métricas do lote ANTES do
            treino.
        notes: Observação livre, em texto.
        registry_path: Onde gravar. Se None, usa `REGISTRY_PATH`.

    Returns:
        A entrada recém-criada.
    """
    path = _resolve(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = _load(path)

    # Só uma entrada por vez carrega is_current: é ela que indica qual
    # versão está de fato no arquivo models/model.pt.
    for entry in entries:
        entry["is_current"] = False

    new_entry = {
        "version": version,
        "model_path": str(model_path),
        "stage": stage,
        "n_samples": n_samples,
        "metrics": metrics or {},
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_current": True,
    }
    entries.append(new_entry)
    path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return new_entry


def current(registry_path: Optional[Path] = None) -> Optional[Dict]:
    """Devolve a entrada marcada como atual, ou None se o registro estiver vazio.

    Se nenhuma entrada tiver `is_current` — um registro editado à mão,
    por exemplo — cai para a última entrada, que é a mais recente.
    """
    entries = _load(_resolve(registry_path))
    for entry in entries:
        if entry.get("is_current"):
            return entry
    return entries[-1] if entries else None


def history(registry_path: Optional[Path] = None) -> List[Dict]:
    """Devolve todas as versões registradas, em ordem cronológica."""
    return _load(_resolve(registry_path))

"""Registra operacionalmente as versões do modelo.

Este módulo usa `models/registry.json` como um registro local de versões,
em vez de depender do MLflow Model Registry.

A escolha é de escopo: o Model Registry exigiria um backend, um servidor
em execução e uma dependência adicional da API. Se esse serviço falhasse,
a API também poderia ser afetada. Como o modelo em produção é um arquivo
`.pt` local, o JSON já é suficiente para registrar a versão atual, a data,
a quantidade de amostras e as métricas.

O MLflow continua sendo usado para acompanhar o treinamento inicial offline,
onde registra parâmetros, métricas e artefatos.

Divisão de responsabilidades:

    models/registry.json  Histórico operacional local de versões.
    data/monitoring.db    Predições e avaliações geradas em runtime.
    mlflow.db             Runs do treinamento inicial.

Semântica de `metrics`:

    stage="initial_training"
        Métricas do conjunto de teste calculadas por
        `training/train_initial.py`.

    stage="incremental_update"
        Métricas do lote rotulado calculadas antes da atualização.
        Elas descrevem o desempenho dos pesos anteriores sobre o lote,
        não o desempenho do modelo após o treinamento incremental.

O campo `notes` registra essa distinção, e `is_current` identifica a
versão operacional ativa correspondente a `models/model.pt`.
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

    O caminho não é definido como valor padrão na assinatura porque
    valores padrão são avaliados quando a função é criada.

    Se fosse usado `registry_path: Path = REGISTRY_PATH`, o parâmetro
    continuaria apontando para o `Path` original mesmo depois de
    `registry.REGISTRY_PATH` ser substituído pelos testes via
    `monkeypatch`.

    Consultar `REGISTRY_PATH` dentro da função permite que os testes
    redirecionem o registro para um diretório temporário, evitando
    alterações no arquivo versionado do repositório.
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
    """Adiciona uma versão ao registro e marca-a como atual.

    Args:
        version: Número da versão do modelo.
        model_path: Caminho do checkpoint `.pt`.
        stage: Etapa que criou a versão. Pode ser `"initial_training"` ou `"incremental_update"`.
        n_samples: Número de amostras usadas nessa etapa.
        metrics: Dicionário simples com as métricas calculadas, como
            `f1`, `precision` e `recall`. Em atualizações incrementais,
            representa o desempenho do lote antes do treinamento.
        notes: Observação adicional sobre a versão.
        registry_path: Caminho do arquivo de registro. Se `None`, usa `REGISTRY_PATH`.

    Returns:
        A entrada criada no registro.
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
    """Retorna a versão operacional atual do registro.

    Devolve a entrada com `is_current=True` ou `None` se o registro estiver
    vazio.

    Se nenhuma entrada estiver marcada como atual, por exemplo, após uma
    edição manual do arquivo, usa a última entrada registrada, assumindo
    que ela representa a versão mais recente.
    """
    entries = _load(_resolve(registry_path))
    for entry in entries:
        if entry.get("is_current"):
            return entry
    return entries[-1] if entries else None


def history(registry_path: Optional[Path] = None) -> List[Dict]:
    """Retorna todas as versões registradas em ordem cronológica."""
    return _load(_resolve(registry_path))

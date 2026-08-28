"""Registro de versões de modelo.

Substitui o MLflow Model Registry por um arquivo JSON.

A justificativa é de escopo: o Model Registry do MLflow exige um
backend store (SQLite ou banco), um servidor rodando e a API passando
a depender dele em tempo de execução. Neste projeto o "modelo em
produção" é um arquivo `.pt` no disco, e o que realmente precisamos
saber é: qual é a versão atual, quando foi criada, com quantas
amostras e com que desempenho.

Um JSON responde a isso em 60 linhas e continua legível por humanos.
O MLflow segue sendo usado para *tracking* de experimentos, que é onde
ele agrega de verdade.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REGISTRY_PATH = Path("models/registry.json")


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
    registry_path: Path = REGISTRY_PATH,
) -> Dict:
    """Adiciona uma entrada ao registro e marca a versão como atual.

    Args:
        version: Número da versão do modelo.
        model_path: Caminho do arquivo `.pt`.
        stage: "initial_training" ou "incremental_update".
        n_samples: Amostras usadas nesta etapa.
        metrics: Métricas associadas, se houver.
        notes: Observação livre.
        registry_path: Onde gravar o JSON.

    Returns:
        A entrada recém-criada.
    """
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    entries = _load(registry_path)
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
    registry_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return new_entry


def current(registry_path: Path = REGISTRY_PATH) -> Optional[Dict]:
    """Devolve a entrada marcada como atual, ou None se o registro estiver vazio."""
    entries = _load(Path(registry_path))
    for entry in entries:
        if entry.get("is_current"):
            return entry
    return entries[-1] if entries else None


def history(registry_path: Path = REGISTRY_PATH) -> List[Dict]:
    """Devolve todas as versões registradas, em ordem cronológica."""
    return _load(Path(registry_path))

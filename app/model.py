"""Modelo MLP em PyTorch com suporte a aprendizado incremental.

A classe `RainModel` encapsula tudo que precisa ser preservado entre
uma sessão e a próxima:

    - os pesos da rede;
    - o estado do otimizador (o Adam guarda médias móveis dos gradientes);
    - o scaler, congelado no treino inicial;
    - o limiar de decisão calibrado;
    - metadados de versão.

Tudo isso vai para um único arquivo `.pt`, o que torna o checkpoint
autocontido: quem carrega o arquivo não precisa de mais nada para
prever ou continuar treinando.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from app.features import FEATURE_NAMES

# --------------------------------------------------------------------------
# Hiperparâmetros padrão
# --------------------------------------------------------------------------

HIDDEN_SIZE: int = 16
LR_INITIAL: float = 1e-3      # treino inicial
LR_UPDATE: float = 1e-4       # atualização incremental (10x menor)
EPOCHS_INITIAL: int = 30
EPOCHS_UPDATE: int = 2
BATCH_SIZE: int = 256
WEIGHT_DECAY: float = 1e-4
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Fixa as sementes para tornar o treino reprodutível."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# A rede
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """Rede densa mínima: entrada -> 16 -> ReLU -> 1.

    A saída é o *logit*, não a probabilidade. O sigmoid é aplicado
    apenas na inferência. Isso permite usar `BCEWithLogitsLoss`, que
    combina sigmoid e log de forma numericamente estável.
    """

    def __init__(self, n_features: int, hidden: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------
# Scaler congelado
# --------------------------------------------------------------------------

@dataclass
class FrozenScaler:
    """Padronização com estatísticas fixas.

    Diferente do `StandardScaler` adaptativo do River, este scaler é
    calculado uma única vez no treino inicial e nunca mais muda.

    O motivo é direto: os pesos da rede foram aprendidos em um espaço
    de features com determinada média e escala. Se as estatísticas de
    normalização mudarem, esse espaço muda junto e os pesos existentes
    deixam de ser válidos. Congelar o scaler é o que permite continuar
    o treinamento em vez de recomeçar.
    """

    mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    std: np.ndarray = field(default_factory=lambda: np.ones(1))

    def fit(self, X: np.ndarray) -> "FrozenScaler":
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Evita divisão por zero em features constantes.
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


# --------------------------------------------------------------------------
# Wrapper principal
# --------------------------------------------------------------------------

class RainModel:
    """Modelo de previsão de chuva com treino inicial e atualização incremental.

    Attributes:
        net: A rede MLP.
        optimizer: Adam. Seu estado é salvo e restaurado junto com os pesos.
        scaler: Padronização congelada.
        threshold: Limiar de decisão calibrado em validação.
        version: Incrementa a cada atualização incremental.
        n_updates: Quantidade de chamadas a `incremental_fit`.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        hidden: int = HIDDEN_SIZE,
        lr: float = LR_INITIAL,
        seed: int = SEED,
    ) -> None:
        set_seed(seed)
        self.feature_names: List[str] = list(feature_names or FEATURE_NAMES)
        self.hidden = hidden
        self.net = MLP(len(self.feature_names), hidden)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
        )
        self.scaler = FrozenScaler()
        self.pos_weight: float = 1.0
        self.threshold: float = 0.5
        self.version: int = 1
        self.n_updates: int = 0
        self.n_samples_seen: int = 0
        self.history: List[Dict] = []

    # ---------------------------------------------------------------- utils

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Aplica o scaler e converte para tensor."""
        return torch.tensor(self.scaler.transform(X), dtype=torch.float32)

    def _loss_fn(self) -> nn.Module:
        """Loss com peso na classe positiva, para lidar com o desbalanceamento.

        `pos_weight` é calculado no treino inicial e reutilizado nas
        atualizações. Recalculá-lo a partir de um lote pequeno seria
        instável: um lote sem nenhum positivo produziria um peso infinito.
        """
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.pos_weight, dtype=torch.float32)
        )

    def _train_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        shuffle: bool = True,
    ) -> float:
        """Laço de treino compartilhado por `fit` e `incremental_fit`."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        Xt = self._to_tensor(X)
        yt = torch.tensor(y, dtype=torch.float32)
        loss_fn = self._loss_fn()
        n = len(Xt)
        last_loss = 0.0

        self.net.train()
        for _ in range(epochs):
            order = torch.randperm(n) if shuffle else torch.arange(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                self.optimizer.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(idx)
            last_loss = epoch_loss / n

        self.n_samples_seen += n
        return last_loss

    # ----------------------------------------------------------- treino

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = EPOCHS_INITIAL,
        batch_size: int = BATCH_SIZE,
        lr: float = LR_INITIAL,
    ) -> float:
        """Treino inicial, do zero.

        Ajusta o scaler, calcula o peso da classe positiva e treina a
        rede. Só deve ser chamado uma vez, na criação do modelo.

        Args:
            X: Matriz de features, na ordem de `self.feature_names`.
            y: Vetor binário do alvo.
            epochs: Épocas de treino.
            batch_size: Tamanho do lote.
            lr: Taxa de aprendizado.

        Returns:
            A loss média da última época.
        """
        self.scaler.fit(X)

        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        self.pos_weight = n_neg / max(n_pos, 1.0)

        loss = self._train_loop(X, y, epochs, batch_size, lr)
        self.history.append({
            "event": "fit",
            "version": self.version,
            "n_samples": int(len(X)),
            "epochs": epochs,
            "lr": lr,
            "loss": round(loss, 5),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return loss

    def incremental_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = EPOCHS_UPDATE,
        batch_size: int = BATCH_SIZE,
        lr: float = LR_UPDATE,
    ) -> float:
        """Continua o treinamento a partir dos pesos atuais.

        O que este método NÃO faz, e é justamente o ponto:

            - não recria a rede;
            - não reinicializa os pesos;
            - não recria o otimizador (o estado do Adam é preservado);
            - não recalcula o scaler.

        A rede segue exatamente de onde parou. As únicas diferenças em
        relação ao `fit` são a taxa de aprendizado, dez vezes menor, e
        o número reduzido de épocas — ambas para evitar que o lote novo
        sobrescreva o que o modelo já aprendia antes.

        Args:
            X: Features das novas observações rotuladas.
            y: Rótulos correspondentes.
            epochs: Épocas sobre o lote novo. Manter baixo.
            batch_size: Tamanho do lote.
            lr: Taxa de aprendizado da atualização.

        Returns:
            A loss média da última época.

        Raises:
            RuntimeError: Se o modelo ainda não passou por `fit`.
        """
        if self.scaler.mean.shape[0] != len(self.feature_names):
            raise RuntimeError(
                "O modelo precisa passar por fit() antes de atualizações "
                "incrementais: o scaler ainda não foi ajustado."
            )

        loss = self._train_loop(X, y, epochs, batch_size, lr)

        self.n_updates += 1
        self.version += 1
        self.history.append({
            "event": "incremental_fit",
            "version": self.version,
            "n_samples": int(len(X)),
            "epochs": epochs,
            "lr": lr,
            "loss": round(loss, 5),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return loss

    # -------------------------------------------------------- inferência

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Devolve a probabilidade da classe positiva."""
        self.net.eval()
        logits = self.net(self._to_tensor(X))
        return torch.sigmoid(logits).numpy()

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Devolve a previsão binária aplicando o limiar de decisão."""
        thr = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= thr).astype(int)

    def calibrate_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        grid: Optional[np.ndarray] = None,
    ) -> float:
        """Escolhe o limiar que maximiza o F1 em um conjunto de validação.

        Importante: deve ser chamado com dados de validação, nunca com
        os dados de teste. Escolher o limiar olhando o teste infla o
        resultado final.

        Args:
            X: Features de validação.
            y: Rótulos de validação.
            grid: Limiares a testar. Padrão: 0.05 a 0.95, passo 0.01.

        Returns:
            O limiar escolhido, já gravado em `self.threshold`.
        """
        grid = np.arange(0.05, 0.95, 0.01) if grid is None else grid
        probs = self.predict_proba(X)

        best_thr, best_f1 = 0.5, -1.0
        for thr in grid:
            pred = (probs >= thr).astype(int)
            tp = float(((pred == 1) & (y == 1)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fn = float(((pred == 0) & (y == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            if f1 > best_f1:
                best_thr, best_f1 = float(thr), f1

        self.threshold = best_thr
        return best_thr

    # ------------------------------------------------------- persistência

    def save(self, path: str | Path) -> Path:
        """Salva o checkpoint completo em um único arquivo `.pt`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_mean": self.scaler.mean,
                "scaler_std": self.scaler.std,
                "feature_names": self.feature_names,
                "hidden": self.hidden,
                "pos_weight": self.pos_weight,
                "threshold": self.threshold,
                "version": self.version,
                "n_updates": self.n_updates,
                "n_samples_seen": self.n_samples_seen,
                "history": self.history,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RainModel":
        """Recarrega um modelo salvo, pronto para prever ou continuar treinando.

        Restaura também o estado do otimizador. Sem isso, o Adam
        recomeçaria com as médias móveis zeradas e os primeiros passos
        após o reload seriam erráticos — o modelo pioraria logo depois
        de ser carregado, sem motivo aparente.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model = cls(
            feature_names=ckpt["feature_names"],
            hidden=ckpt["hidden"],
        )
        model.net.load_state_dict(ckpt["model_state_dict"])
        model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model.scaler = FrozenScaler(
            mean=np.asarray(ckpt["scaler_mean"]),
            std=np.asarray(ckpt["scaler_std"]),
        )
        model.pos_weight = ckpt["pos_weight"]
        model.threshold = ckpt["threshold"]
        model.version = ckpt["version"]
        model.n_updates = ckpt["n_updates"]
        model.n_samples_seen = ckpt["n_samples_seen"]
        model.history = ckpt.get("history", [])
        return model

    # -------------------------------------------------------- metadados

    def info(self) -> Dict:
        """Resumo do estado do modelo, usado pelos endpoints da API."""
        return {
            "version": self.version,
            "n_updates": self.n_updates,
            "n_samples_seen": self.n_samples_seen,
            "n_features": len(self.feature_names),
            "hidden_size": self.hidden,
            "threshold": round(self.threshold, 4),
            "n_parameters": sum(p.numel() for p in self.net.parameters()),
        }

    def __repr__(self) -> str:
        return f"RainModel({json.dumps(self.info())})"

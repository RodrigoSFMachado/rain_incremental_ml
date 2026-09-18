"""Modelo MLP em PyTorch com suporte a aprendizado incremental.

`RainModel` reúne os componentes necessários para prever e continuar
treinando o modelo entre diferentes sessões:

- pesos da rede;
- estado do otimizador Adam;
- scaler usado no treinamento inicial;
- limiar de decisão;
- metadados da versão.

Esses dados são salvos em um único arquivo `.pt`, tornando o checkpoint
autocontido para predição e treinamento incremental.

Termos usados no módulo:

- `logit`: saída bruta da rede, no intervalo `(-inf, +inf)`;
- `probability`: resultado de `sigmoid(logit)`, entre `0` e `1`;
- `prediction`: decisão binária baseada em `probability >= threshold`.

`predict_proba` retorna a probabilidade. `predict` retorna a decisão
binária. A API expõe ambas, junto com o limiar utilizado, permitindo
que o cliente aplique outro corte se necessário.
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
    """Fixa as sementes para tornar os experimentos reproduzíveis.

    `np.random.seed` e `torch.manual_seed` alteram o estado aleatório
    global do processo, não apenas o estado deste objeto.

    Como `RainModel.__init__` chama esta função, criar ou carregar um
    modelo reinicia a sequência aleatória usada pelo programa. Isso
    também pode afetar o embaralhamento dos lotes em treinamentos
    posteriores.

    O comportamento é determinístico, mas a semente não é isolada por
    instância.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# A rede
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """Rede neural densa mínima: entrada -> 16 -> ReLU -> 1.

    A camada final retorna o logit, não uma probabilidade. O sigmoid é
    aplicado somente durante a inferência.

    No treinamento, a saída é usada diretamente com `BCEWithLogitsLoss`,
    que combina sigmoid e cálculo do log de forma numericamente estável.
    Aplicar sigmoid separadamente antes da loss pode causar perda de
    precisão quando os logits têm valores muito altos ou muito baixos.
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
    """Padroniza as features usando estatísticas fixas.

    O scaler é ajustado uma única vez no treinamento inicial e permanece
    inalterado durante as atualizações incrementais.

    A rede aprendeu pesos em um espaço definido pela média e pelo desvio
    padrão originais. Se essas estatísticas mudarem, a escala das entradas
    também muda e os pesos deixam de representar o mesmo problema.

    Manter o scaler congelado preserva a compatibilidade entre o treinamento
    inicial, as predições e o aprendizado incremental, evitando a necessidade
    de recriar o modelo.
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

    A classe reúne a rede, o estado do treinamento e os metadados necessários
    para realizar predições, salvar checkpoints e continuar aprendendo.

    Attributes:
        net: Rede neural MLP.
        optimizer: Otimizador Adam, com estado salvo e restaurado junto aos pesos.
        scaler: Scaler ajustado no treino inicial e mantido fixo.
        threshold: Limiar de decisão calibrado no conjunto de validação.
        version: Versão atual do modelo, incrementada a cada atualização incremental.
        n_updates: Número de chamadas a `incremental_fit`.
        n_samples_seen: Número de amostras usadas no treinamento ao longo da vida do modelo,
            incluindo `fit` e todas as chamadas a `incremental_fit`. Não inclui predições
            servidas; esse histórico é mantido no SQLite, na tabela `predictions`.
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
        """Define a função de perda com peso para a classe positiva.

        Como os casos de chuva representam cerca de 5% dos dados, os exemplos
        negativos podem dominar o gradiente. Sem `pos_weight`, a rede poderia
        aprender a prever sempre "não vai chover".

        O `pos_weight` é calculado no treinamento inicial e reutilizado nas
        atualizações incrementais. Recalculá-lo em cada lote seria instável:
        um lote sem exemplos positivos produziria um peso indefinido ou infinito.
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
        """Executa o laço de treinamento usado por `fit` e `incremental_fit`.

        O mesmo procedimento é utilizado no treino inicial e nas atualizações
        incrementais. A diferença é que o treino incremental começa com os
        pesos, o estado do otimizador e o scaler já preservados.

        Assim, aprendizado incremental não é um algoritmo separado: é a aplicação
        de novos passos de gradiente sobre o estado existente do modelo.
        """
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
        """Executa o treinamento inicial do modelo.

        O método deve ser chamado uma única vez, quando o modelo é criado.
        Ele ajusta o scaler, calcula o peso da classe positiva e treina a rede
        por várias épocas.

        Diferentemente de `incremental_fit`, começa com o modelo sem estado
        treinado e define os componentes usados pelas atualizações futuras.

        Args:
            X: Matriz de features na ordem de `self.feature_names`.
            y: Vetor binário do alvo.
            epochs: Número de épocas de treinamento.
            batch_size: Tamanho dos lotes.
            lr: Taxa de aprendizado.

        Returns:
            Loss média da última época.
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
        """Continua o treinamento a partir do estado atual do modelo.

        O método aplica novos passos de gradiente sem recriar ou reinicializar
        os componentes aprendidos:

        - não recria a rede nem reinicializa seus pesos;
        - preserva o estado do otimizador Adam;
        - não recalcula o scaler;
        - não recalibra o limiar de decisão.

        O modelo continua exatamente de onde parou. Em relação ao `fit`, usa
        uma taxa de aprendizado menor e menos épocas para reduzir o risco de o
        novo lote sobrescrever o conhecimento anterior.

        Args:
            X: Features das novas observações rotuladas.
            y: Rótulos correspondentes.
            epochs: Número de épocas sobre o novo lote.
            batch_size: Tamanho dos lotes.
            lr: Taxa de aprendizado da atualização.

        Returns:
            Loss média da última época.

        Raises:
            RuntimeError: Se o treinamento inicial ainda não foi executado.
        """
        # O scaler nunca ajustado ainda tem shape (1,), o default do
        # dataclass. É o sinal de que `fit` não rodou — e treinar sem
        # scaler produziria pesos em uma escala que nada mais reconhece.
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
        """Retorna a probabilidade estimada da classe positiva, entre 0 e 1.

        Como a função de perda usa um `pos_weight` elevado, a saída não deve ser
        interpretada como uma probabilidade calibrada. Por exemplo, `0.84` não
        significa necessariamente 84% de chance de chuva.

        A saída é mais adequada para ordenar os casos por nível de risco, como
        na avaliação por ROC-AUC. Para interpretar os valores como probabilidades
        reais, seria necessária uma etapa adicional de calibração.
        """
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
        """Escolhe o limiar que maximiza o F1 no conjunto de validação.

        O método deve ser executado apenas com dados de validação. O conjunto de
        teste não pode ser usado para escolher o limiar, pois isso transforma o
        teste em parte do treinamento e produz uma estimativa otimista.

        O valor `0.5` não é necessariamente adequado neste projeto. A classe
        positiva representa cerca de 5% dos dados e a loss usa `pos_weight`, o
        que altera a distribuição das saídas. Por isso, o limiar é ajustado
        explicitamente; neste projeto, ficou próximo de `0.84`.

        Args:
            X: Features do conjunto de validação.
            y: Rótulos do conjunto de validação.
            grid: Limiar a testar. O padrão cobre `0.05` a `0.94`,
                com passo de `0.01`.

        Returns:
            O limiar escolhido e armazenado em `self.threshold`.
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
        """Salva o estado completo do modelo em um único arquivo `.pt`.

        O checkpoint inclui mais do que os pesos da rede. Ele armazena tudo o
        que é necessário para restaurar o modelo e continuar o treinamento:

        - pesos da rede;
        - estado do otimizador;
        - scaler;
        - limiar de decisão;
        - versão e demais metadados.

        Salvar apenas o `state_dict()` permitiria fazer predições, mas perderia
        o estado do otimizador e os parâmetros usados no pré-processamento e na
        decisão. Nesse caso, o aprendizado incremental deixaria de continuar
        corretamente.
        """
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
        """"Carrega um checkpoint confiável para predição ou treino incremental.

        Restaura os pesos da rede, o estado do otimizador e os metadados
        necessários para continuar o treinamento.

        `weights_only=False` é usado porque o checkpoint contém arrays NumPy e
        outros objetos além de tensores. Como essa opção usa desserialização
        completa via pickle, carregue apenas checkpoints produzidos e controlados
        pelo próprio projeto.
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
        """Retorna um resumo do estado atual do modelo.

        Usado pela API para expor a versão, o histórico de atualizações,
        a quantidade de amostras usadas no treinamento e a configuração
        básica da rede.
        """
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

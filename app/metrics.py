"""Cálculo de métricas de classificação binária.

Decisão central deste módulo: em vez de guardar a métrica já calculada,
guardamos a matriz de confusão. Isso permite reagregar depois por
qualquer recorte (por ano, por cenário, acumulado) sem recalcular nada
— e evita o erro de tirar "média de F1s", que não é o F1 do conjunto.

Por que essa distinção importa: uma janela com 3 chuvas e outra com 90
pesam igual em uma média simples. Somando os quadrantes e derivando o
F1 no fim, cada observação pesa o que deve.

Sobre a escolha da métrica principal: com ~5% de positivos, a acurácia
sozinha não diz nada. Um modelo que responde "não vai chover" sempre
acerta 95% das horas e tem recall zero — é inútil e parece ótimo. O F1
combina precisão (dos alertas emitidos, quantos se confirmaram) e
recall (das chuvas ocorridas, quantas foram previstas), e só é alto
quando as duas são. A ROC-AUC mede coisa diferente: a capacidade de
ordenar o risco, independente de qualquer limiar. Um modelo pode ter
AUC alta e F1 baixo se o limiar estiver mal escolhido.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def confusion_summary(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, int]:
    """Conta os quatro quadrantes da matriz de confusão.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_pred: Previsões binárias (0/1), já com o limiar aplicado.

    Returns:
        Dicionário com as chaves "tp", "tn", "fp" e "fn".
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
    }


def compute_metrics(cm: Dict[str, int]) -> Dict[str, float]:
    """Deriva acurácia, precisão, recall e F1 a partir da matriz de confusão.

    Args:
        cm: Saída de `confusion_summary`, ou qualquer soma de várias delas.

    Returns:
        Dicionário com "accuracy", "precision", "recall" e "f1".

    Notes:
        Os denominadores são checados um a um. Um lote de duas semanas
        sem nenhuma chuva produz tp = fn = 0 e recall indefinido; aqui
        vira 0.0 em vez de exceção, porque a agregação precisa
        continuar rodando.
    """
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def roc_auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    """Calcula a AUC-ROC pelo método dos postos (estatística de Mann-Whitney).

    Implementado à mão para não adicionar scikit-learn às dependências
    da API, que precisa ser leve. A equivalência é conhecida: a AUC é a
    probabilidade de um positivo sorteado ao acaso receber score maior
    que um negativo sorteado ao acaso.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_score: Probabilidades previstas (não as decisões binárias).

    Returns:
        A área sob a curva ROC. Devolve 0.5 se houver apenas uma classe,
        que é o valor de um classificador aleatório.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Postos médios, tratando empates. Sem esse tratamento, scores
    # idênticos receberiam postos diferentes por ordem de chegada e a
    # AUC ficaria dependente da ordenação do array.
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1

    sum_pos_ranks = ranks[y_true == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    """Avalia um conjunto completo: métricas de limiar + AUC + matriz.

    Args:
        y_true: Rótulos verdadeiros.
        y_prob: Probabilidades previstas.
        threshold: Limiar de decisão.

    Returns:
        Métricas de classificação, ROC-AUC e os quatro quadrantes da
        matriz de confusão, em um único dicionário plano. Os quadrantes
        são inteiros; as demais chaves, floats arredondados.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_summary(y_true, y_pred)
    metrics = compute_metrics(cm)
    metrics["roc_auc"] = round(roc_auc(y_true, y_prob), 4)
    metrics.update(cm)
    return metrics

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for highly imbalanced multi-label classification.
    Penalizes well-classified examples (easy negatives) and focuses on hard examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal terms
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce_loss

        if class_weights is not None:
            loss = loss * class_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HierarchicalLoss(nn.Module):
    """
    Penalizes GO DAG violations: P(child) > P(parent).
    Accepts a boolean mask `child_parent_mask` of shape (num_terms, num_terms)
    where mask[i, j] is True if term i is a child of term j.
    """

    def __init__(self, penalty_weight=1.0):
        super().__init__()
        self.weight = penalty_weight

    def forward(
        self, logits: torch.Tensor, child_parent_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: (batch_size, num_terms)
        child_parent_mask: (num_terms, num_terms)
        """
        probs = torch.sigmoid(logits)

        # We want to penalize cases where prob(child) > prob(parent)
        # For efficiency, we expand probs:
        # P_c (batch, num_terms, 1) and P_p (batch, 1, num_terms)
        P_c = probs.unsqueeze(-1)
        P_p = probs.unsqueeze(1)

        # Difference: child_prob - parent_prob
        diff = P_c - P_p

        # Only penalize positive differences (where child > parent) for actual child-parent pairs
        violations = torch.relu(diff) ** 2

        # Apply mask and mean over the batch
        # child_parent_mask should be on the same device
        masked_violations = violations * child_parent_mask

        loss = masked_violations.sum(dim=(1, 2)).mean()
        return self.weight * loss


def multilabel_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Compute true positives, false positives, and false negatives
    for multilabel predictions given logits and binary targets.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        targets = targets.long()

        tp = int(((preds == 1) & (targets == 1)).sum().item())
        fp = int(((preds == 1) & (targets == 0)).sum().item())
        fn = int(((preds == 0) & (targets == 1)).sum().item())

    return tp, fp, fn


def f1_from_stats(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_fmax(
    logits: torch.Tensor, targets: torch.Tensor, num_thresholds: int = 200
) -> Tuple[float, float]:
    """
    Calculates Fmax score commonly used in CAFA protein function prediction challenges.
    Scans over multiple thresholds to find the maximum possible F1 score.
    Returns:
        fmax: The maximum F1 score
        best_threshold: The threshold yielding that maximum score
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    targets = targets.cpu().numpy()

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    fmax = 0.0
    best_th = 0.5

    for th in thresholds:
        preds = (probs >= th).astype(int)

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        # Avoid div by zero
        if tp == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        if f1 > fmax:
            fmax = f1
            best_th = float(th)

    return float(fmax), best_th


def multilabel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[int, int]:
    """
    Compute simple micro-averaged accuracy over all label decisions.
    Returns:
        correct: number of correct (prediction == target) decisions
        total: total number of decisions
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        targets = targets.long()
        correct = int((preds == targets).sum().item())
        total = int(targets.numel())
    return correct, total

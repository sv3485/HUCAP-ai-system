"""Advanced accuracy analysis for HUCAP evaluation pipeline.

Generates per-bin accuracy metrics, risk-coverage curves, and F1 scores
using the correct optimal threshold (NOT hardcoded 0.5).
"""

import json
import numpy as np
from sklearn.metrics import f1_score


def precision_at_k(y_true, y_probs, k=5):
    valid_samples = 0
    total_prec = 0.0
    for i in range(y_true.shape[0]):
        if y_true[i].sum() == 0: continue
        valid_samples += 1
        top_k_idx = np.argsort(y_probs[i])[::-1][:k]
        hits = sum(y_true[i, idx] for idx in top_k_idx)
        total_prec += hits / k
    return total_prec / valid_samples if valid_samples > 0 else 0.0


def generate_risk_coverage(y_true, y_probs, thresholds, pred_threshold=0.35):
    """Generate risk-coverage curve using the correct prediction threshold."""
    max_probs = np.max(y_probs, axis=1)
    y_pred_bin = (y_probs >= pred_threshold).astype(int)
    
    results = []
    for t in sorted(thresholds):
        mask = max_probs >= t
        coverage = np.mean(mask)
        if mask.sum() > 0:
            acc = f1_score(y_true[mask], y_pred_bin[mask], average='micro', zero_division=0)
        else:
            acc = 0.0
        results.append({
            "threshold": float(t),
            "coverage": float(coverage),
            "accuracy": float(acc)
        })
    return results


def bin_metric(metric_values, y_true, y_probs, bin_edges, pred_threshold=0.35, bin_labels=None):
    """Bin accuracy metric using the correct prediction threshold."""
    y_pred_bin = (y_probs >= pred_threshold).astype(int)
    results = []
    
    for i in range(len(bin_edges)-1):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (metric_values >= low) & (metric_values < high)
        count = mask.sum()
        if count > 0:
            acc = f1_score(y_true[mask], y_pred_bin[mask], average='micro', zero_division=0)
        else:
            acc = 0.0
            
        label = bin_labels[i] if bin_labels else f"{low:.2f}-{high:.2f}"
        if i == len(bin_edges)-2 and not bin_labels:
            label = f">= {low:.2f}"
            
        results.append({"bin": label, "accuracy": float(acc), "count": int(count)})
    return results


def run_accuracy_analysis(y_true, y_probs, entropies, confidences, uacs, threshold=0.35):
    """Run full accuracy analysis using the given prediction threshold.

    Args:
        threshold: The optimal binarization threshold from training.
                   Previously hardcoded to 0.5, causing F1=0.0 because
                   filtered probabilities rarely exceed 0.5.
    """
    y_pred_bin = (y_probs >= threshold).astype(int)
    
    micro_f1 = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    
    print(f"[AccuracyAnalysis] threshold={threshold:.3f}  micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}")
    
    top_1 = precision_at_k(y_true, y_probs, k=1)
    top_3 = precision_at_k(y_true, y_probs, k=3)
    top_5 = precision_at_k(y_true, y_probs, k=5)
    
    entropy_bins = [0.0, 3.80, 4.00, 4.30, 5.0]
    entropy_labels = ["Low (<3.8)", "Medium (3.8-4.0)", "High (4.0-4.3)", "Max (>4.3)"]
    
    conf_bins = [0.0, 0.40, 0.60, 1.01]
    conf_labels = ["Low (<40%)", "Medium (40-60%)", "High (>60%)"]
    
    uac_bins = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
    
    acc_by_entropy = bin_metric(entropies, y_true, y_probs, entropy_bins, pred_threshold=threshold, bin_labels=entropy_labels)
    acc_by_conf = bin_metric(confidences, y_true, y_probs, conf_bins, pred_threshold=threshold, bin_labels=conf_labels)
    acc_by_uac = bin_metric(uacs, y_true, y_probs, uac_bins, pred_threshold=threshold)
    
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    risk_cov = generate_risk_coverage(y_true, y_probs, thresholds, pred_threshold=threshold)
    
    # Eval specifically at 0.45 rejection threshold (HUCAP Default)
    max_probs = np.max(y_probs, axis=1)
    covered_mask = max_probs >= 0.45
    coverage = float(np.mean(covered_mask))
    
    acc_before = float(micro_f1)
    if covered_mask.sum() > 0:
        acc_after = float(f1_score(y_true[covered_mask], y_pred_bin[covered_mask], average='micro', zero_division=0))
    else:
        acc_after = 0.0
    
    res = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "optimal_threshold": float(threshold),
        "top_k_accuracy": {
            "top_1": float(top_1),
            "top_3": float(top_3),
            "top_5": float(top_5),
        },
        "coverage": coverage,
        "rejection_rate": float(1.0 - coverage),
        "accuracy_before_rejection": float(acc_before),
        "accuracy_after_rejection": float(acc_after),
        "accuracy_by_entropy": acc_by_entropy,
        "accuracy_by_confidence": acc_by_conf,
        "accuracy_by_uac": acc_by_uac,
        "risk_coverage": risk_cov,
        "data_source": "evaluation_pipeline"
    }
    return res

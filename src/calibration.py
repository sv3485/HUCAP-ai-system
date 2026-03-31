"""
Calibration & Reliability Evaluation Module.

Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Brier Score. Also provides functions for generating Reliability Diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import json

def compute_calibration_metrics(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Computes ECE, MCE, and Brier Score for binary/multi-label predictions.
    
    Args:
        confidences: 1D array of predicted probabilities/confidences [0, 1].
        accuracies:  1D array of binary true labels (0 or 1).
        n_bins:      Number of bins for binning confidences.
        
    Returns:
        Dict spanning 'ECE', 'MCE', 'Brier'.
    """
    if len(confidences) == 0:
        return {"ECE": 0.0, "MCE": 0.0, "Brier": 0.0, "bins": []}
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    mce = 0.0
    total = len(confidences)
    
    brier_score = float(np.mean((confidences - accuracies) ** 2))
    
    bin_data = []
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
            
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(accuracies[mask].mean())
        bin_count = int(mask.sum())
        
        weight = bin_count / total
        diff = abs(bin_acc - bin_conf)
        
        ece += weight * diff
        mce = max(mce, diff)
        
        bin_data.append({
            "range": f"{bin_boundaries[i]:.2f}-{bin_boundaries[i+1]:.2f}",
            "conf": bin_conf,
            "acc": bin_acc,
            "count": bin_count
        })
        
    return {
        "ECE": float(ece),
        "MCE": float(mce),
        "Brier": brier_score,
        "bins": bin_data
    }

def plot_reliability_diagram(
    metrics_dict: Dict[str, float],
    save_path: str = "results/plots/reliability_diagram.png"
):
    """
    Plots a standardized reliability diagram (Accuracy vs Confidence).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    bins = metrics_dict.get("bins", [])
    if not bins:
        return
        
    confs = [b["conf"] for b in bins]
    accs = [b["acc"] for b in bins]
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.plot(confs, accs, marker='o', color='blue', label="Model Calibration")
    
    # Optional bar chart overlay for distribution
    fractions = [b["count"] / sum(x["count"] for x in bins) for b in bins]
    plt.bar(confs, fractions, width=0.08, color='gray', alpha=0.3, label="Data Distribution")
    
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Observed Accuracy (Fraction of Positives)")
    plt.title(f"Reliability Diagram\nECE: {metrics_dict['ECE']:.4f}  |  MCE: {metrics_dict['MCE']:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability diagram to {save_path}")

def generate_calibration_report(
    flat_confidences: List[float], 
    flat_labels: List[int],
    save_path: str = "results/calibration_report.json"
) -> Dict:
    """Convenience wrapper for evaluation pipelines."""
    conf_arr = np.array(flat_confidences)
    acc_arr = np.array(flat_labels)
    
    metrics = compute_calibration_metrics(conf_arr, acc_arr)
    plot_reliability_diagram(metrics, save_path.replace(".json", ".png"))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics

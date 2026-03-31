"""
Entropy vs. Confidence Analysis
Generates scatter plots and correlation metrics to prove that prediction
confidence scales dynamically with sequence uncertainty.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr, spearmanr
import os

from src.config import data_config
from src.transformer_model import ProteinTransformerClassifier
from src.data_preprocessing import load_dataset_cache, split_dataset
from src.uncertainty import classify_complexity, adjust_confidence


def main():
    print("Loading test dataset splits...")
    dataset, term_vocabs = load_dataset_cache("outputs/preprocessed_dataset.json")
    train, val, test = split_dataset(dataset, seed=42)

    # We only need a subset for scatter plot visuals (e.g. 500 seqs)
    if len(test) > 500:
        test = test[:500]

    print("Loading final 35M Checkpoint...")
    model_name = "facebook/esm2_t12_35M_UR50D"
    device = torch.device("cpu")
    ckpt = torch.load("models/protein_transformer_multitask.pt", map_location="cpu", weights_only=False)

    num_labels_dict = {"F": len(term_vocabs.get("F", []))}
    model = ProteinTransformerClassifier(model_name, num_labels_dict)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    train_thresh = float(ckpt.get("best_threshold", 0.35))
    T = 1.35

    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    def calibrate_confidence(raw_prob: float, threshold: float) -> float:
        if raw_prob >= threshold:
            return 0.5 + 0.5 * ((raw_prob - threshold) / (1.0 - threshold))
        else:
            return 0.5 * (raw_prob / threshold)

    entropies = []
    top_confidences = []

    print(f"Running inference on {len(test)} sequences...")
    with torch.no_grad():
        for i, data in enumerate(test):
            seq = data["sequence"]
            if len(seq) > 2000:
                seq = seq[:2000]

            seq_type, _, entropy, _ = classify_complexity(seq)

            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=2000).to(device)
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            logits = outputs["F"].squeeze(0).cpu().numpy()

            # Apply temp scaling + sigmoid
            probs = 1.0 / (1.0 + np.exp(-logits / T))
            top_raw_prob = float(np.max(probs))

            # Calibration + Scale
            scaled_conf = calibrate_confidence(top_raw_prob, train_thresh)
            if seq_type != "structured":
                scaled_conf = adjust_confidence(scaled_conf, entropy)

            entropies.append(entropy)
            top_confidences.append(scaled_conf)

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(test)}...")

    # Compute correlation
    e_arr = np.array(entropies)
    c_arr = np.array(top_confidences)

    pearson_e, pval_e = pearsonr(e_arr, c_arr)
    spearman_e, _ = spearmanr(e_arr, c_arr)

    print("\n=== CORRELATION RESULTS ===")
    print(f"Entropy vs Confidence (Pearson):  {pearson_e:.4f} (p={pval_e:.2e})")
    print(f"Entropy vs Confidence (Spearman): {spearman_e:.4f}")

    # Plot
    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(e_arr, c_arr, alpha=0.5, color='royalblue', edgecolor='w', s=30)

    # Regression line
    m, b = np.polyfit(e_arr, c_arr, 1)
    x_line = np.linspace(e_arr.min(), e_arr.max(), 100)
    plt.plot(x_line, m * x_line + b, color='red', linewidth=2, linestyle='--',
             label=f'Trend (r={pearson_e:.2f})')

    plt.axvline(x=3.80, color='gray', linestyle=':', label='Low-Complexity Threshold (P10)')
    plt.axvline(x=3.85, color='orange', linestyle=':', alpha=0.6, label='Medium Threshold (P30)')

    plt.xlabel("Sequence Entropy (bits)", fontsize=12)
    plt.ylabel("Top Predicted Confidence", fontsize=12)
    plt.title("Entropy vs. Uncertainty-Scaled Confidence", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/entropy_vs_confidence.png", dpi=300)
    plt.close()

    # Save correlation report
    report = {
        "pearson_r": float(pearson_e),
        "pearson_p": float(pval_e),
        "spearman_r": float(spearman_e),
        "n_samples": len(test),
        "entropy_range": [float(e_arr.min()), float(e_arr.max())],
        "confidence_range": [float(c_arr.min()), float(c_arr.max())]
    }
    with open("results/entropy_confidence_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved plot to results/plots/entropy_vs_confidence.png")
    print("Saved report to results/entropy_confidence_report.json")


if __name__ == "__main__":
    main()

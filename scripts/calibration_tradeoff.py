"""
Calibration vs Performance Trade-off Study.

Sweeps temperature T from 0.5 to 5.0 and plots:
- ECE vs AUPRC
- ECE vs Fmax
to find the Pareto-optimal calibration strength.
"""

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import EsmTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from src.config import data_config
from src.transformer_model import ProteinTransformerClassifier
from src.metrics import calculate_fmax
from src.data_preprocessing import load_dataset_cache, split_dataset
from src.calibration import compute_calibration_metrics
from src.train_transformer import ProteinSequenceDataset


def _build_labels(entries, term_vocab):
    term_to_idx = {t: i for i, t in enumerate(term_vocab)}
    labels = {}
    for entry in entries:
        pid = entry["protein_id"]
        vec = np.zeros(len(term_vocab), dtype=np.float32)
        for term in entry.get("go_terms", {}).get("F", []):
            idx = term_to_idx.get(term)
            if idx is not None:
                vec[idx] = 1.0
        labels[pid] = vec
    return labels


def main():
    print("Calibration vs Performance Trade-off Study")
    print("=" * 50)

    dataset, term_vocabs = load_dataset_cache("outputs/preprocessed_dataset.json")
    train, val, test = split_dataset(dataset, seed=42)

    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    device = torch.device("cpu")

    ckpt = torch.load("models/protein_transformer_multitask.pt", map_location="cpu", weights_only=False)
    num_labels_dict = {"F": len(term_vocabs.get("F", []))}
    model = ProteinTransformerClassifier(model_name, num_labels_dict)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    entry_ids = [d["protein_id"] for d in test]
    sequences = {d["protein_id"]: d["sequence"] for d in test}
    labels_f = _build_labels(test, term_vocabs["F"])
    test_ds = ProteinSequenceDataset(entry_ids, sequences, {"F": labels_f}, tokenizer, data_config.max_seq_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_ds, batch_size=4, collate_fn=collator)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_logits.append(outputs["F"].cpu())
            all_targets.append(batch["labels"].cpu())

    logits = torch.cat(all_logits, 0).numpy()
    targets = torch.cat(all_targets, 0).numpy()

    # Sweep temperature
    temperatures = np.linspace(0.5, 5.0, 30)
    results = []

    for T in temperatures:
        probs = 1.0 / (1.0 + np.exp(-logits / T))

        # Top-5 filtering
        filtered = np.zeros_like(probs)
        for i in range(probs.shape[0]):
            idx = np.argsort(probs[i])[::-1][:5]
            for j in idx:
                filtered[i, j] = probs[i, j]

        fmax, _ = calculate_fmax(torch.tensor(logits), torch.tensor(targets))
        auprc = float(average_precision_score(targets, filtered, average='micro'))

        flat_p, flat_y = filtered.flatten(), targets.flatten()
        mask = flat_p > 0
        cal = compute_calibration_metrics(flat_p[mask], flat_y[mask])

        results.append({
            "T": round(float(T), 3),
            "Fmax": float(fmax),
            "AUPRC": round(auprc, 4),
            "ECE": round(float(cal["ECE"]), 4),
            "Brier": round(float(cal["Brier"]), 4),
        })

    # Find Pareto optimal
    best_idx = min(range(len(results)), key=lambda i: results[i]["ECE"])
    print(f"\nPareto-optimal T={results[best_idx]['T']:.3f}: "
          f"ECE={results[best_idx]['ECE']:.4f}, AUPRC={results[best_idx]['AUPRC']:.4f}")

    # Plot
    os.makedirs("results/plots", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ts = [r["T"] for r in results]
    eces = [r["ECE"] for r in results]
    auprcs = [r["AUPRC"] for r in results]
    briers = [r["Brier"] for r in results]

    # ECE vs AUPRC
    ax1.plot(eces, auprcs, 'o-', color='royalblue', markersize=4)
    ax1.scatter([results[best_idx]["ECE"]], [results[best_idx]["AUPRC"]],
               color='red', s=100, zorder=5, label=f'Optimal T={results[best_idx]["T"]:.2f}')
    ax1.set_xlabel("ECE (Calibration Error)", fontsize=12)
    ax1.set_ylabel("AUPRC", fontsize=12)
    ax1.set_title("Calibration vs Ranking Performance", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Temperature vs Metrics
    ax2.plot(ts, eces, 'o-', color='crimson', markersize=4, label='ECE')
    ax2.plot(ts, briers, 's-', color='orange', markersize=4, label='Brier')
    ax2.plot(ts, auprcs, '^-', color='royalblue', markersize=4, label='AUPRC')
    ax2.axvline(x=results[best_idx]["T"], color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel("Temperature T", fontsize=12)
    ax2.set_ylabel("Metric Value", fontsize=12)
    ax2.set_title("Temperature Sweep", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/calibration_tradeoff.png", dpi=300)
    plt.close()
    print("Saved plot to results/plots/calibration_tradeoff.png")

    with open("results/calibration_tradeoff.json", "w") as f:
        json.dump({"sweep": results, "pareto_optimal": results[best_idx]}, f, indent=2)
    print("Saved data to results/calibration_tradeoff.json")


if __name__ == "__main__":
    main()

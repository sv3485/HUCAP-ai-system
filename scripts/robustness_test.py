"""
Robustness & Generalization Testing.

Tests system stability under adversarial/degraded conditions:
1. Random amino acid mutations (noise injection)
2. Sequence truncation (partial sequences)
3. Synthetic low-complexity inputs (polyQ, poly-Ala, repeats)

Measures performance drop and calibration stability.
"""

import json
import os
import random
import numpy as np
import torch
from transformers import EsmTokenizer

from src.transformer_model import ProteinTransformerClassifier
from src.data_preprocessing import load_dataset_cache, split_dataset
from src.uncertainty import classify_complexity, calculate_sequence_entropy

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def mutate_sequence(seq, mutation_rate=0.1):
    """Introduce random point mutations."""
    seq = list(seq)
    n_mut = max(1, int(len(seq) * mutation_rate))
    positions = random.sample(range(len(seq)), min(n_mut, len(seq)))
    for pos in positions:
        seq[pos] = random.choice(AMINO_ACIDS)
    return "".join(seq)


def truncate_sequence(seq, keep_fraction=0.5):
    """Keep only a fraction of the sequence."""
    keep = max(5, int(len(seq) * keep_fraction))
    return seq[:keep]


def generate_synthetic_lc(length=200):
    """Generate synthetic low-complexity sequences."""
    patterns = [
        "Q" * length,                                    # Poly-Q
        "A" * length,                                    # Poly-Ala
        ("AG" * (length // 2))[:length],                 # Dipeptide repeat
        ("AGAG" * (length // 4))[:length],               # Tetrapeptide repeat
        "".join(random.choices("AQ", k=length)),         # Binary composition
    ]
    return patterns


def main():
    print("=" * 60)
    print("ROBUSTNESS & GENERALIZATION TESTING")
    print("=" * 60)

    dataset, term_vocabs = load_dataset_cache("outputs/preprocessed_dataset.json")
    _, _, test = split_dataset(dataset, seed=42)

    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    device = torch.device("cpu")

    ckpt = torch.load("models/protein_transformer_multitask.pt", map_location="cpu", weights_only=False)
    num_labels_dict = {"F": len(term_vocabs.get("F", []))}
    model = ProteinTransformerClassifier(model_name, num_labels_dict)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    T = 1.35
    subset = test[:50]  # Use 50 sequences for speed

    def predict_single(seq):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=2000)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
        logits = outputs["F"].squeeze(0).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits / T))
        return probs

    # ── 1. Baseline predictions ─────────────────────────────────────────
    print("\n── Baseline Predictions ──")
    baseline_confs = []
    for d in subset:
        probs = predict_single(d["sequence"])
        baseline_confs.append(float(np.max(probs)))
    baseline_mean = np.mean(baseline_confs)
    print(f"  Mean top confidence: {baseline_mean:.4f}")

    # ── 2. Mutation Robustness ──────────────────────────────────────────
    print("\n── Mutation Robustness ──")
    mutation_results = {}
    for rate in [0.05, 0.10, 0.20, 0.30]:
        mut_confs = []
        conf_changes = []
        for i, d in enumerate(subset):
            mut_seq = mutate_sequence(d["sequence"], mutation_rate=rate)
            probs = predict_single(mut_seq)
            top_conf = float(np.max(probs))
            mut_confs.append(top_conf)
            conf_changes.append(abs(top_conf - baseline_confs[i]))

        mean_conf = np.mean(mut_confs)
        mean_change = np.mean(conf_changes)
        mutation_results[f"{int(rate*100)}%"] = {
            "mean_confidence": round(mean_conf, 4),
            "mean_change": round(mean_change, 4),
            "drop_from_baseline": round(baseline_mean - mean_conf, 4),
        }
        print(f"  {rate*100:.0f}% mutation: conf={mean_conf:.4f}, Δ={mean_change:.4f}")

    # ── 3. Truncation Robustness ────────────────────────────────────────
    print("\n── Truncation Robustness ──")
    truncation_results = {}
    for frac in [0.75, 0.50, 0.25]:
        trunc_confs = []
        for i, d in enumerate(subset):
            trunc_seq = truncate_sequence(d["sequence"], keep_fraction=frac)
            if len(trunc_seq) < 5:
                continue
            probs = predict_single(trunc_seq)
            trunc_confs.append(float(np.max(probs)))

        mean_conf = np.mean(trunc_confs)
        truncation_results[f"{int(frac*100)}%_kept"] = {
            "mean_confidence": round(mean_conf, 4),
            "drop_from_baseline": round(baseline_mean - mean_conf, 4),
        }
        print(f"  {frac*100:.0f}% kept: conf={mean_conf:.4f}, drop={baseline_mean - mean_conf:.4f}")

    # ── 4. Synthetic Low-Complexity ─────────────────────────────────────
    print("\n── Synthetic Low-Complexity Inputs ──")
    synthetic_seqs = generate_synthetic_lc(200)
    synthetic_names = ["Poly-Q", "Poly-Ala", "AG-repeat", "AGAG-repeat", "Binary-AQ"]
    synthetic_results = []

    for name, seq in zip(synthetic_names, synthetic_seqs):
        seq_type, unc_level, entropy, complexity = classify_complexity(seq)
        probs = predict_single(seq)
        top_conf = float(np.max(probs))
        result = {
            "name": name,
            "entropy": round(entropy, 3),
            "complexity": round(complexity, 3),
            "seq_type": seq_type,
            "uncertainty": unc_level,
            "top_confidence": round(top_conf, 4),
        }
        synthetic_results.append(result)
        print(f"  {name}: entropy={entropy:.2f}, type={seq_type}, conf={top_conf:.4f}")

    # ── 5. Calibration Stability ────────────────────────────────────────
    print("\n── Calibration Stability Summary ──")
    stability = {
        "baseline_mean_conf": round(baseline_mean, 4),
        "max_conf_drop_mutation": round(max(r["drop_from_baseline"] for r in mutation_results.values()), 4),
        "max_conf_drop_truncation": round(max(r["drop_from_baseline"] for r in truncation_results.values()), 4),
        "synthetic_all_flagged": all(r["seq_type"] != "structured" for r in synthetic_results),
    }
    print(f"  Max mutation drop: {stability['max_conf_drop_mutation']:.4f}")
    print(f"  Max truncation drop: {stability['max_conf_drop_truncation']:.4f}")
    print(f"  All synthetic flagged: {stability['synthetic_all_flagged']}")

    # Save
    os.makedirs("results", exist_ok=True)
    report = {
        "mutation_robustness": mutation_results,
        "truncation_robustness": truncation_results,
        "synthetic_lc": synthetic_results,
        "stability_summary": stability,
    }
    with open("results/robustness_test.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved to results/robustness_test.json")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()

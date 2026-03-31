"""
Statistical Significance Testing for Calibration Improvements.

Validates that calibration improvements are statistically significant using:
1. Paired Wilcoxon signed-rank test (non-parametric)
2. Per-sample calibration error comparison
3. Bootstrap confidence intervals for ECE
"""

import json
import os
import numpy as np
import torch
from scipy import stats
from transformers import EsmTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from src.config import data_config
from src.transformer_model import ProteinTransformerClassifier
from src.data_preprocessing import load_dataset_cache, split_dataset
from src.calibration_advanced import load_calibration_params, apply_temperature_scaling
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


def bootstrap_ece(probs, labels, n_bootstrap=1000, n_bins=10):
    """Bootstrap confidence interval for ECE."""
    n = len(probs)
    eces = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        p_b, y_b = probs[idx], labels[idx]
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (p_b > bins[i]) & (p_b <= bins[i + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(y_b[mask].mean() - p_b[mask].mean())
        eces.append(ece)
    return np.array(eces)


def per_sample_calibration_error(probs, labels, n_bins=10):
    """Compute per-sample calibration error (assigns each sample the error of its bin)."""
    bins = np.linspace(0, 1, n_bins + 1)
    errors = np.zeros(len(probs))
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_error = abs(labels[mask].mean() - probs[mask].mean())
        errors[mask] = bin_error
    return errors


def main():
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTING")
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

    ids = [d["protein_id"] for d in test]
    seqs = {d["protein_id"]: d["sequence"] for d in test}
    labs_f = _build_labels(test, term_vocabs["F"])
    test_ds = ProteinSequenceDataset(ids, seqs, {"F": labs_f}, tokenizer, data_config.max_seq_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_ds, batch_size=4, collate_fn=collator)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_logits.append(outputs["F"].cpu())
            all_targets.append(batch["labels"].cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()

    temperature, iso_cal = load_calibration_params()

    # Compute probabilities for each method
    uncal_probs = 1.0 / (1.0 + np.exp(-logits / 1.35))
    temp_probs = apply_temperature_scaling(logits, temperature)
    iso_probs = iso_cal.predict(uncal_probs)

    flat_y = targets.flatten()
    methods = {
        "Uncalibrated": uncal_probs.flatten(),
        "Temperature": temp_probs.flatten(),
        "Isotonic": iso_probs.flatten(),
    }

    # Filter to non-trivial predictions
    mask = methods["Uncalibrated"] > 0.01
    flat_y_m = flat_y[mask]

    print(f"\nAnalyzing {mask.sum()} non-trivial predictions...")

    # ── 1. Per-Sample Calibration Error ─────────────────────────────────
    print("\n── Per-Sample Calibration Errors ──")
    psce = {}
    for name, probs in methods.items():
        psce[name] = per_sample_calibration_error(probs[mask], flat_y_m)
        print(f"  {name}: mean={psce[name].mean():.4f}, std={psce[name].std():.4f}")

    # ── 2. Wilcoxon Signed-Rank Tests ───────────────────────────────────
    print("\n── Wilcoxon Signed-Rank Tests ──")
    comparisons = [
        ("Uncalibrated", "Temperature"),
        ("Uncalibrated", "Isotonic"),
        ("Temperature", "Isotonic"),
    ]

    sig_results = []
    for a, b in comparisons:
        diff = psce[a] - psce[b]
        nonzero = diff[diff != 0]
        if len(nonzero) < 10:
            print(f"  {a} vs {b}: insufficient non-zero differences")
            continue

        stat, p_value = stats.wilcoxon(nonzero)
        effect_size = abs(diff.mean()) / max(diff.std(), 1e-8)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        result = {
            "comparison": f"{a} vs {b}",
            "wilcoxon_stat": round(float(stat), 2),
            "p_value": float(p_value),
            "significance": sig,
            "mean_improvement": round(float(diff.mean()), 6),
            "cohens_d": round(float(effect_size), 4),
        }
        sig_results.append(result)
        print(f"  {a} vs {b}: W={stat:.1f}, p={p_value:.2e} {sig}, d={effect_size:.3f}")

    # ── 3. Bootstrap ECE Confidence Intervals ───────────────────────────
    print("\n── Bootstrap 95% CI for ECE ──")
    bootstrap_results = {}
    for name, probs in methods.items():
        eces = bootstrap_ece(probs[mask], flat_y_m, n_bootstrap=1000)
        ci_low, ci_high = np.percentile(eces, [2.5, 97.5])
        bootstrap_results[name] = {
            "mean_ece": round(float(eces.mean()), 4),
            "ci_95_lower": round(float(ci_low), 4),
            "ci_95_upper": round(float(ci_high), 4),
        }
        print(f"  {name}: ECE={eces.mean():.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # ── 4. Paired t-test (parametric check) ─────────────────────────────
    print("\n── Paired t-tests ──")
    ttest_results = []
    for a, b in comparisons:
        t_stat, p_value = stats.ttest_rel(psce[a], psce[b])
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        ttest_results.append({
            "comparison": f"{a} vs {b}",
            "t_stat": round(float(t_stat), 4),
            "p_value": float(p_value),
            "significance": sig,
        })
        print(f"  {a} vs {b}: t={t_stat:.3f}, p={p_value:.2e} {sig}")

    # Save
    os.makedirs("results", exist_ok=True)
    report = {
        "wilcoxon_tests": sig_results,
        "paired_ttests": ttest_results,
        "bootstrap_ece_ci": bootstrap_results,
        "n_samples": int(mask.sum()),
    }
    with open("results/statistical_tests.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved to results/statistical_tests.json")


if __name__ == "__main__":
    main()

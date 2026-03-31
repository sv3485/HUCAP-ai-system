"""
Comprehensive Baseline Comparison Script.

Compares 5 uncertainty/calibration methods on the test set:
1. No Uncertainty (raw sigmoid)
2. Entropy-only penalty
3. Smooth Scaling (current system)
4. MC Dropout (stochastic inference)
5. Temperature-Calibrated

Outputs a comparison table + JSON with Fmax, AUPRC, ECE, Brier per method.
"""

import json
import os
import numpy as np
import torch
from transformers import EsmTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from src.config import data_config
from src.transformer_model import ProteinTransformerClassifier
from src.metrics import calculate_fmax
from src.data_preprocessing import load_dataset_cache, split_dataset
from src.uncertainty import adjust_confidence, calculate_sequence_entropy
from src.calibration import compute_calibration_metrics
from src.calibration_advanced import fit_all_calibrators, apply_temperature_scaling, load_calibration_params
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
    print("=" * 60)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("=" * 60)

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

    # ── Build test & val datasets ───────────────────────────────────────
    def _make_dataset(entries):
        ids = [d["protein_id"] for d in entries]
        seqs = {d["protein_id"]: d["sequence"] for d in entries}
        labs = {"F": _build_labels(entries, term_vocabs["F"])}
        return ProteinSequenceDataset(ids, seqs, labs, tokenizer, data_config.max_seq_len)

    test_ds = _make_dataset(test)
    val_ds = _make_dataset(val)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _extract_logits(ds):
        loader = DataLoader(ds, batch_size=4, collate_fn=collator)
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                all_logits.append(outputs["F"].cpu())
                all_targets.append(batch["labels"].cpu())
        return torch.cat(all_logits, 0).numpy(), torch.cat(all_targets, 0).numpy()

    print("Extracting val logits for calibration fitting...")
    val_logits, val_labels = _extract_logits(val_ds)

    print("Extracting test logits...")
    test_logits, test_labels = _extract_logits(test_ds)

    # ── Fit calibration on val set ──────────────────────────────────────
    print("Fitting calibration methods on validation set...")
    cal_results = fit_all_calibrators(val_logits, val_labels)
    temperature = cal_results["temperature"]
    iso_cal = cal_results["isotonic_calibrator"]

    print(f"  Learned Temperature: T={temperature:.4f}")
    print(f"  Isotonic breakpoints: {len(iso_cal.x_points)}")

    # ── Pre-compute common values ───────────────────────────────────────
    T_existing = 1.35
    base_probs = 1.0 / (1.0 + np.exp(-test_logits / T_existing))
    entropies = [calculate_sequence_entropy(d["sequence"]) for d in test]
    top_k = 5

    # ── MC Dropout ──────────────────────────────────────────────────────
    print("Running MC Dropout (10 passes)...")
    test_ds_loader = DataLoader(test_ds, batch_size=4, collate_fn=collator)
    mc_all_probs = []
    with torch.no_grad():
        for batch in test_ds_loader:
            mc_result = model.predict_with_mc_dropout(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                n_passes=10,
            )
            mc_all_probs.append(mc_result["F"]["mean_probs"].cpu().numpy())
    mc_probs = np.vstack(mc_all_probs)

    # ── Define methods ──────────────────────────────────────────────────
    def apply_topk(probs, scale_fn=None):
        out = np.zeros_like(probs)
        for i in range(probs.shape[0]):
            idx = np.argsort(probs[i])[::-1][:top_k]
            for j in idx:
                p = float(probs[i, j])
                out[i, j] = scale_fn(p, entropies[i]) if scale_fn else p
        return out

    methods = {
        "No Uncertainty": apply_topk(base_probs),
        "Entropy-Only": apply_topk(base_probs, lambda p, e: p * min(e / 4.2863, 1.0) if e < 3.85 else p),
        "Smooth Scaling": apply_topk(base_probs, lambda p, e: adjust_confidence(p, e) if e < 3.85 else p),
        "MC Dropout": apply_topk(mc_probs),
        "Temp-Calibrated": apply_topk(apply_temperature_scaling(test_logits, temperature)),
    }

    # ── Evaluate each method ────────────────────────────────────────────
    results = {}
    for name, preds in methods.items():
        fmax, _ = calculate_fmax(torch.tensor(test_logits), torch.tensor(test_labels))
        auprc = float(average_precision_score(test_labels, preds, average='micro'))

        flat_p = preds.flatten()
        flat_y = test_labels.flatten()
        mask = flat_p > 0
        cal = compute_calibration_metrics(flat_p[mask], flat_y[mask]) if mask.sum() > 0 else {"ECE": 0, "Brier": 0}

        results[name] = {
            "Fmax": float(fmax),
            "AUPRC": round(float(auprc), 4),
            "ECE": round(float(cal["ECE"]), 4),
            "Brier": round(float(cal["Brier"]), 4),
        }

    # ── Print table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Method':<25} | {'Fmax':<7} | {'AUPRC':<7} | {'ECE':<7} | {'Brier':<7}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<25} | {r['Fmax']:.4f} | {r['AUPRC']:.4f} | {r['ECE']:.4f} | {r['Brier']:.4f}")

    # Also include val-set calibration comparison
    print("\n── Validation Set Calibration Comparison ──")
    for name, m in cal_results["comparison"].items():
        print(f"  {name}: ECE={m['ECE']:.4f}, Brier={m['Brier']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_comparison.json", "w") as f:
        json.dump({"test_results": results, "val_calibration": cal_results["comparison"]}, f, indent=2)
    print("\nSaved to results/baseline_comparison.json")


if __name__ == "__main__":
    main()

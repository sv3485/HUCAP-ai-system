"""
Ablation Study Framework
Evaluates the impact of the Uncertainty Calibration Module by comparing:
1. Baseline (No uncertainty penalization)
2. Hard Penalty (Binary 0.75 multiplier - old system)
3. Smooth Entropy Scaling (Current data-driven system)

Outputs comparative Fmax, AUPRC, and ECE metrics.
"""

import json
import numpy as np
import torch
import os
from transformers import EsmTokenizer

from src.config import data_config
from src.transformer_model import ProteinTransformerClassifier
from src.metrics import calculate_fmax
from src.data_preprocessing import load_dataset_cache, split_dataset
from src.uncertainty import adjust_confidence, calculate_sequence_entropy
from src.calibration import compute_calibration_metrics
from src.train_transformer import ProteinSequenceDataset
from sklearn.metrics import average_precision_score


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
    print("Loading test dataset splits...")
    dataset, term_vocabs = load_dataset_cache("outputs/preprocessed_dataset.json")
    train, val, test = split_dataset(dataset, seed=42)

    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    device = torch.device("cpu")

    print("Loading final 35M Checkpoint...")
    ckpt = torch.load("models/protein_transformer_multitask.pt", map_location="cpu", weights_only=False)

    num_labels_dict = {"F": len(term_vocabs.get("F", []))}
    model = ProteinTransformerClassifier(model_name, num_labels_dict)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    T = 1.35
    test_seqs = [d["sequence"] for d in test]
    entropies = [calculate_sequence_entropy(s) for s in test_seqs]

    # Build dataset and labels
    entry_ids = [d["protein_id"] for d in test]
    sequences = {d["protein_id"]: d["sequence"] for d in test}
    labels_f = _build_labels(test, term_vocabs["F"])
    labels_dict = {"F": labels_f}

    test_dataset = ProteinSequenceDataset(
        entry_ids=entry_ids,
        sequences=sequences,
        labels_dict=labels_dict,
        tokenizer=tokenizer,
        max_length=data_config.max_seq_len,
    )

    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_dataset, batch_size=4, collate_fn=collator)

    # Get raw logits
    print("Extracting raw logits from test set...")
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs["F"]
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Base probabilities
    base_probs = 1.0 / (1.0 + np.exp(-all_logits / T))

    # Define ablation configs
    configs = {
        "Baseline (No Uncertainty)": lambda prob, ent: prob,
        "Hard Penalty (0.75x)": lambda prob, ent: prob * 0.75 if ent < 3.85 else prob,
        "Smooth Scaling (Current)": lambda prob, ent: adjust_confidence(prob, ent) if ent < 3.85 else prob,
    }

    top_k = 5
    results = {}

    print("\nStarting Ablation Evaluation...")
    for config_name, scale_func in configs.items():
        print(f"  Evaluating: {config_name}")

        y_pred = np.zeros_like(base_probs)

        for i in range(base_probs.shape[0]):
            top_idx = np.argsort(base_probs[i])[::-1][:top_k]
            ent = entropies[i]

            for idx in top_idx:
                y_pred[i, idx] = scale_func(float(base_probs[i, idx]), ent)

        # Fmax
        fmax, best_th = calculate_fmax(torch.tensor(all_logits), torch.tensor(all_targets))
        # AUPRC
        auprc = float(average_precision_score(all_targets, y_pred, average='micro'))

        # Calibration (ECE/Brier on non-zero predictions)
        flat_true = all_targets.flatten()
        flat_pred = y_pred.flatten()
        pred_mask = flat_pred > 0
        if pred_mask.sum() > 0:
            cal = compute_calibration_metrics(flat_pred[pred_mask], flat_true[pred_mask])
        else:
            cal = {"ECE": 0.0, "Brier": 0.0}

        results[config_name] = {
            "Fmax": float(fmax),
            "AUPRC": float(auprc),
            "ECE": float(cal["ECE"]),
            "Brier": float(cal["Brier"]),
        }

    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Configuration':<30} | {'Fmax':<7} | {'AUPRC':<7} | {'ECE':<7} | {'Brier':<7}")
    print("-" * 60)

    for name, res in results.items():
        print(f"{name:<30} | {res['Fmax']:.4f} | {res['AUPRC']:.4f} | {res['ECE']:.4f} | {res['Brier']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_study.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full ablation results to results/ablation_study.json")


if __name__ == "__main__":
    main()

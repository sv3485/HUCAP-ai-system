"""
Error Analysis Module.

Identifies and analyzes failure cases:
1. High-confidence false positives (overconfident wrong predictions)
2. Low-entropy false negatives (missed predictions on structured sequences)
3. Pattern analysis across GO terms
"""

import json
import os
import numpy as np
import torch
from transformers import EsmTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from collections import defaultdict

from src.config import data_config
from src.transformer_model import ProteinTransformerClassifier
from src.data_preprocessing import load_dataset_cache, split_dataset
from src.uncertainty import classify_complexity
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
    print("ERROR ANALYSIS MODULE")
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

    entry_ids = [d["protein_id"] for d in test]
    sequences = {d["protein_id"]: d["sequence"] for d in test}
    labels_f = _build_labels(test, term_vocabs["F"])

    test_ds = ProteinSequenceDataset(
        entry_ids, sequences, {"F": labels_f}, tokenizer, data_config.max_seq_len
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_ds, batch_size=4, collate_fn=collator)

    # Extract predictions
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_logits.append(outputs["F"].cpu())
            all_targets.append(batch["labels"].cpu())

    logits = torch.cat(all_logits, 0).numpy()
    targets = torch.cat(all_targets, 0).numpy()

    T = 1.35
    probs = 1.0 / (1.0 + np.exp(-logits / T))
    threshold = float(ckpt.get("best_threshold", 0.35))
    preds_bin = (probs >= threshold).astype(int)

    vocab = term_vocabs["F"]

    # ── 1. High-Confidence False Positives ──────────────────────────────
    print("\n── High-Confidence False Positives ──")
    fp_cases = []
    for i in range(len(test)):
        fp_idx = np.where((preds_bin[i] == 1) & (targets[i] == 0))[0]
        for idx in fp_idx:
            fp_cases.append({
                "protein_id": test[i]["protein_id"],
                "predicted_term": vocab[idx],
                "confidence": round(float(probs[i, idx]), 4),
                "sequence_length": len(test[i]["sequence"]),
                "entropy": round(classify_complexity(test[i]["sequence"])[2], 3),
                "seq_type": classify_complexity(test[i]["sequence"])[0],
            })

    fp_cases.sort(key=lambda x: x["confidence"], reverse=True)
    top_fp = fp_cases[:10]
    print(f"  Total FP: {len(fp_cases)}")
    for i, fp in enumerate(top_fp):
        print(f"  {i+1}. {fp['protein_id']} → {fp['predicted_term']} "
              f"(conf={fp['confidence']:.3f}, entropy={fp['entropy']:.2f}, type={fp['seq_type']})")

    # ── 2. High-Confidence False Negatives ──────────────────────────────
    print("\n── Missed Predictions (False Negatives on Structured Sequences) ──")
    fn_cases = []
    for i in range(len(test)):
        seq_type = classify_complexity(test[i]["sequence"])[0]
        if seq_type != "structured":
            continue
        fn_idx = np.where((preds_bin[i] == 0) & (targets[i] == 1))[0]
        for idx in fn_idx:
            fn_cases.append({
                "protein_id": test[i]["protein_id"],
                "missed_term": vocab[idx],
                "raw_prob": round(float(probs[i, idx]), 4),
                "entropy": round(classify_complexity(test[i]["sequence"])[2], 3),
            })

    fn_cases.sort(key=lambda x: x["raw_prob"], reverse=True)
    top_fn = fn_cases[:10]
    print(f"  Total FN on structured: {len(fn_cases)}")
    for i, fn in enumerate(top_fn):
        print(f"  {i+1}. {fn['protein_id']} missed {fn['missed_term']} "
              f"(prob={fn['raw_prob']:.3f})")

    # ── 3. Per-Term Error Rate ──────────────────────────────────────────
    print("\n── Per-Term Error Rates ──")
    term_fp = defaultdict(int)
    term_fn = defaultdict(int)
    term_total = defaultdict(int)

    for i in range(len(test)):
        for j in range(len(vocab)):
            if targets[i, j] == 1:
                term_total[vocab[j]] += 1
            if preds_bin[i, j] == 1 and targets[i, j] == 0:
                term_fp[vocab[j]] += 1
            if preds_bin[i, j] == 0 and targets[i, j] == 1:
                term_fn[vocab[j]] += 1

    term_errors = []
    for term in vocab:
        total = term_total.get(term, 0)
        fpr = term_fp.get(term, 0)
        fnr = term_fn.get(term, 0)
        if total > 0 or fpr > 0:
            term_errors.append({
                "term": term,
                "true_positives": total - fnr,
                "false_positives": fpr,
                "false_negatives": fnr,
                "support": total,
                "error_rate": round((fpr + fnr) / max(1, total + fpr), 3),
            })

    term_errors.sort(key=lambda x: x["error_rate"], reverse=True)
    top_errors = term_errors[:10]
    for te in top_errors:
        print(f"  {te['term']}: FP={te['false_positives']}, FN={te['false_negatives']}, "
              f"support={te['support']}, error_rate={te['error_rate']}")

    # ── 4. Complexity vs Error Analysis ─────────────────────────────────
    print("\n── Error Rate by Sequence Type ──")
    type_stats = defaultdict(lambda: {"total": 0, "errors": 0})
    for i in range(len(test)):
        seq_type = classify_complexity(test[i]["sequence"])[0]
        n_errors = int(np.sum(np.abs(preds_bin[i] - targets[i])))
        type_stats[seq_type]["total"] += 1
        type_stats[seq_type]["errors"] += n_errors

    for stype, stats in type_stats.items():
        avg_err = stats["errors"] / max(1, stats["total"])
        print(f"  {stype}: {stats['total']} seqs, avg errors/seq={avg_err:.1f}")

    # ── Save report ─────────────────────────────────────────────────────
    report = {
        "top_false_positives": top_fp,
        "top_false_negatives": top_fn,
        "top_error_terms": top_errors,
        "error_by_sequence_type": {k: v for k, v in type_stats.items()},
        "total_fp": len(fp_cases),
        "total_fn_structured": len(fn_cases),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/error_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved to results/error_analysis.json")


if __name__ == "__main__":
    main()

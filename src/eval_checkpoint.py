import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import EsmTokenizer
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score

from src.data_preprocessing import load_dataset_cache, split_dataset
from src.transformer_model import ProteinTransformerClassifier
from src.metrics import calculate_fmax
from src.config import data_config, train_config
from src.train_transformer import ProteinSequenceDataset


def precision_at_k(y_true, y_pred, k=5):
    N = y_true.shape[0]
    total_prec = 0.0
    valid_samples = 0
    for i in range(N):
        if y_true[i].sum() == 0:
            continue
        valid_samples += 1
        top_k_idx = np.argsort(y_pred[i])[::-1][:k]
        hits = sum(y_true[i, idx] for idx in top_k_idx)
        total_prec += hits / k
    return total_prec / valid_samples if valid_samples > 0 else 0.0


def _build_labels_from_cache(entries, term_vocab):
    """Build multi-hot label vectors directly from cached JSON dataset entries."""
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
    cache_file = "outputs/preprocessed_dataset.json"
    dataset, term_vocabs = load_dataset_cache(cache_file)
    if len(dataset) < 40000:
        raise ValueError("Dataset not fully loaded")
    train, val, test = split_dataset(dataset, seed=42)

    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    print("Building test dataset...")
    entry_ids = [d["protein_id"] for d in test]
    sequences = {d["protein_id"]: d["sequence"] for d in test}
    num_labels_dict = {"F": len(term_vocabs.get("F", []))}

    # Build labels directly from cached GO term annotations
    labels_f = _build_labels_from_cache(test, term_vocabs["F"])
    labels_dict = {"F": labels_f}

    test_dataset = ProteinSequenceDataset(
        entry_ids=entry_ids,
        sequences=sequences,
        labels_dict=labels_dict,
        tokenizer=tokenizer,
        max_length=data_config.max_seq_len,
    )

    print("Loading final 35M Checkpoint...")
    model_path = "models/protein_transformer_multitask.pt"
    if not os.path.exists(model_path):
        print("Model file not found. Please wait for training to finish.")
        return

    model = ProteinTransformerClassifier(model_name, num_labels_dict)

    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Manual evaluation over test set
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print("Executing final predictions...")
    all_logits = []
    all_targets = []

    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_dataset, batch_size=4, collate_fn=collator)

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

    # APPLY POST-PROCESSING (Matched to backend/app.py)
    # 1. Temperature Scaling
    T = 1.35
    probs = 1.0 / (1.0 + np.exp(-all_logits / T))

    # Apply sequence complexity analysis using unified module
    from src.uncertainty import classify_complexity, adjust_confidence
    test_sequences = [d["sequence"] for d in test]
    complexity_data = [classify_complexity(seq) for seq in test_sequences]
    # complexity_data[i] = (seq_type, uncertainty, entropy, complexity_score)

    # 2. Top-K Thresholding & Target Suppression
    top_k = 5
    filtered_probs = np.zeros_like(probs)
    
    for i in range(probs.shape[0]):
        # Get top K indices
        top_idx = np.argsort(probs[i])[::-1][:top_k]
        
        seq_type_i, _, entropy_i, _ = complexity_data[i]

        for idx in top_idx:
            prob = probs[i, idx]
            term = term_vocabs["F"][idx]
            
            # Smooth entropy-aware confidence scaling (replaces hard 0.75)
            if seq_type_i != "structured":
                prob = adjust_confidence(prob, entropy_i)

            # Systematic False Positive Soft-Filtering
            if prob < 0.45 and term in ["GO:0003677", "GO:0010333", "GO:0016491", "GO:0004497"]:
                continue
                
            # Biological Context penalty
            if term == "GO:0005515" and prob < 0.8:
                prob = prob * 0.95
                
            filtered_probs[i, idx] = prob

    probs = filtered_probs

    # ── HUCAP Advanced Accuracy Analysis ──
    from src.uncertainty import calculate_uac
    entropies = np.array([comp[2] for comp in complexity_data])
    confidences = np.max(probs, axis=1)
    uacs = np.array([calculate_uac(confidences[i], entropies[i]) for i in range(len(confidences))])

    # Calculate optimal threshold FIRST (needed for accuracy analysis)
    fmax, best_th = calculate_fmax(torch.tensor(all_logits), torch.tensor(all_targets))

    from src.accuracy_analysis import run_accuracy_analysis
    accuracy_stats = run_accuracy_analysis(all_targets, probs, entropies, confidences, uacs, threshold=float(best_th))
    with open("results/accuracy_analysis.json", "w") as f:
        json.dump(accuracy_stats, f, indent=4)
    print("Saved advanced Accuracy Analysis to results/accuracy_analysis.json")

    # Calculate PR Curve
    precision, recall, _ = precision_recall_curve(all_targets.ravel(), probs.ravel())

    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='indigo', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.grid(True, alpha=0.3)
    plt.savefig("results/plots/pr_curve.png", dpi=300)
    plt.close()
    print("Saved PR Curve to results/plots/pr_curve.png")

    # Calculate Advanced Metrics
    fmax, best_th = calculate_fmax(torch.tensor(all_logits), torch.tensor(all_targets))
    auprc = average_precision_score(all_targets, probs, average='micro')

    preds_bin = (probs >= best_th).astype(int)
    micro_f1 = f1_score(all_targets, preds_bin, average='micro')
    if micro_f1 == 0:
        raise ValueError("Training failed — F1 is zero")
    macro_f1 = f1_score(all_targets, preds_bin, average='macro', zero_division=0)

    p3 = precision_at_k(all_targets, probs, k=3)
    p5 = precision_at_k(all_targets, probs, k=5)

    # ── Calibration Metrics (XAI/Reliability) ──
    from src.calibration import compute_calibration_metrics
    pred_mask = probs.flatten() > 0  # Ignore massive 0-class imbalance
    cal_metrics = compute_calibration_metrics(probs.flatten()[pred_mask], all_targets.flatten()[pred_mask])

    metrics = {
        "Test_Fmax": float(fmax),
        "Test_AUPRC": float(auprc),
        "Test_Micro_F1": float(micro_f1),
        "Test_Macro_F1": float(macro_f1),
        "Test_Precision@3": float(p3),
        "Test_Precision@5": float(p5),
        "Test_ECE": float(cal_metrics["ECE"]),
        "Test_Brier": float(cal_metrics["Brier"]),
        "Best_Threshold": float(best_th),
        "Seed": 42
    }

    with open("results/advanced_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved evaluation metrics to results/advanced_metrics.json")
    print(json.dumps(metrics, indent=2))

    # Error Analysis
    print("Running Error Analysis...")
    from collections import defaultdict
    mispredicted_freqs = defaultdict(int)
    confusion = defaultdict(int)

    voc = term_vocabs["F"]

    for i in range(len(all_targets)):
        pred = preds_bin[i]
        true = all_targets[i]

        fp = np.where((pred == 1) & (true == 0))[0]
        fn = np.where((pred == 0) & (true == 1))[0]

        for p in fp:
            mispredicted_freqs[voc[p]] += 1
        for n in fn:
            mispredicted_freqs[voc[n]] += 1

        for p in fp:
            for n in fn:
                confusion[f"{voc[n]} -> {voc[p]}"] += 1

    top_errors = sorted(mispredicted_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
    top_confusions = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]

    err_data = {
        "Most_Mispredicted_Terms": dict(top_errors),
        "Top_Confusion_Pairs": dict(top_confusions)
    }

    with open("results/enhanced_error_analysis.json", "w") as f:
        json.dump(err_data, f, indent=4)
    print("Saved Error Analysis to results/enhanced_error_analysis.json")

    # Plot Training Loss if logs exist
    import glob
    logs = glob.glob("logs/experiment_*.json")
    if logs:
        latest_log = max(logs, key=os.path.getctime)
        with open(latest_log, "r") as f:
            data = json.load(f)
            history = data.get("metrics_per_epoch", [])

            epochs = []
            losses = []

            for h in history:
                if 'loss' in h and 'epoch' in h:
                    epochs.append(h['epoch'])
                    losses.append(h['loss'])

            if epochs:
                plt.figure(figsize=(8, 6))
                plt.plot(epochs, losses, color='royalblue', lw=2)
                plt.xlabel('Epochs')
                plt.ylabel('Training Loss')
                plt.title('Epoch vs Training Loss')
                plt.grid(True, alpha=0.3)
                plt.savefig("results/plots/training_loss.png", dpi=300)
                plt.close()
                print("Saved Training Loss Curve to results/plots/training_loss.png")


if __name__ == "__main__":
    main()


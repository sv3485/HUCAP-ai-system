"""Multi-task Transformer training pipeline for protein function prediction."""

import json
import logging
import os
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .config import DataConfig, data_config, paths, train_config
from .data import (
    SequenceAugmenter,
    build_label_matrix,
    load_fasta_sequences,
    load_train_terms,
)
from .metrics import FocalLoss, HierarchicalLoss, f1_from_stats, calculate_fmax
from .transformer_model import ProteinTransformerClassifier
from .utils import get_device, set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class ProteinSequenceDataset(Dataset):
    """
    Dataset yielding raw amino acid sequences and multi-hot label vectors for multiple aspects.
    """

    def __init__(
        self,
        entry_ids: List[str],
        sequences: Dict[str, str],
        labels_dict: Dict[str, Dict[str, np.ndarray]],
        tokenizer: AutoTokenizer,
        max_length: int,
        augmenter: SequenceAugmenter = None,
    ) -> None:
        self.entry_ids = entry_ids
        self.sequences = sequences
        self.labels_dict = labels_dict  # aspect -> {entry_id -> np.array}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.entry_ids)

    def __getitem__(self, idx: int):
        entry_id = self.entry_ids[idx]
        seq = self.sequences[entry_id]

        if self.augmenter is not None:
            seq = self.augmenter(seq)

        encoding = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        for aspect, labels in self.labels_dict.items():
            if entry_id in labels:
                item[f"labels_{aspect}"] = torch.from_numpy(labels[entry_id]).float()
            else:
                # If no labels, fill with zeros
                shape = len(next(iter(labels.values())))
                item[f"labels_{aspect}"] = torch.zeros(shape, dtype=torch.float32)

        # HF Trainer inherently expects 'labels' key to exist for evaluation loops
        if "labels_F" in item:
            item["labels"] = item["labels_F"]

        return item


def build_multitask_datasets(
    config: DataConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[Dataset, Dataset, Dict]:
    train_dir = os.path.join(config.dataset_root, "Train")
    train_terms_path = os.path.join(train_dir, "train_terms.tsv")
    train_fasta_path = os.path.join(train_dir, "train_sequences.fasta")

    sequences = load_fasta_sequences(train_fasta_path)

    aspects = ["F"]
    labels_dict = {}
    term_vocabs = {}
    term_to_idxs = {}

    valid_entry_ids = set(sequences.keys())

    for aspect in aspects:
        df_terms, term_vocab = load_train_terms(
            train_terms_path=train_terms_path,
            aspect=aspect,
            min_term_frequency=config.min_term_frequency,
        )
        labels, term_to_idx = build_label_matrix(df_terms, term_vocab)
        labels_dict[aspect] = labels
        term_vocabs[aspect] = term_vocab
        term_to_idxs[aspect] = term_to_idx
        valid_entry_ids = valid_entry_ids.intersection(labels.keys())

    valid_entry_ids = sorted(list(valid_entry_ids))
    logger.info(
        "Total sequences with all three GO branch annotations: %d", len(valid_entry_ids)
    )

    # Optional Data Augmentation for training
    augmenter = SequenceAugmenter(mutation_prob=0.03, mask_prob=0.01)

    dataset = ProteinSequenceDataset(
        valid_entry_ids,
        sequences,
        labels_dict,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        augmenter=None,  # Will apply augmentations only to train dataset below
    )

    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Attach augmenter to train_dataset manually
    train_dataset.dataset.augmenter = augmenter

    # Slice a manageable subset if dataset is very large
    max_train = min(config.max_train_samples, len(train_dataset))
    max_val = min(500, len(val_dataset))
    train_dataset = torch.utils.data.Subset(train_dataset, range(max_train))
    val_dataset = torch.utils.data.Subset(val_dataset, range(max_val))

    metadata = {
        "term_vocabs": term_vocabs,
        "term_to_idxs": term_to_idxs,
    }
    return train_dataset, val_dataset, metadata


# ---------------------------------------------------------------------------
# Swiss-Prot + GAF dataset builder (NEW)
# ---------------------------------------------------------------------------

def build_swissprot_datasets(
    config: DataConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[Dataset, Dataset, Dict]:
    """
    Build training and validation datasets from Swiss-Prot FASTA + GAF files
    using the new preprocessing pipeline.
    """
    from .data_preprocessing import (
        load_or_build_dataset,
        split_dataset,
        _cache_path,
    )

    cache_file = _cache_path(paths.project_root)

    # Run the full preprocessing pipeline (or load from cache)
    dataset, term_vocabs = load_or_build_dataset(
        fasta_path=config.fasta_path,
        gaf_path=config.gaf_path,
        cache_file=cache_file,
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len_filter,
        max_go_terms=config.max_go_terms_per_protein,
        top_n_go_terms=int(config.top_n_go_terms),
        min_term_frequency=config.min_term_frequency,
    )

    logger.info("Preprocessed dataset: %d proteins", len(dataset))
    print("Total samples:", len(dataset))
    for aspect, vocab in term_vocabs.items():
        logger.info("  Aspect %s: %d GO terms", aspect, len(vocab))

    # Save dataset stats
    stats_path = os.path.join(paths.outputs_dir, "dataset_stats.json")
    try:
        lengths = [len(e["sequence"]) for e in dataset]
        labels_per_protein = [sum(len(v) for v in e["go_terms"].values()) for e in dataset]
        stats = {
            "total_proteins": len(dataset),
            "go_term_distribution": {aspect: len(vocab) for aspect, vocab in term_vocabs.items()},
            "avg_labels_per_protein": float(np.mean(labels_per_protein)) if labels_per_protein else 0.0,
            "seq_length_min": int(np.min(lengths)) if lengths else 0,
            "seq_length_max": int(np.max(lengths)) if lengths else 0,
            "seq_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        logger.info("Saved dataset_stats.json to %s", stats_path)
    except Exception as e:
        logger.warning(f"Could not save dataset_stats: {e}")

    # Split into train / val / test
    train_data, val_data, test_data = split_dataset(
        dataset,
        train_ratio=1.0 - config.val_ratio - config.test_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=train_config.random_seed,
    )

    # Build term_to_idx mappings
    term_to_idxs = {
        aspect: {t: i for i, t in enumerate(vocab)}
        for aspect, vocab in term_vocabs.items()
    }

    # Convert to label matrices
    def _build_labels(
        data_split: List[dict],
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, np.ndarray]]]:
        sequences = {}
        labels_dict: Dict[str, Dict[str, np.ndarray]] = {a: {} for a in ["F"]}

        for entry in data_split:
            pid = entry["protein_id"]
            sequences[pid] = entry["sequence"]

            for aspect in ["F"]:
                vec = np.zeros(len(term_vocabs.get(aspect, [])), dtype=np.float32)
                for term in entry["go_terms"].get(aspect, []):
                    idx = term_to_idxs.get(aspect, {}).get(term)
                    if idx is not None:
                        vec[idx] = 1.0
                labels_dict[aspect][pid] = vec

        return sequences, labels_dict

    train_sequences, train_labels = _build_labels(train_data)
    val_sequences, val_labels = _build_labels(val_data)

    # Filter to proteins that have at least F-aspect labels
    train_ids = sorted([
        pid for pid in train_sequences
        if train_labels["F"][pid].sum() > 0
    ])
    val_ids = sorted([
        pid for pid in val_sequences
        if val_labels["F"][pid].sum() > 0
    ])

    logger.info("Train proteins with labels: %d", len(train_ids))
    logger.info("Val proteins with labels: %d", len(val_ids))

    # Create augmenter
    augmenter = SequenceAugmenter(mutation_prob=0.03, mask_prob=0.01)

    # Build datasets
    train_dataset = ProteinSequenceDataset(
        train_ids,
        train_sequences,
        train_labels,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        augmenter=augmenter,
    )

    val_dataset = ProteinSequenceDataset(
        val_ids,
        val_sequences,
        val_labels,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        augmenter=None,
    )

    # Cap dataset sizes
    max_train = min(config.max_train_samples, len(train_dataset))
    max_val = min(500, len(val_dataset))
    if max_train < len(train_dataset):
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_train))
    if max_val < len(val_dataset):
        val_dataset = torch.utils.data.Subset(val_dataset, range(max_val))

    logger.info("Final train size: %d, val size: %d", len(train_dataset), len(val_dataset))

    metadata = {
        "term_vocabs": term_vocabs,
        "term_to_idxs": term_to_idxs,
    }
    return train_dataset, val_dataset, metadata


class MultilabelTrainer(Trainer):
    def __init__(self, *args, class_weights=None, hierarchy_masks=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        total_loss = 0.0
        # outputs is a dict: aspect -> logits
        for aspect, logits in outputs.items():
            label_key = f"labels_{aspect}"
            if label_key in inputs:
                targets = inputs[label_key]
                loss = self.bce_loss(logits, targets)
                total_loss += loss

        if return_outputs:
            return total_loss, outputs["F"]
        return total_loss


def compute_metrics(eval_pred):
    """Compute Fmax, F1, and AUPRC on the F-aspect predictions."""
    try:
        logits = torch.tensor(eval_pred.predictions)
        labels = torch.tensor(eval_pred.label_ids)

        # Ensure shapes match dynamically across both batch dim (0) and num_terms dim (-1)
        # Mismatches in batch dim happen during distributed evaluation or remainder batches
        if logits.shape[0] != labels.shape[0]:
            min_batch = min(logits.shape[0], labels.shape[0])
            logits = logits[:min_batch, ...]
            labels = labels[:min_batch, ...]
            
        if logits.shape[-1] != labels.shape[-1]:
            min_dim = min(logits.shape[-1], labels.shape[-1])
            logits = logits[..., :min_dim]
            labels = labels[..., :min_dim]

        probs = torch.sigmoid(logits)
        preds = (probs > 0.3).float()

        tp = (preds * labels).sum().item()
        fp = (preds * (1 - labels)).sum().item()
        fn = ((1 - preds) * labels).sum().item()

        f1 = f1_from_stats(tp, fp, fn)
        fmax, best_threshold = calculate_fmax(logits, labels)

        try:
            from sklearn.metrics import average_precision_score

            auprc = average_precision_score(
                labels.numpy(), probs.numpy(), average="micro"
            )
        except (ImportError, ValueError):
            auprc = 0.0

        return {"f1": f1, "fmax": fmax, "auprc": auprc, "best_threshold": best_threshold}
    except Exception as e:
        logger.warning("Error in compute_metrics: %s", e)
        return {"f1": 0.0, "fmax": 0.0, "auprc": 0.0, "best_threshold": 0.5}


class PredictionSanityCheckCallback(TrainerCallback):
    def __init__(self, val_dataset, term_vocabs):
        self.val_dataset = val_dataset
        self.term_vocabs = term_vocabs

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        logger.info("\n--- Prediction Sanity Check after Epoch %s ---", state.epoch)
        samples = [self.val_dataset[0], self.val_dataset[1]]
        with torch.no_grad():
            for i, item in enumerate(samples):
                input_ids = item["input_ids"].unsqueeze(0).to(args.device)
                attention_mask = item["attention_mask"].unsqueeze(0).to(args.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                logits = outputs.get("F", outputs.get(list(outputs.keys())[0]))[0]
                probs = torch.sigmoid(logits)
                top_probs, top_indices = torch.topk(probs, 5)
                
                term_names = [self.term_vocabs["F"][idx] for idx in top_indices.cpu().numpy()]
                prob_vals = top_probs.cpu().numpy()
                formatted_probs = [float(f"{p:.3f}") for p in prob_vals]
                
                logger.info("Sample %d Top 5 F-terms: %s", i, list(zip(term_names, formatted_probs)))
        logger.info("----------------------------------------------\n")



def main() -> None:
    set_seed(train_config.random_seed)
    device = get_device()
    logger.info("Using device: %s", device)
    logger.info("Data source: %s", data_config.data_source)

    # Upgraded base model resolving higher embedding resolution (35M parameters)
    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Build datasets based on data source ---
    if data_config.data_source == "swissprot_gaf":
        logger.info("Building datasets from Swiss-Prot + GAF pipeline...")
        train_dataset, val_dataset, metadata = build_swissprot_datasets(
            data_config, tokenizer
        )
    else:
        logger.info("Building datasets from CAFA-6 pipeline...")
        train_dataset, val_dataset, metadata = build_multitask_datasets(
            data_config, tokenizer
        )

    term_vocabs = metadata["term_vocabs"]
    num_terms_dict = {aspect: len(vocab) for aspect, vocab in term_vocabs.items()}

    logger.info("Number of training proteins: %d", len(train_dataset))
    print("Training samples:", len(train_dataset))
    logger.info("Number of validation proteins: %d", len(val_dataset))
    for a, n in num_terms_dict.items():
        logger.info("GO terms for aspect %s: %d", a, n)

    logger.info("Calculating class weights for Focal Loss...")
    class_weights_dict = {}
    for aspect in ["F"]:
        if aspect not in num_terms_dict or num_terms_dict[aspect] == 0:
            continue
        pos_counts = np.zeros(num_terms_dict[aspect])
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            label_key = f"labels_{aspect}"
            if label_key in item:
                pos_counts += item[label_key].numpy()

        total_samples = len(train_dataset)
        safe_counts = np.maximum(pos_counts, 1.0)
        weights = 1.0 / np.log(safe_counts + 1.0)
        weights = weights / max(1.0, np.mean(weights))
        class_weights_dict[aspect] = torch.tensor(weights, dtype=torch.float32)

    logger.info("Building GO hierarchy masks for Hierarchical Loss...")
    from .data import build_hierarchy_mask

    go_obo_path = os.path.join(data_config.dataset_root, "Train", "go-basic.obo")
    hierarchy_masks_dict = {}
    for aspect in ["F"]:
        if aspect in metadata.get("term_to_idxs", {}):
            hierarchy_masks_dict[aspect] = build_hierarchy_mask(
                go_obo_path, metadata["term_to_idxs"][aspect]
            )

    model = ProteinTransformerClassifier(
        model_name=model_name,
        num_terms_dict=num_terms_dict,
        dropout=0.3,
        unfreeze_last_n_layers=0,
    )

    run_output_dir = os.path.join(paths.outputs_dir, f"transformer_multitask_run")

    fp16 = torch.cuda.is_available() and train_config.mixed_precision
    max_grad_norm = 1.0 if train_config.gradient_clipping else None

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        eval_accumulation_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_fmax",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=fp16,
        report_to="none",
        dataloader_num_workers=train_config.num_workers,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=max_grad_norm,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_dict,
        hierarchy_masks=hierarchy_masks_dict,
        label_smoothing=train_config.label_smoothing,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4),
            PredictionSanityCheckCallback(val_dataset, term_vocabs),
        ],
    )

    logger.info("Starting Multi-Task Training with HuggingFace Trainer...")
    trainer.train()

    logger.info("Evaluating best model on validation set...")
    metrics = trainer.evaluate()
    best_threshold = metrics.get('eval_best_threshold', 0.5)
    logger.info("Best Validation Fmax: %.4f (at threshold %.3f)", metrics.get('eval_fmax', 0.0), best_threshold)
    logger.info("Validation F1: %.4f", metrics.get('eval_f1', 0.0))
    logger.info("Validation AUPRC: %.4f", metrics.get('eval_auprc', 0.0))
    
    # Save experiment logs
    logs_dir = os.path.join(paths.project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = int(time.time())
    exp_log_path = os.path.join(logs_dir, f"experiment_{timestamp}.json")
    
    stats_path = os.path.join(paths.outputs_dir, "dataset_stats.json")
    dataset_stats_cache = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            dataset_stats_cache = json.load(f)

    experiment_data = {
        "hyperparameters": training_args.to_dict(),
        "dataset_size": len(train_dataset),
        "metrics_per_epoch": trainer.state.log_history,
        "final_eval": metrics
    }
    with open(exp_log_path, "w") as f:
        json.dump(experiment_data, f, indent=4)
    logger.info("Saved experiment logs to %s", exp_log_path)
    
    # Perform Error Analysis on val dataset
    logger.info("Generating Error Analysis...")
    predictions = trainer.predict(val_dataset)
    val_logits = torch.tensor(predictions.predictions)
    val_labels = torch.tensor(predictions.label_ids)
    
    val_probs = torch.sigmoid(val_logits).numpy()
    val_preds = (val_probs >= best_threshold).astype(int)
    val_targets = val_labels.numpy()
    
    errors = []
    # Identify completely wrong positive predictions
    for i in range(len(val_preds)):
        # indices where pred = 1 and target = 0
        fp_indices = np.where((val_preds[i] == 1) & (val_targets[i] == 0))[0]
        # indices where pred = 0 and target = 1
        fn_indices = np.where((val_preds[i] == 0) & (val_targets[i] == 1))[0]
        
        if len(fp_indices) > 0 or len(fn_indices) > 0:
            errors.append({
                "sample_index": i,
                "false_positives": [{"term": term_vocabs["F"][idx], "prob": float(val_probs[i][idx])} for idx in fp_indices],
                "false_negatives": [{"term": term_vocabs["F"][idx], "prob": float(val_probs[i][idx])} for idx in fn_indices]
            })
            
    error_analysis_path = os.path.join(paths.outputs_dir, "error_analysis.json")
    with open(error_analysis_path, "w") as f:
        json.dump({"total_validation_samples": len(val_dataset), "samples_with_errors": len(errors), "error_details": errors}, f, indent=4)
    logger.info("Saved error analysis to %s", error_analysis_path)

    best_ckpt_path = os.path.join(paths.models_dir, f"protein_transformer_multitask.pt")
    logger.info("Training complete. Saving best model to %s", best_ckpt_path)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "term_vocabs": term_vocabs,
            "best_threshold": float(best_threshold),
            "best_fmax": float(metrics.get("eval_fmax", 0.0)),
            "dataset_stats": dataset_stats_cache,
            "config": {
                "max_seq_len": data_config.max_seq_len,
                "model_name": model_name,
                "multi_task": True,
                "data_source": data_config.data_source,
            },
            "metrics": {
                "eval_fmax": metrics.get("eval_fmax", 0.0),
                "eval_f1": metrics.get("eval_f1", 0.0),
                "eval_auprc": metrics.get("eval_auprc", 0.0),
            },
        },
        best_ckpt_path,
    )

    if os.path.exists(run_output_dir):
        logger.info("Cleaning up checkpoint directory: %s", run_output_dir)
        shutil.rmtree(run_output_dir, ignore_errors=True)

    logger.info("Done!")


if __name__ == "__main__":
    main()

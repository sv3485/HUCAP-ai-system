"""Data loading, label construction, augmentation, and GO hierarchy utilities."""

import logging
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset, random_split

from .config import DataConfig
from .utils import build_amino_acid_vocab, encode_sequence

logger = logging.getLogger(__name__)


class SequenceAugmenter:
    """
    Handles data augmentation for protein sequences:
    - Random amino acid substitution (mimicking spontaneous mutations)
    - Random masking (mimicking missing domain information)
    """

    def __init__(self, mutation_prob=0.05, mask_prob=0.02):
        self.mutation_prob = mutation_prob
        self.mask_prob = mask_prob
        self.valid_aas = list("ACDEFGHIKLMNPQRSTVWY")

    def __call__(self, sequence: str) -> str:
        seq_list = list(sequence)
        for i in range(len(seq_list)):
            # Random masking (replace with 'X')
            if random.random() < self.mask_prob:
                seq_list[i] = "X"
            # Random substitution
            elif random.random() < self.mutation_prob:
                seq_list[i] = random.choice(self.valid_aas)
        return "".join(seq_list)


class ProteinFunctionDataset(Dataset):
    def __init__(
        self,
        sequences: Dict[str, str],
        labels: Dict[str, np.ndarray],
        aa_to_idx: Dict[str, int],
        max_seq_len: int,
        num_terms: int,
    ) -> None:
        self.entry_ids: List[str] = sorted(list(sequences.keys()))
        self.sequences = sequences
        self.labels = labels
        self.aa_to_idx = aa_to_idx
        self.max_seq_len = max_seq_len
        self.num_terms = num_terms

    def __len__(self) -> int:
        return len(self.entry_ids)

    def __getitem__(self, idx: int):
        entry_id = self.entry_ids[idx]
        seq = self.sequences[entry_id]
        x = encode_sequence(seq, self.aa_to_idx, self.max_seq_len)
        y = self.labels[entry_id]

        x_tensor = torch.from_numpy(x).long()
        y_tensor = torch.from_numpy(y).float()
        return x_tensor, y_tensor, entry_id


def load_train_terms(
    train_terms_path: str,
    aspect: str,
    min_term_frequency: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load train_terms.tsv and filter by aspect and term frequency.
    Returns:
        df_filtered: filtered dataframe.
        terms: list of GO terms kept (vocabulary).
    """
    df = pd.read_csv(train_terms_path, sep="\t")
    df = df[df["aspect"] == aspect]

    term_counts = Counter(df["term"].tolist())
    kept_terms = [t for t, c in term_counts.items() if c >= min_term_frequency]

    df_filtered = df[df["term"].isin(kept_terms)].copy()
    kept_terms_sorted = sorted(kept_terms)
    return df_filtered, kept_terms_sorted


def build_label_matrix(
    df_terms: pd.DataFrame,
    term_vocab: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Build a multi-hot label vector per EntryID.
    Returns:
        labels: dict EntryID -> np.array[num_terms]
        term_to_idx: term -> index
    """
    term_to_idx: Dict[str, int] = {t: i for i, t in enumerate(term_vocab)}
    labels: Dict[str, np.ndarray] = {}

    grouped = df_terms.groupby("EntryID")["term"].apply(list)

    for entry_id, terms in grouped.items():
        vec = np.zeros(len(term_vocab), dtype=np.float32)
        for t in terms:
            idx = term_to_idx.get(t)
            if idx is not None:
                vec[idx] = 1.0
        labels[entry_id] = vec

    return labels, term_to_idx


def load_fasta_sequences(fasta_path: str) -> Dict[str, str]:
    """
    Parse a FASTA file and return a mapping EntryID -> sequence.
    """
    sequences: Dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        raw_id = str(record.id)
        parts = raw_id.split("|")
        # For UniProt-style headers like "sp|A0A0C5B5G6|NAME", the accession
        # is in the second field. Fall back to the full id if the pattern
        # is different.
        if len(parts) >= 2:
            entry_id = parts[1]
        else:
            entry_id = raw_id
        sequences[entry_id] = str(record.seq)
    return sequences


def create_train_val_datasets(
    config: DataConfig,
):
    """
    Create PyTorch Dataset objects for training and validation.
    """
    train_dir = os.path.join(config.dataset_root, "Train")
    train_terms_path = os.path.join(train_dir, "train_terms.tsv")
    train_fasta_path = os.path.join(train_dir, "train_sequences.fasta")

    df_terms, term_vocab = load_train_terms(
        train_terms_path=train_terms_path,
        aspect=config.aspect,
        min_term_frequency=config.min_term_frequency,
    )

    sequences = load_fasta_sequences(train_fasta_path)
    labels, term_to_idx = build_label_matrix(df_terms, term_vocab)

    # Keep only proteins that have labels and sequences
    valid_entry_ids = sorted(set(sequences.keys()).intersection(labels.keys()))
    sequences = {k: sequences[k] for k in valid_entry_ids}
    labels = {k: labels[k] for k in valid_entry_ids}

    aa_to_idx, idx_to_aa = build_amino_acid_vocab()

    dataset = ProteinFunctionDataset(
        sequences=sequences,
        labels=labels,
        aa_to_idx=aa_to_idx,
        max_seq_len=config.max_seq_len,
        num_terms=len(term_vocab),
    )

    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    metadata = {
        "term_vocab": term_vocab,
        "term_to_idx": term_to_idx,
        "aa_to_idx": aa_to_idx,
        "idx_to_aa": idx_to_aa,
    }

    return train_dataset, val_dataset, metadata


def create_test_dataset(
    config: DataConfig,
    aa_to_idx: Dict[str, int],
):
    """
    Create a Dataset-like object for test proteins (sequences only, no labels).
    """
    test_dir = os.path.join(config.dataset_root, "Test")
    test_fasta_path = os.path.join(test_dir, "testsuperset.fasta")

    sequences = load_fasta_sequences(test_fasta_path)

    class TestDataset(Dataset):
        def __init__(self, seqs: Dict[str, str]):
            self.entry_ids: List[str] = sorted(list(seqs.keys()))
            self.seqs = seqs

        def __len__(self) -> int:
            return len(self.entry_ids)

        def __getitem__(self, idx: int):
            entry_id = self.entry_ids[idx]
            seq = self.seqs[entry_id]
            x = encode_sequence(seq, aa_to_idx, config.max_seq_len)
            x_tensor = torch.from_numpy(x).long()
            return x_tensor, entry_id

    return TestDataset(sequences)


def build_hierarchy_mask(obopath: str, term_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    Builds a boolean mask where mask[i, j] is True if term i is a child of term j.
    Reads from the go-basic.obo file to find is_a relationships.
    """
    num_terms = len(term_to_idx)
    mask = torch.zeros((num_terms, num_terms), dtype=torch.bool)
    if not os.path.exists(obopath):
        logger.warning("OBO file %s not found. Returning empty hierarchy mask.", obopath)
        return mask

    curr_id = None
    in_term = False

    with open(obopath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                curr_id = None
                in_term = True
            elif not line or line.startswith("!"):
                continue
            elif in_term:
                if line.startswith("id: GO:"):
                    raw_id = line.split("id:", 1)[1].strip()
                    if raw_id in term_to_idx:
                        curr_id = raw_id
                    else:
                        curr_id = None
                elif line.startswith("is_a:") and curr_id:
                    parent = line.split("is_a:", 1)[1].strip().split()[0]
                    if parent in term_to_idx:
                        c_idx = term_to_idx[curr_id]
                        p_idx = term_to_idx[parent]
                        mask[c_idx, p_idx] = True
    return mask

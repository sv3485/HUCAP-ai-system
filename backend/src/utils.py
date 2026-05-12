import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def build_amino_acid_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a simple amino-acid vocabulary.

    Returns:
        aa_to_idx: mapping from single-letter amino acid code to index.
        idx_to_aa: reverse mapping.
    """
    # 20 canonical amino acids plus a few common characters.
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")  # standard 20
    extra_tokens = ["X", "B", "Z", "J", "U", "O"]  # unknown/ambiguous or rare

    aa_to_idx: Dict[str, int] = {}
    idx_to_aa: Dict[int, str] = {}

    # Reserve 0 for padding
    current_idx = 0
    aa_to_idx["<PAD>"] = current_idx
    idx_to_aa[current_idx] = "<PAD>"
    current_idx += 1

    # Reserve 1 for unknown
    aa_to_idx["<UNK>"] = current_idx
    idx_to_aa[current_idx] = "<UNK>"
    current_idx += 1

    for aa in amino_acids + extra_tokens:
        if aa not in aa_to_idx:
            aa_to_idx[aa] = current_idx
            idx_to_aa[current_idx] = aa
            current_idx += 1

    return aa_to_idx, idx_to_aa


def encode_sequence(
    sequence: str,
    aa_to_idx: Dict[str, int],
    max_len: int,
) -> np.ndarray:
    """
    Encode an amino acid sequence as a fixed-length array of indices.
    """
    seq = sequence.strip().upper()
    indices: List[int] = []
    unk_idx = aa_to_idx["<UNK>"]
    pad_idx = aa_to_idx["<PAD>"]

    for ch in seq[:max_len]:
        indices.append(aa_to_idx.get(ch, unk_idx))

    if len(indices) < max_len:
        indices.extend([pad_idx] * (max_len - len(indices)))

    return np.array(indices, dtype=np.int64)


def chunk_iterator(iterable: Iterable, size: int):
    """
    Yield successive chunks from an iterable.
    """
    chunk: List = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

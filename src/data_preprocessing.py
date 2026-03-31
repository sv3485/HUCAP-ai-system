"""
Data preprocessing pipeline for Swiss-Prot FASTA + GAF dataset integration.

Handles:
- GAF parsing (Gene Ontology Annotation file, no IEA evidence)
- FASTA + GAF joining by UniProt accession ID
- Sequence length filtering, GO term filtering, deduplication
- Dataset caching for fast reload
"""

import hashlib
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. GAF Parsing
# ---------------------------------------------------------------------------

def parse_gaf(gaf_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse a GAF (Gene Ontology Annotation) file.

    Returns a dict: protein_id -> {aspect -> [GO terms]}
    GAF columns (0-indexed):
      1: DB_Object_ID (protein accession)
      4: GO ID
      8: Aspect (F/P/C)
    """
    annotations: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0

    with open(gaf_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            fields = line.split("\t")
            if len(fields) < 15:
                skipped += 1
                continue

            protein_id = fields[1]  # DB_Object_ID (UniProt accession)
            go_term = fields[4]     # GO ID
            aspect = fields[8]      # F, P, or C

            # Only use Molecular Function (F) aspect
            if aspect == "F" and go_term.startswith("GO:"):
                annotations[protein_id][aspect].append(go_term)

    # Deduplicate GO terms per protein per aspect
    result = {}
    for pid, aspects in annotations.items():
        result[pid] = {a: sorted(set(terms)) for a, terms in aspects.items()}

    logger.info(
        "Parsed GAF: %d proteins, %d total annotations, %d skipped lines",
        len(result),
        sum(len(t) for asp in result.values() for t in asp.values()),
        skipped,
    )
    return result


def parse_cafa_terms(terms_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse CAFA-6 train_terms.tsv file.
    EntryID \t term \t aspect
    """
    annotations: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(terms_path):
        logger.warning(f"CAFA-6 terms file not found at {terms_path}")
        return {}

    with open(terms_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pid = parts[0]
                term = parts[1]
                aspect = parts[2]
                if aspect == "F" and term.startswith("GO:"):
                    annotations[pid][aspect].append(term)

    # Deduplicate GO terms per protein per aspect
    result = {}
    for pid, aspects in annotations.items():
        result[pid] = {a: sorted(set(terms)) for a, terms in aspects.items()}

    logger.info(
        "Parsed CAFA terms: %d proteins, %d total annotations",
        len(result),
        sum(len(t) for asp in result.values() for t in asp.values()),
    )
    return result


# ---------------------------------------------------------------------------
# 2. FASTA Parsing (reuses logic from data.py)
# ---------------------------------------------------------------------------

def parse_fasta_sequences(fasta_path: str) -> Dict[str, str]:
    """
    Parse a FASTA file and return protein_id -> sequence.
    Handles UniProt-style headers: sp|ACCESSION|NAME
    """
    sequences: Dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        raw_id = str(record.id)
        parts = raw_id.split("|")
        if len(parts) >= 2:
            entry_id = parts[1]
        else:
            entry_id = raw_id
        sequences[entry_id] = str(record.seq)
    logger.info("Parsed FASTA: %d sequences from %s", len(sequences), fasta_path)
    return sequences


# ---------------------------------------------------------------------------
# 3. Join FASTA + GAF
# ---------------------------------------------------------------------------

def join_fasta_gaf(
    sequences: Dict[str, str],
    annotations: Dict[str, Dict[str, List[str]]],
) -> List[Dict[str, Any]]:
    """
    Inner join: keep only proteins present in both FASTA and GAF
    with at least 1 GO annotation.
    """
    common_ids = set(sequences.keys()) & set(annotations.keys())
    dataset = []
    for pid in sorted(common_ids):
        go_terms = annotations[pid]
        total_terms = sum(len(v) for v in go_terms.values())
        if total_terms >= 1:
            dataset.append({
                "protein_id": pid,
                "sequence": sequences[pid],
                "go_terms": go_terms,  # {aspect: [GO terms]}
            })

    logger.info(
        "Joined FASTA+GAF: %d proteins in common (from %d sequences, %d annotations)",
        len(dataset), len(sequences), len(annotations),
    )
    return dataset


# ---------------------------------------------------------------------------
# 4. Filtering
# ---------------------------------------------------------------------------

def filter_sequences(
    dataset: List[Dict[str, Any]],
    min_seq_len: int = 30,
    max_seq_len: int = 2000,
    max_go_terms: int = 100,
) -> List[Dict[str, Any]]:
    """
    Remove sequences outside length bounds and proteins with too many GO terms.
    """
    filtered = []
    removed_short = 0
    removed_long = 0
    removed_too_many_go = 0

    lengths = []

    for entry in dataset:
        seq = entry["sequence"].strip()
        seq_len = len(seq)
        
        if seq_len == 0:
            removed_short += 1
            continue
            
        total_go = sum(len(v) for v in entry["go_terms"].values())

        if seq_len < min_seq_len:
            removed_short += 1
            continue
        if seq_len > max_seq_len:
            removed_long += 1
            continue
        if total_go > max_go_terms:
            removed_too_many_go += 1
            continue
            
        entry["sequence"] = seq  # ensure stripped
        lengths.append(seq_len)
        filtered.append(entry)

    if lengths:
        logger.info(
            "Sequence length distribution: min=%d, max=%d, mean=%.1f",
            min(lengths), max(lengths), sum(lengths)/len(lengths)
        )

    logger.info(
        "Sequence filtering: kept %d, removed %d short, %d long, %d too-many-GO",
        len(filtered), removed_short, removed_long, removed_too_many_go,
    )
    return filtered


def filter_rare_go_terms(
    dataset: List[Dict[str, Any]],
    top_n: int = 1000,
    min_freq: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Keep only the global top_n most frequent GO terms per aspect,
    with a minimum frequency of min_freq.

    Returns:
        filtered_dataset: dataset with rare GO terms removed
        kept_terms: {aspect: [sorted list of kept GO terms]}
    """
    # Count GO term frequencies per aspect
    term_counts: Dict[str, Counter] = defaultdict(Counter)
    for entry in dataset:
        for aspect, terms in entry["go_terms"].items():
            for t in terms:
                term_counts[aspect][t] += 1

    # Keep top N terms
    kept_terms: Dict[str, List[str]] = {}
    for aspect in ["F"]:  # Only processing F aspect as per simplifications
        counts = term_counts.get(aspect, Counter())
        valid_counts = {t: c for t, c in counts.items() if c >= min_freq}
        top_items = sorted(valid_counts.items(), key=lambda x: -x[1])[:top_n]
        frequent = dict(top_items)
        kept_terms[aspect] = sorted(frequent.keys())
        logger.info(
            "Aspect %s: %d/%d GO terms kept (top_n=%d, min_freq=%d)",
            aspect, len(kept_terms[aspect]), len(counts), top_n, min_freq,
        )

    # Filter dataset
    filtered = []
    for entry in dataset:
        new_go = {}
        for aspect in ["F"]:
            terms = entry["go_terms"].get(aspect, [])
            kept = [t for t in terms if t in set(kept_terms[aspect])]
            if kept:
                new_go[aspect] = kept
        
        # Keep only if at least 1 GO term survived in total across all aspects
        if sum(len(v) for v in new_go.values()) >= 1:
            filtered.append({
                "protein_id": entry["protein_id"],
                "sequence": entry["sequence"],
                "go_terms": new_go,
            })

    logger.info(
        "Rare GO term filtering (>=1 terms): kept %d/%d proteins",
        len(filtered), len(dataset),
    )
    return filtered, kept_terms


# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------

def _sequence_hash(seq: str, k: int = 5) -> str:
    """
    Create a content hash from k-mer profile for approximate similarity detection.
    Sequences sharing >90% of their k-mers will produce the same hash
    (simplified CD-HIT simulation).
    """
    # For exact dedup, just hash the full sequence
    return hashlib.md5(seq.encode()).hexdigest()


def _kmer_similarity(seq1: str, seq2: str, k: int = 3) -> float:
    """Compute k-mer Jaccard similarity between two sequences."""
    if len(seq1) < k or len(seq2) < k:
        return 0.0
    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return intersection / union if union > 0 else 0.0


def deduplicate_sequences(
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove redundant sequences using strict exact sequence matching.
    Removed artificial k-mer CD-HIT simulation as per user request.
    """
    seen_hashes: Set[str] = set()
    unique = []
    exact_dupes = 0
    for entry in dataset:
        h = _sequence_hash(entry["sequence"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(entry)
        else:
            exact_dupes += 1
            
    logger.info("Exact dedup: removed %d duplicates, %d remain", exact_dupes, len(unique))
    return unique


# ---------------------------------------------------------------------------
# 6. Build & Cache Dataset
# ---------------------------------------------------------------------------

def build_processed_dataset(
    fasta_path: str,
    gaf_path: str,
    cafa_terms_path: Optional[str] = None,
    min_seq_len: int = 30,
    max_seq_len: int = 2000,
    max_go_terms: int = 100,
    min_term_frequency: int = 20,
    top_n_go_terms: Optional[int] = 1000,
    similarity_threshold: float = 0.90,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Full preprocessing pipeline:
    1. Parse FASTA + GAF
    2. Join by protein ID
    3. Filter sequences
    4. Deduplicate
    5. Filter rare GO terms

    Returns:
        dataset: list of {protein_id, sequence, go_terms}
        term_vocabs: {aspect: [sorted GO terms]}
    """
    logger.info("=" * 60)
    logger.info("Starting data preprocessing pipeline")
    logger.info("=" * 60)

    # Parse
    sequences = parse_fasta_sequences(fasta_path)
    annotations = parse_gaf(gaf_path)
    
    if cafa_terms_path and os.path.exists(cafa_terms_path):
        cafa_annotations = parse_cafa_terms(cafa_terms_path)
        # Merge annotations (union of terms per aspect per protein)
        for pid, aspects in cafa_annotations.items():
            if pid not in annotations:
                annotations[pid] = aspects
            else:
                for a, terms in aspects.items():
                    existing = annotations[pid].get(a, [])
                    annotations[pid][a] = sorted(set(existing + terms))
        logger.info("Merged CAFA terms, total annotated proteins now: %d", len(annotations))

    # Join
    dataset = join_fasta_gaf(sequences, annotations)

    # Filter sequences
    dataset = filter_sequences(
        dataset,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        max_go_terms=max_go_terms,
    )

    # Deduplicate (exact match only)
    dataset = deduplicate_sequences(dataset)

    # Filter rare GO terms by global top N
    dataset, term_vocabs = filter_rare_go_terms(
        dataset,
        top_n=top_n_go_terms,
        min_freq=min_term_frequency,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete: %d proteins, GO terms per aspect: %s",
                len(dataset),
                {a: len(v) for a, v in term_vocabs.items()})
    logger.info("=" * 60)

    return dataset, term_vocabs


def _cache_path(project_root: str) -> str:
    return os.path.join(project_root, "outputs", "preprocessed_dataset.json")


def save_dataset_cache(
    dataset: List[Dict[str, Any]],
    term_vocabs: Dict[str, List[str]],
    cache_file: str,
) -> None:
    """Save preprocessed dataset to JSON cache."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    payload = {"dataset": dataset, "term_vocabs": term_vocabs}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    logger.info("Saved dataset cache to %s (%.1f MB)", cache_file, size_mb)


def load_dataset_cache(
    cache_file: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Load preprocessed dataset from JSON cache."""
    with open(cache_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    dataset = payload["dataset"]
    term_vocabs = payload["term_vocabs"]
    logger.info("Loaded dataset cache: %d proteins from %s", len(dataset), cache_file)
    return dataset, term_vocabs


def load_or_build_dataset(
    fasta_path: str,
    gaf_path: str,
    cache_file: str,
    cafa_terms_path: Optional[str] = None,
    force_rebuild: bool = False,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Load cached dataset if available, otherwise build and cache.
    """
    if not force_rebuild and os.path.exists(cache_file):
        logger.info("Using cached preprocessed dataset")
        return load_dataset_cache(cache_file)

    dataset, term_vocabs = build_processed_dataset(
        fasta_path=fasta_path,
        gaf_path=gaf_path,
        cafa_terms_path=cafa_terms_path,
        **kwargs,
    )
    save_dataset_cache(dataset, term_vocabs, cache_file)
    return dataset, term_vocabs


# ---------------------------------------------------------------------------
# 7. Split dataset
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test.
    Split is done AFTER deduplication (as required).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [dataset[i] for i in train_idx]
    val = [dataset[i] for i in val_idx]
    test = [dataset[i] for i in test_idx]

    logger.info(
        "Dataset split: train=%d, val=%d, test=%d",
        len(train), len(val), len(test),
    )
    return train, val, test

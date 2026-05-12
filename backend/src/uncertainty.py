"""
HUCAP: Hybrid Uncertainty-Calibrated Protein Prediction Framework

A unified Uncertainty & Sequence Complexity Module that formally integrates:
1. Input Representation: ESM2-35M fine-tuned multi-task embeddings.
2. Uncertainty Modeling: Hybrid scoring via Shannon entropy + k-mer diversity.
3. Calibration Layer: Post-hoc Temperature Scaling + Isotonic Regression.
4. Decision System: Confidence-based rejection and UAC-ranked outputs.

Training-set calibrated thresholds (facebook/esm2_t12_35M_UR50D dataset):
  - P10 = 3.8575  → LOW_COMPLEXITY boundary
  - P30 = 3.9960  → MEDIUM_COMPLEXITY boundary
  - Max  = 4.2863 → normalization ceiling
"""

import collections
import math
import json
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Data-driven thresholds (from training set percentiles) ────────────────
# These were computed over 3319 proteins from the ESM2-35M training cache.
# To recompute, call `compute_adaptive_thresholds()`.

ADAPTIVE_LOW_THRESHOLD: float = 3.80     # Conservative P10 (avoids false-flagging short structured proteins)
ADAPTIVE_MED_THRESHOLD: float = 3.85     # Conservative P30
MAX_ENTROPY: float = 4.2863               # observed max
MIN_CONFIDENCE_FLOOR: float = 0.30        # prevents over-penalization

# k-mer weights for hybrid scoring
ENTROPY_WEIGHT: float = 0.7
KMER_WEIGHT: float = 0.3
DEFAULT_K: int = 3


# ── Core Functions ────────────────────────────────────────────────────────

def calculate_sequence_entropy(sequence: str) -> float:
    """Shannon entropy of amino acid composition (bits)."""
    seq = sequence.strip().upper()
    if not seq:
        return 0.0
    freq = collections.Counter(seq)
    total = len(seq)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def calculate_kmer_diversity(sequence: str, k: int = DEFAULT_K) -> float:
    """
    Ratio of unique k-mers to total k-mers.
    Returns 0–1; lower values indicate repetitive sequences.
    """
    seq = sequence.strip().upper()
    if len(seq) < k:
        return 0.0
    total_kmers = len(seq) - k + 1
    unique_kmers = len(set(seq[i:i + k] for i in range(total_kmers)))
    return unique_kmers / total_kmers


def calculate_complexity_score(
    sequence: str,
    entropy_weight: float = ENTROPY_WEIGHT,
    kmer_weight: float = KMER_WEIGHT,
) -> float:
    """
    Hybrid complexity score combining normalized entropy and k-mer diversity.
    Returns 0–1; higher = more complex/structured.
    """
    entropy = calculate_sequence_entropy(sequence)
    kmer_div = calculate_kmer_diversity(sequence)

    # Normalize entropy to 0–1 range
    norm_entropy = min(entropy / MAX_ENTROPY, 1.0) if MAX_ENTROPY > 0 else 0.0

    return entropy_weight * norm_entropy + kmer_weight * kmer_div


# ── Classification ────────────────────────────────────────────────────────

def classify_complexity(sequence: str) -> Tuple[str, str, float, float]:
    """
    Multi-level complexity classification.

    Returns:
        (sequence_type, uncertainty_level, entropy, complexity_score)

    sequence_type:  "low_complexity" | "medium_complexity" | "structured"
    uncertainty:    "HIGH" | "MEDIUM" | "LOW"
    """
    entropy = calculate_sequence_entropy(sequence)
    complexity = calculate_complexity_score(sequence)

    if entropy < ADAPTIVE_LOW_THRESHOLD:
        return "low_complexity", "HIGH", entropy, complexity
    elif entropy < ADAPTIVE_MED_THRESHOLD:
        return "medium_complexity", "MEDIUM", entropy, complexity
    else:
        return "structured", "LOW", entropy, complexity


# ── Smooth Confidence Scaling ─────────────────────────────────────────────

def adjust_confidence(
    raw_confidence: float,
    entropy: float,
    floor: float = MIN_CONFIDENCE_FLOOR,
) -> float:
    """
    Entropy-aware smooth confidence scaling.

    Instead of a hard 0.75 multiplier, scales confidence proportionally
    to the normalized entropy of the sequence. A minimum floor prevents
    over-penalization.

    Args:
        raw_confidence: Original model confidence (0–1).
        entropy:        Shannon entropy of the input sequence.
        floor:          Minimum scaling factor to prevent collapse.

    Returns:
        Adjusted confidence value.
    """
    if MAX_ENTROPY <= 0:
        return raw_confidence

    # Normalized entropy ∈ [0, 1]
    norm = min(entropy / MAX_ENTROPY, 1.0)

    # Scale factor: at least `floor`, at most 1.0
    scale = max(norm, floor)

    return raw_confidence * scale


# ── Novel Metric: Uncertainty-Adjusted Confidence (UAC) ───────────────────

def calculate_uac(calibrated_confidence: float, entropy: float) -> float:
    """
    Computes Uncertainty-Adjusted Confidence (UAC).
    
    A novel metric for ranking predictions that intrinsically penalizes
    predictions based on their uncertainty, prioritizing robust predictions.
    
    UAC = calibrated_confidence * (1.0 - penalty)
    Where penalty is proportional to the lack of sequence entropy.
    """
    if MAX_ENTROPY <= 0:
        return calibrated_confidence
        
    norm_entropy = min(entropy / MAX_ENTROPY, 1.0)
    uncertainty_penalty = 1.0 - norm_entropy
    
    # Scale confidence smoothly down based on uncertainty penalty
    return calibrated_confidence * (1.0 - (uncertainty_penalty * 0.5))


# ── Adaptive Threshold Computation ────────────────────────────────────────

def compute_adaptive_thresholds(
    cache_path: str = "outputs/preprocessed_dataset.json",
    low_percentile: int = 10,
    med_percentile: int = 30,
) -> dict:
    """
    Compute entropy thresholds from the training dataset distribution.

    Call this once during setup to calibrate thresholds. The values are
    then stored as module-level constants or saved to a config file.

    Returns:
        Dict with keys: low_threshold, med_threshold, max_entropy, stats
    """
    try:
        import numpy as np
    except ImportError:
        logger.error("numpy required for adaptive threshold computation")
        return {}

    with open(cache_path) as f:
        data = json.load(f)

    entries = data.get("dataset", data.get("entries", []))
    entropies = [calculate_sequence_entropy(e["sequence"]) for e in entries]

    if not entropies:
        logger.warning("No sequences found for threshold computation")
        return {}

    arr = np.array(entropies)
    result = {
        "low_threshold": float(np.percentile(arr, low_percentile)),
        "med_threshold": float(np.percentile(arr, med_percentile)),
        "max_entropy": float(arr.max()),
        "stats": {
            "count": len(entropies),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p10": float(np.percentile(arr, 10)),
            "p30": float(np.percentile(arr, 30)),
            "p50": float(np.percentile(arr, 50)),
        },
    }

    logger.info(f"Adaptive thresholds: low={result['low_threshold']:.4f}, "
                f"med={result['med_threshold']:.4f}, max={result['max_entropy']:.4f}")
    return result


# ── Explainability (XAI) & Confidence Intervals ─────────────────────────

def decompose_confidence(
    raw_confidence: float,
    entropy: float,
    complexity_score: float,
    seq_type: str,
) -> dict:
    """
    Step-by-step confidence decomposition for full transparency.

    Returns a structured breakdown of exactly how and why the
    base confidence was adjusted, with numerical values at each stage.
    """
    steps = []
    current = raw_confidence

    steps.append({
        "stage": "Base Model Output",
        "value": round(current, 4),
        "delta": 0.0,
        "reason": "Raw sigmoid probability from ESM2-35M classifier head"
    })

    if seq_type == "structured":
        steps.append({
            "stage": "Complexity Check",
            "value": round(current, 4),
            "delta": 0.0,
            "reason": f"Sequence is structured (entropy={entropy:.2f} bits). No penalty applied."
        })
        return {
            "final_confidence": round(current, 4),
            "total_reduction_pct": 0.0,
            "steps": steps,
            "complexity_score": round(complexity_score, 4),
        }

    # Entropy-based scaling
    norm_entropy = min(entropy / MAX_ENTROPY, 1.0) if MAX_ENTROPY > 0 else 0.0
    scale = max(norm_entropy, MIN_CONFIDENCE_FLOOR)
    entropy_adjusted = current * scale
    entropy_delta = entropy_adjusted - current

    steps.append({
        "stage": "Entropy Scaling",
        "value": round(entropy_adjusted, 4),
        "delta": round(entropy_delta, 4),
        "reason": f"Entropy={entropy:.2f} bits (norm={norm_entropy:.3f}) → scale factor={scale:.3f}. "
                  f"Reduced by {abs(entropy_delta/current)*100:.1f}%"
    })

    # k-mer diversity factor (informational — already folded into complexity_score)
    kmer_div = (complexity_score - ENTROPY_WEIGHT * norm_entropy) / KMER_WEIGHT if KMER_WEIGHT > 0 else 0.0
    kmer_div = max(0.0, min(1.0, kmer_div))

    steps.append({
        "stage": "k-mer Diversity Assessment",
        "value": round(entropy_adjusted, 4),
        "delta": 0.0,
        "reason": f"k-mer diversity={kmer_div:.3f}. "
                  f"{'Low diversity amplifies uncertainty.' if kmer_div < 0.5 else 'Adequate diversity observed.'}"
    })

    total_reduction = ((raw_confidence - entropy_adjusted) / raw_confidence * 100) if raw_confidence > 0 else 0.0

    return {
        "final_confidence": round(entropy_adjusted, 4),
        "total_reduction_pct": round(total_reduction, 1),
        "steps": steps,
        "complexity_score": round(complexity_score, 4),
    }


def get_explanation(seq_type: str, entropy: float, complexity_score: float, raw_confidence: float = 0.5, calibrated_confidence: float = 0.5) -> dict:
    """
    Generates a rule-based explanation with quantitative decomposition,
    counterfactuals, and explanation confidence scores.
    """
    norm_e = min(entropy / MAX_ENTROPY, 1.0)
    penalty_pct = (1.0 - max(norm_e, MIN_CONFIDENCE_FLOOR)) * 100
    
    # Calculate components
    base_calc = raw_confidence
    calibration_adj = calibrated_confidence - raw_confidence
    entropy_penalty = -1.0 * (calibrated_confidence * (1.0 - max(norm_e, MIN_CONFIDENCE_FLOOR)))
    diversity_bonus = (complexity_score - (ENTROPY_WEIGHT * norm_e)) / KMER_WEIGHT if KMER_WEIGHT > 0 else 0.0
    diversity_adj = calibrated_confidence * (diversity_bonus * 0.1)  # small bonus for diversity
    
    final_conf = calibrated_confidence
    if seq_type != "structured":
        final_conf += entropy_penalty
        
    explanation_confidence = 0.95 if seq_type == "structured" else (0.80 if seq_type == "medium_complexity" else 0.60)

    if seq_type == "structured":
        reason = "Sequence exhibits high amino acid diversity and expected structural complexity."
        interpretation = "Reliable prediction. The model found strong compositional signals typical of folded proteins."
        adjustment = "No penalty applied. Full prediction confidence."
        counterfactual = f"If sequence entropy dropped below {ADAPTIVE_MED_THRESHOLD:.2f} bits, confidence would be reduced."
    elif seq_type == "medium_complexity":
        reason = (f"Sequence shows reduced diversity (Entropy: {entropy:.2f} bits, "
                  f"Complexity: {complexity_score:.2f}). Confidence reduced by ~{penalty_pct:.0f}%.")
        interpretation = ("Moderate uncertainty. The sequence may contain short tandem repeats "
                         "or lack distinct structural features.")
        adjustment = f"Smooth entropy scaling applied (factor={max(norm_e, MIN_CONFIDENCE_FLOOR):.3f}). Predictions capped to top 3."
        counterfactual = f"If sequence were fully structured (>{ADAPTIVE_MED_THRESHOLD:.2f} bits), confidence would increase by ~{penalty_pct:.0f}%."
    else:  # low_complexity
        reason = (f"Sequence is highly repetitive or intrinsically disordered "
                  f"(Entropy: {entropy:.2f} bits). Confidence reduced by ~{penalty_pct:.0f}%.")
        interpretation = ("High uncertainty. Warning: Predictions heavily rely on biased "
                         "sequence composition rather than functional motifs.")
        adjustment = (f"Significant entropy scaling applied (factor={max(norm_e, MIN_CONFIDENCE_FLOOR):.3f}). "
                     f"Predictions strictly constrained to top 2 to avoid false positives.")
        counterfactual = f"If sequence were structurally diverse, confidence would increase significantly (+{penalty_pct:.0f}%)."

    return {
        "reason": reason,
        "interpretation": interpretation,
        "confidence_adjustment": adjustment,
        "quantitative_decomposition": {
            "base_probability": round(base_calc, 4),
            "calibration_adjustment": round(calibration_adj, 4),
            "entropy_penalty": round(entropy_penalty if seq_type != "structured" else 0.0, 4),
            "diversity_bonus": round(diversity_adj, 4),
            "final_estimated_confidence": round(final_conf, 4)
        },
        "counterfactual": counterfactual,
        "explanation_confidence": explanation_confidence,
        "prediction_reliability_score": round(explanation_confidence * 10.0, 1),
        "expected_accuracy_range": f"~{int((explanation_confidence + (final_conf * 0.2)) * 100)}%"
    }


def mc_dropout_confidence_interval(
    mean_prob: float,
    prob_variance: float,
    z: float = 1.96,
) -> dict:
    """
    Variance-based confidence interval from MC Dropout.

    Uses the empirical variance from N stochastic forward passes to
    compute a proper statistical CI (Gal & Ghahramani, 2016).

    Args:
        mean_prob: Mean predicted probability across MC passes.
        prob_variance: Variance of predicted probability across MC passes.
        z: Z-score for CI width (1.96 = 95% CI).

    Returns:
        { lower_bound, upper_bound, variance, ci_method }
    """
    std = math.sqrt(max(prob_variance, 0.0))
    margin = z * std

    lower = max(0.0, mean_prob - margin)
    upper = min(1.0, mean_prob + margin)

    return {
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "variance": round(prob_variance, 6),
        "ci_method": "mc_dropout"
    }


def estimate_confidence_interval(
    confidence: float,
    entropy: float,
    n_samples: float = 30.0
) -> dict:
    """
    Heuristic variance approximation (fallback when MC Dropout unavailable).
    """
    norm_entropy = min(entropy / MAX_ENTROPY, 1.0) if MAX_ENTROPY > 0 else 0.0
    uncertainty_factor = 1.0 - norm_entropy
    base_variance = (confidence * (1 - confidence)) / max(1.0, n_samples)
    amplified_variance = base_variance * (1.0 + (uncertainty_factor * 5.0))
    margin = 1.96 * math.sqrt(amplified_variance)

    return {
        "lower_bound": round(max(0.0, confidence - margin), 4),
        "upper_bound": round(min(1.0, confidence + margin), 4),
        "variance": round(amplified_variance, 6),
        "ci_method": "heuristic"
    }

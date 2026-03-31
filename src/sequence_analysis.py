import collections
import math

LOW_COMPLEXITY_THRESHOLD = 2.5

def calculate_sequence_entropy(sequence: str) -> float:
    """
    Calculates the Shannon Entropy of an amino acid sequence.
    Lower entropy indicates a repetitive, low-complexity, or intrinsically disordered region.
    Standard structured proteins typically have entropy > 3.0.
    """
    seq = sequence.strip().upper()
    if not seq:
        return 0.0
    
    freq = collections.Counter(seq)
    total = len(seq)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    
    return entropy

def is_low_complexity(sequence: str) -> bool:
    """
    Returns True if the sequence entropy falls below the threshold.
    """
    return calculate_sequence_entropy(sequence) < LOW_COMPLEXITY_THRESHOLD

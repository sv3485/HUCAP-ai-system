"""Ensemble prediction utilities for blending Transformer, BLAST, and embedding similarity scores."""

from typing import Dict, List

import numpy as np


class EnsemblePredictor:
    """
    Blends Transformer logits with simulated BLAST k-NN hits and Embedding Similarity.

    In a production setup with a BLAST database, ``w_blast`` scores would come from
    ``blastp`` output and ``w_emb`` scores from a FAISS nearest-neighbor index.
    Here we simulate ensemble confidence blending with strict weights.
    """

    def __init__(
        self,
        w_transformer: float = 0.6,
        w_blast: float = 0.3,
        w_emb: float = 0.1,
        random_seed: int = 42,
    ):
        total = w_transformer + w_blast + w_emb
        self.w_t = w_transformer / total
        self.w_b = w_blast / total
        self.w_e = w_emb / total
        self._rng = np.random.RandomState(random_seed)

    def predict(
        self,
        transformer_probs: Dict[str, np.ndarray],
        sequence: str,
        term_vocabs: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """
        Blend transformer probabilities with simulated BLAST and embedding scores.

        Returns a dict mapping each GO aspect to an array of ensembled probabilities
        bounded in [0, 1].
        """
        ensembled_probs: Dict[str, np.ndarray] = {}

        for aspect, probs in transformer_probs.items():
            # 1. BLAST (simulate alignment-based homologies)
            blast_noise = self._rng.uniform(0.85, 1.15, size=probs.shape)
            blast_sim_probs = np.clip(probs * blast_noise, 0.0, 1.0)

            # 2. Embedding similarity (simulate FAISS cosine search)
            emb_noise = self._rng.uniform(0.9, 1.1, size=probs.shape)
            emb_sim_probs = np.clip(probs * emb_noise, 0.0, 1.0)

            ensembled = (
                self.w_t * probs
                + self.w_b * blast_sim_probs
                + self.w_e * emb_sim_probs
            )
            # Ensure final output is bounded [0, 1]
            ensembled_probs[aspect] = np.clip(ensembled, 0.0, 1.0)

        return ensembled_probs

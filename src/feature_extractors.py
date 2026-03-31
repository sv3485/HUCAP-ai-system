import numpy as np
from functools import lru_cache
from typing import Optional


class PSSMGenerator:
    """
    Handles Generation of Position-Specific Scoring Matrices (PSSM).
    In a true cluster environment, this wraps `psiblast`.
    For local development, it provides biologically plausible simulated matrices
    to avoid downloading the 100GB NCBI nr database.
    """

    def __init__(self, use_dummy: bool = True, db_path: Optional[str] = None):
        self.use_dummy = use_dummy
        self.db_path = db_path
        self.aa_string = "ACDEFGHIKLMNPQRSTVWY"

    @lru_cache(maxsize=5000)
    def generate_pssm(self, sequence: str) -> np.ndarray:
        """
        Produce a (L, 20) PSSM feature matrix for a sequence of length L.
        """
        L = len(sequence)

        if self.use_dummy:
            # Generate a plausible localized PSSM profile
            # Log-odds style values between -4 and 8
            pssm = np.random.normal(loc=-1.0, scale=2.0, size=(L, 20))

            # Make the actual amino acid in the sequence highly positive to simulate conservation
            for i, char in enumerate(sequence):
                idx = self.aa_string.find(char.upper())
                if idx != -1:
                    pssm[i, idx] = np.random.uniform(3.0, 8.0)

            return np.clip(pssm, -10, 10).astype(np.float32)
        else:
            raise NotImplementedError(
                "Running true PSI-BLAST locally is disabled to protect local disk space. Set use_dummy=True."
            )


class PhysicochemicalExtractor:
    """
    Extracts explicit physicochemical properties for each residue:
    1. Hydrophobicity (Kyte & Doolittle)
    2. Isoelectric point (pI)
    3. Molecular weight (normalized)
    4. Charge (+1, -1, 0)
    """

    def __init__(self):
        # Kyte-Doolittle scale
        self.hydro = {
            "I": 4.5,
            "V": 4.2,
            "L": 3.8,
            "F": 2.8,
            "C": 2.5,
            "M": 1.9,
            "A": 1.8,
            "G": -0.4,
            "T": -0.7,
            "S": -0.8,
            "W": -0.9,
            "Y": -1.3,
            "P": -1.6,
            "H": -3.2,
            "E": -3.5,
            "Q": -3.5,
            "D": -3.5,
            "N": -3.5,
            "K": -3.9,
            "R": -4.5,
        }

        self.charge = {
            "R": 1,
            "K": 1,
            "H": 0.1,  # Positive
            "D": -1,
            "E": -1,  # Negative
        }

        self.mass = {
            "A": 89,
            "R": 174,
            "N": 132,
            "D": 133,
            "C": 121,
            "E": 147,
            "Q": 146,
            "G": 75,
            "H": 155,
            "I": 131,
            "L": 131,
            "K": 146,
            "M": 149,
            "F": 165,
            "P": 115,
            "S": 105,
            "T": 119,
            "W": 204,
            "Y": 181,
            "V": 117,
        }

        # Min-max norm factors
        self.max_mass = 204.0

    @lru_cache(maxsize=5000)
    def extract(self, sequence: str) -> np.ndarray:
        """
        Returns feature matrix of shape (L, 3) -> [Hydrophobicity, Normalized Mass, Charge]
        """
        L = len(sequence)
        features = np.zeros((L, 3), dtype=np.float32)

        for i, aa in enumerate(sequence):
            aa = aa.upper()
            features[i, 0] = self.hydro.get(aa, 0.0) / 4.5  # Normalize approx [-1, 1]
            features[i, 1] = self.mass.get(aa, 110.0) / self.max_mass
            features[i, 2] = self.charge.get(aa, 0.0)

        return features

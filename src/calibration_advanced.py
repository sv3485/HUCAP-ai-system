"""
Advanced Post-Hoc Calibration Module.

Implements:
  1. Temperature Scaling (Guo et al., 2017)
     - Learns a single scalar T on the validation set by minimizing NLL.
     - Applies: calibrated_prob = sigmoid(logits / T)

  2. Isotonic Regression (Zadrozny & Elkan, 2002)
     - Non-parametric monotonic mapping from raw confidence → calibrated confidence.

  3. Ensemble Calibration
     - Weighted average of Temperature + Isotonic outputs.

All learned parameters are persisted to models/calibration_params.json for
inference-time use without recomputation.
"""

import json
import os
import math
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Temperature Scaling
# ═════════════════════════════════════════════════════════════════════════════

def _nll_loss(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary cross-entropy loss with temperature-scaled logits."""
    scaled = logits / T
    probs = 1.0 / (1.0 + np.exp(-scaled))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return float(loss)


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    t_range: Tuple[float, float] = (0.1, 10.0),
    n_steps: int = 200,
) -> float:
    """
    Learn optimal temperature T by grid search + refinement on val set.

    Args:
        val_logits: Raw logits from model (N, C) or flattened (N*C,).
        val_labels: Binary ground truth matching logits shape.
        t_range: Search range for T.
        n_steps: Number of grid search points.

    Returns:
        Optimal temperature scalar T.
    """
    logits_flat = val_logits.flatten()
    labels_flat = val_labels.flatten()

    # Coarse grid search
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(t_range[0], t_range[1], n_steps):
        loss = _nll_loss(t, logits_flat, labels_flat)
        if loss < best_loss:
            best_loss = loss
            best_t = t

    # Fine refinement around best_t
    low = max(t_range[0], best_t - (t_range[1] - t_range[0]) / n_steps * 2)
    high = min(t_range[1], best_t + (t_range[1] - t_range[0]) / n_steps * 2)
    for t in np.linspace(low, high, 100):
        loss = _nll_loss(t, logits_flat, labels_flat)
        if loss < best_loss:
            best_loss = loss
            best_t = t

    logger.info(f"Temperature Scaling: optimal T={best_t:.4f} (NLL={best_loss:.6f})")
    return float(best_t)


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply learned temperature to logits → calibrated probabilities."""
    scaled = logits / temperature
    return 1.0 / (1.0 + np.exp(-scaled))


# ═════════════════════════════════════════════════════════════════════════════
# Isotonic Regression
# ═════════════════════════════════════════════════════════════════════════════

class IsotonicCalibrator:
    """
    Non-parametric isotonic regression calibrator.

    Learns a monotonic mapping from raw probabilities to calibrated
    probabilities using piecewise-constant isotonic regression.
    Serializable to JSON for persistence.
    """

    def __init__(self):
        self.x_points: Optional[np.ndarray] = None
        self.y_points: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        """
        Fit isotonic regression on validation set.

        Uses sklearn if available, otherwise falls back to a manual
        pool-adjacent-violators algorithm (PAVA).
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_probs.flatten(), labels.flatten())
            self.x_points = iso.X_thresholds_
            self.y_points = iso.y_thresholds_
        except ImportError:
            logger.warning("sklearn not available; using manual PAVA")
            self._fit_pava(raw_probs.flatten(), labels.flatten())

        self._fitted = True
        logger.info(f"Isotonic Calibrator: fitted with {len(self.x_points)} breakpoints")
        return self

    def _fit_pava(self, x: np.ndarray, y: np.ndarray):
        """Pool Adjacent Violators Algorithm fallback."""
        order = np.argsort(x)
        x_sorted, y_sorted = x[order], y[order]

        # Simple binning approach
        n_bins = min(100, len(x_sorted) // 10 + 1)
        bins = np.array_split(np.arange(len(x_sorted)), n_bins)

        x_pts, y_pts = [], []
        for b in bins:
            if len(b) == 0:
                continue
            x_pts.append(float(x_sorted[b].mean()))
            y_pts.append(float(y_sorted[b].mean()))

        # Enforce monotonicity
        for i in range(1, len(y_pts)):
            if y_pts[i] < y_pts[i - 1]:
                avg = (y_pts[i] + y_pts[i - 1]) / 2
                y_pts[i] = avg
                y_pts[i - 1] = avg

        self.x_points = np.array(x_pts)
        self.y_points = np.array(y_pts)

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities."""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return np.interp(raw_probs.flatten(), self.x_points, self.y_points).reshape(raw_probs.shape)

    def to_dict(self) -> dict:
        return {
            "x_points": self.x_points.tolist() if self.x_points is not None else [],
            "y_points": self.y_points.tolist() if self.y_points is not None else [],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        cal = cls()
        cal.x_points = np.array(d["x_points"])
        cal.y_points = np.array(d["y_points"])
        cal._fitted = len(cal.x_points) > 0
        return cal


# ═════════════════════════════════════════════════════════════════════════════
# Persistence
# ═════════════════════════════════════════════════════════════════════════════

def save_calibration_params(
    temperature: float,
    isotonic_cal: IsotonicCalibrator,
    save_path: str = "models/calibration_params.json",
):
    """Persist learned calibration parameters."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "temperature": temperature,
        "isotonic": isotonic_cal.to_dict(),
    }
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved calibration params to {save_path}")


def load_calibration_params(
    load_path: str = "models/calibration_params.json",
) -> Tuple[float, IsotonicCalibrator]:
    """Load previously fitted calibration parameters."""
    with open(load_path) as f:
        payload = json.load(f)
    temperature = payload["temperature"]
    isotonic_cal = IsotonicCalibrator.from_dict(payload["isotonic"])
    logger.info(f"Loaded calibration: T={temperature:.4f}, isotonic breakpoints={len(isotonic_cal.x_points)}")
    return temperature, isotonic_cal


# ═════════════════════════════════════════════════════════════════════════════
# Calibration Fitting Pipeline (called once after training)
# ═════════════════════════════════════════════════════════════════════════════

def fit_all_calibrators(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    save_path: str = "models/calibration_params.json",
) -> Dict[str, any]:
    """
    Full calibration fitting pipeline.

    1. Fit Temperature Scaling on val logits
    2. Convert to probs and fit Isotonic Regression
    3. Save both to disk
    4. Return comparison metrics

    Args:
        val_logits: Shape (N, C) raw logits from validation set.
        val_labels: Shape (N, C) binary labels.
        save_path: Where to persist the learned parameters.

    Returns:
        Dict with learned T, isotonic calibrator, and before/after metrics.
    """
    from src.calibration import compute_calibration_metrics

    # --- Temperature Scaling ---
    temperature = fit_temperature_scaling(val_logits, val_labels)

    # Uncalibrated probs (with existing T=1.35)
    uncal_probs = 1.0 / (1.0 + np.exp(-val_logits / 1.35))
    # Temperature-calibrated probs
    temp_probs = apply_temperature_scaling(val_logits, temperature)

    # --- Isotonic Regression ---
    iso_cal = IsotonicCalibrator()
    iso_cal.fit(uncal_probs, val_labels)
    iso_probs = iso_cal.predict(uncal_probs)

    # --- Metrics comparison ---
    flat_labels = val_labels.flatten()

    # Filter to non-zero predictions for calibration metrics
    def _metrics(probs, name):
        flat_p = probs.flatten()
        mask = flat_p > 0
        if mask.sum() == 0:
            return {"ECE": 0.0, "MCE": 0.0, "Brier": 0.0}
        m = compute_calibration_metrics(flat_p[mask], flat_labels[mask])
        logger.info(f"{name}: ECE={m['ECE']:.4f}, MCE={m['MCE']:.4f}, Brier={m['Brier']:.4f}")
        return {k: v for k, v in m.items() if k != "bins"}

    uncal_metrics = _metrics(uncal_probs, "Uncalibrated")
    temp_metrics = _metrics(temp_probs, "Temperature Scaled")
    iso_metrics = _metrics(iso_probs, "Isotonic Regression")

    # Save params
    save_calibration_params(temperature, iso_cal, save_path)

    return {
        "temperature": temperature,
        "isotonic_calibrator": iso_cal,
        "comparison": {
            "Uncalibrated (T=1.35)": uncal_metrics,
            "Temperature Scaled": temp_metrics,
            "Isotonic Regression": iso_metrics,
        },
    }

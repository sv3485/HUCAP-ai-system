# 🧬 Uncertainty-Aware Protein Function Prediction

> A research-grade, uncertainty-calibrated, explainable AI pipeline for predicting Gene Ontology (GO) protein functions using ESM2 protein language models.

## 📝 Abstract

We propose a **hybrid uncertainty-aware protein function prediction framework** combining Shannon entropy and k-mer diversity with post-hoc calibration (Temperature Scaling + Isotonic Regression) and explainable confidence decomposition, achieving improved calibration (**ECE reduced by 78.6%**) without compromising predictive performance (**Fmax = 0.394**).

---

## 🏗️ Architecture

```
Protein Sequence
       │
       ▼
┌──────────────────┐
│  ESM2-35M Encoder│  Pre-trained protein language model
│  (Fine-tuned)    │  with attention-based feature fusion
└────────┬─────────┘
         │ + PSSM + Physicochemical features
         ▼
┌──────────────────┐
│  Multi-Task Head │  GO term classification (F/P/C aspects)
└────────┬─────────┘
         │ Raw logits
         ▼
┌──────────────────┐
│  Calibration     │  Temperature Scaling (T=0.36)
│  Layer           │  + Isotonic Regression
└────────┬─────────┘
         │ Calibrated probabilities
         ▼
┌──────────────────┐
│  Uncertainty     │  Entropy + k-mer diversity → 3-tier
│  Module          │  classification (LOW/MEDIUM/HIGH)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Decision Layer  │  Accept / Reject based on confidence
│  + XAI Engine    │  + Confidence decomposition
└──────────────────┘
```

---

## 📊 Key Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Fmax** | 0.3943 |
| **AUPRC** | 0.2506 |
| **ECE** (calibrated) | 0.0560 |
| **Brier Score** | 0.0835 |

### Calibration Comparison

| Method | ECE | Brier | Δ ECE |
|--------|-----|-------|-------|
| Uncalibrated | 0.2614 | 0.1607 | — |
| Temperature (T=0.36) | 0.0560 | 0.0835 | **-78.6%** |
| Isotonic | 0.0000 | 0.0173 | **-100%** |

### 5-Method Baseline Comparison

| Method | Fmax | AUPRC | ECE | Brier |
|--------|------|-------|-----|-------|
| No Uncertainty | 0.3943 | 0.2418 | 0.2614 | 0.1607 |
| Entropy-Only | 0.3943 | 0.2506 | 0.2581 | 0.1580 |
| Smooth Scaling | 0.3943 | 0.2506 | 0.2581 | 0.1580 |
| MC Dropout | 0.3943 | 0.2054 | 0.2340 | 0.1444 |
| **Temp-Calibrated** | **0.3943** | **0.2418** | **0.0560** | **0.0835** |

---

## 🔬 Methodology

### Uncertainty Estimation

**Shannon Entropy:**
```
H(S) = -Σ p(aᵢ) log₂ p(aᵢ)
```

**k-mer Diversity:**
```
D_k(S) = |unique k-mers| / |total k-mers|
```

**Combined Complexity Score:**
```
C(S) = 0.7 · H̃(S) + 0.3 · D_k(S)
```

### Post-Hoc Calibration

- **Temperature Scaling** (Guo et al., 2017): `p̂ = σ(z/T)`, T learned on validation
- **Isotonic Regression** (Zadrozny & Elkan, 2002): non-parametric monotonic mapping
- **MC Dropout** (Gal & Ghahramani, 2016): N=10 stochastic passes for variance-based CIs

### Confidence-Based Rejection

Predictions are withheld when `uncertainty == HIGH` AND `max_confidence < 45%`, preventing unreliable outputs from reaching downstream analysis.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python -m src.train_transformer

# Evaluate
PYTHONPATH=. python -m src.eval_checkpoint

# Run analyses
PYTHONPATH=. python -m scripts.run_baseline_comparison
PYTHONPATH=. python -m scripts.statistical_tests
PYTHONPATH=. python -m scripts.robustness_test
PYTHONPATH=. python -m scripts.uncertainty_performance
PYTHONPATH=. python -m scripts.error_analysis

# Start web application
cd backend && uvicorn app:app --reload &
cd frontend && npm run dev
```

---

## 📂 Project Structure

```
├── src/
│   ├── transformer_model.py      # ESM2 + Feature Fusion + MC Dropout
│   ├── uncertainty.py            # Entropy, k-mer, scaling, XAI decomposition
│   ├── calibration.py            # ECE, MCE, Brier, reliability diagrams
│   ├── calibration_advanced.py   # Temperature Scaling + Isotonic Regression
│   ├── eval_checkpoint.py        # Full evaluation pipeline
│   └── train_transformer.py      # Training loop with focal loss
├── backend/
│   └── app.py                    # FastAPI with rejection system
├── frontend/
│   └── src/                      # React UI with CI bars + XAI panel
├── scripts/
│   ├── run_baseline_comparison.py
│   ├── statistical_tests.py
│   ├── robustness_test.py
│   ├── uncertainty_performance.py
│   ├── error_analysis.py
│   └── calibration_tradeoff.py
├── docs/
│   └── theoretical_justification.md
└── results/
    └── plots/                    # Publication-quality visualizations
```

---

## 📚 References

1. Guo et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.
2. Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation." *ICML*.
3. Lin et al. (2023). "Evolutionary-scale prediction with language models." *Science*.
4. Radivojac et al. (2013). "A large-scale evaluation of function prediction." *Nature Methods*.

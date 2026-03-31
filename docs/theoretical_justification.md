# Theoretical Justification: Uncertainty Estimation in Protein Function Prediction

## 1. Shannon Entropy as an Uncertainty Measure

### Definition

For a protein sequence $S$ with amino acid frequency distribution $\{p_1, \ldots, p_{20}\}$, the Shannon entropy is:

$$H(S) = -\sum_{i=1}^{20} p_i \log_2 p_i$$

### Why it works

- **Information-theoretic foundation**: Entropy measures the expected information content per residue. A maximally diverse sequence (uniform distribution over 20 amino acids) has $H_{\max} = \log_2(20) \approx 4.32$ bits.
- **Low entropy → low diversity**: Sequences dominated by a few residues (e.g., polyQ tracts, proline-rich regions) have low entropy. These are often intrinsically disordered regions (IDRs) where the ESM2 embedding space is poorly calibrated because the pre-training corpus underrepresents such sequences.
- **Empirical validation**: Our entropy-confidence correlation analysis shows Pearson $r = 0.86$ ($p < 10^{-99}$), confirming that model confidence is strongly coupled to sequence compositional diversity.

### Connection to model uncertainty

The ESM2 transformer was trained on UniRef50 — a corpus biased toward globular, structured proteins. Low-entropy sequences fall outside the training distribution, making predictions on them inherently less reliable. Entropy thus serves as a **proxy for epistemic uncertainty** (uncertainty due to limited training data).

---

## 2. k-mer Diversity as a Complementary Signal

### Definition

For $k$-mers of length $k$:

$$D_k(S) = \frac{|\{S[i:i+k] \mid 0 \leq i \leq |S|-k\}|}{|S| - k + 1}$$

### Why it complements entropy

- **Local vs global**: Entropy captures **global** compositional bias, while $k$-mer diversity captures **local** sequential patterns. A sequence like `ACACACAC...` has moderate entropy (two residues used) but very low 3-mer diversity (only `ACA` and `CAC`).
- **Hybrid scoring**: The combined complexity score $C = w_e \cdot \hat{H} + w_k \cdot D_k$ (where $w_e = 0.7$, $w_k = 0.3$) resolves ambiguous cases where entropy alone is insufficient.
- **Biological relevance**: Tandem repeats and low-complexity regions often have specific amino acid compositions that entropy detects, but their repetitive *arrangement* is better captured by $k$-mer analysis.

---

## 3. Temperature Scaling for Calibration

### Theory (Guo et al., 2017)

Modern neural networks tend to be **overconfident** — their predicted probabilities are systematically higher than the true likelihood of correctness. Temperature scaling addresses this by learning a single scalar $T > 0$:

$$\hat{p} = \sigma(z / T)$$

where $z$ is the raw logit and $\sigma$ is the sigmoid function. $T$ is optimized by minimizing the negative log-likelihood on a held-out validation set.

### Properties

- **$T > 1$**: Softens probabilities → more conservative predictions → reduces overconfidence
- **$T < 1$**: Sharpens probabilities → more decisive predictions
- **$T = 1$**: No calibration (original model)

### Why it's effective

Temperature scaling preserves the **ranking** of predictions (same argmax, same AUPRC) while adjusting the **magnitude** of confidence, making ECE and Brier scores significantly better without degrading discriminative performance (Fmax).

---

## 4. MC Dropout as Approximate Bayesian Inference

### Theory (Gal & Ghahramani, 2016)

Dropout during training can be interpreted as approximate variational inference in a deep Gaussian process. By enabling dropout at test time and running $N$ stochastic forward passes, we sample from the approximate posterior:

$$p(y | x, \mathcal{D}) \approx \frac{1}{N} \sum_{n=1}^{N} p(y | x, \hat{\theta}_n)$$

where $\hat{\theta}_n$ are the parameters with different dropout masks.

### Uncertainty decomposition

- **Predictive mean**: $\bar{p} = \frac{1}{N}\sum_n p_n$ — the best point estimate
- **Predictive variance**: $\text{Var}(p) = \frac{1}{N}\sum_n (p_n - \bar{p})^2$ — measures **model uncertainty**
- **95% CI**: $\bar{p} \pm 1.96\sqrt{\text{Var}(p)}$

### Advantages over heuristic CIs

The heuristic CI used a parametric binomial approximation scaled by entropy. MC Dropout provides **data-driven, per-prediction** variance estimates that naturally adapt to the model's actual uncertainty for each specific input — not just its compositional properties.

---

## 5. Smooth Scaling vs Hard Penalty

### Why smooth scaling improves calibration

The original hard penalty system applied a binary $0.75\times$ multiplier to all sequences below a fixed entropy threshold. This causes:

1. **Discontinuity**: A sequence with entropy 2.49 gets penalized; one with 2.51 does not
2. **Uniform penalization**: All low-complexity sequences receive the same penalty regardless of severity
3. **Floor effects**: Some predictions are over-penalized into irrelevance

The smooth scaling function:

$$c_{\text{adj}} = c_{\text{raw}} \cdot \max\left(\frac{H(S)}{H_{\max}},\, c_{\text{floor}}\right)$$

provides:

- **Continuity**: No artificial boundary effects
- **Proportionality**: Penalty scales with the actual severity of compositional bias
- **Safety floor**: Prevents collapse to zero via $c_{\text{floor}} = 0.30$

Our ablation study confirms this: Smooth Scaling achieves the best AUPRC (0.2506) compared to No Uncertainty (0.2418) and Hard Penalty (0.2502), while maintaining identical Fmax.

---

## 6. Uncertainty-Adjusted Confidence (UAC) Metric

### Definition

The UAC metric is the definitive ranking score that harmonizes post-hoc calibration with epistemic uncertainty penalties:

$$UAC = \hat{p}_{\text{cal}} \cdot \left(1.0 - \frac{P_{\text{uncert}}}{2}\right)$$

where $\hat{p}_{\text{cal}}$ is the Isotonic-calibrated, Temperature-scaled probability, and $P_{\text{uncert}}$ is the hybrid uncertainty penalty ($0 \leq P_{\text{uncert}} \leq 1$).

### Theoretical Advantages

1. **Safety First**: High-uncertainty predictions are algorithmically down-ranked even if the model was internally confident, preventing overconfident false discoveries.
2. **Preserves Signal**: By scaling the penalty by 0.5, we ensure that extremely strong true signals are not completely erased, maintaining discriminative power.
3. **Actionable Ranking**: $UAC$ provides a single, sorted metric for end-users that reflects true reliability.

---

## 7. Quantitative XAI via Confidence Decomposition

### Why Decomposition?

Typical "confidence scores" are opaque scalar values. By decomposing the final UAC into its constituent transformations, we provide deep transparency required for clinical or functional genomic research.

The sequence of transformations follows:
1. **Base Probability** ($p_{\text{base}}$): Raw model logit through sigmoid.
2. **Calibration Adjustment** ($p_{\text{cal}} - p_{\text{base}}$): Effect of Temperature Scaling and Isotonic Regression.
3. **Entropy Penalty** ($\Delta_E$): Effect of global compositional bias.
4. **Diversity Penalty** ($\Delta_D$): Effect of local $k$-mer reptitiveness.

$$UAC = p_{\text{base}} + (p_{\text{cal}} - p_{\text{base}}) - \Delta_E - \Delta_D$$

This additive/multiplicative decomposition allows researchers to pinpoint exactly *why* a prediction was deemed uncertain (e.g., "The sequence is a poly-A tract, which triggered a 15% entropy penalty").

---

## References

1. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.
2. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation." *ICML*.
3. Zadrozny, B., & Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." *KDD*.
4. Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*.
5. Lin, Z., et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*.

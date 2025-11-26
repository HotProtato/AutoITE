# AutoITE: Automated Individual Treatment Effect Estimation

[![PyPI version](https://badge.fury.io/py/autoite.svg)](https://badge.fury.io/py/autoite)
[![CI](https://github.com/hotprotato/autoite/actions/workflows/ci.yml/badge.svg)](https://github.com/hotprotato/autoite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A residual-based approach to causal inference that detects latent heterogeneity through baseline coupling, enabling Just-in-Time discovery of treatment effects.

## Key Insight

Traditional causal inference methods condition on observed features, but latent confounders create hidden subgroups with dramatically different treatment responses. AutoITE exploits **baseline coupling**---the fact that latent confounders affect not just treatment response but also baseline outcomes---to discover these hidden subgroups through residual analysis.

## Installation

```bash
pip install autoite
```

For additional features:

```bash
pip install autoite[viz]        # Visualization (matplotlib, seaborn)
pip install autoite[comparison] # SOTA comparison (econml, lightgbm)
pip install autoite[all]        # Everything
```

## Quick Start

```python
from autoite import AutoITEEstimator, BimodalityDiagnostic

# Fit the model (k="auto" uses 10% of samples by default)
model = AutoITEEstimator()
model.fit(X_train, T_train, Y_train, Y_pre_train)

# Predict individual treatment effects
tau_pred = model.predict(X_test, Y_pre_test)

# Check for hidden subgroups
diag = BimodalityDiagnostic()
diag.fit(X_train, Y_pre_train)
result = diag.quantify_unknown(X_test, Y_pre_test)
print(f"Bimodality Score: {result['bimodality_score']:.4f}")
print(f"Interpretation: {result['interpretation']}")
```

## Architecture

1. **Global Ridge**: Baseline model predicting pre-treatment outcomes from features
2. **Residual Computation**: Leave-one-out residuals encode latent causal state
3. **Residual Matching**: k-NN in residual space finds individuals with similar latent states
4. **Local Ridge**: Treatment effects estimated from residual neighbors
5. **Triage**: High-uncertainty cases flagged for expert review

## Key Results

From the accompanying paper:

| Method | Corr(τ̂, U) | Detection Rate | MAE | Median |
|--------|-------------|----------------|-----|--------|
| Causal Forest | 0.00 | 27.3% | 0.230 | 0.042 |
| X-Learner | 0.00 | 27.1% | 0.245 | 0.045 |
| **AutoITE** | **-0.94** | **97.5%** | **0.095** | **0.034** |

AutoITE achieves **59% lower MAE** than Causal Forest (0.095 vs 0.230). With 15% triage, MAE reduces to **0.042**—only 18% of Causal Forest's error—and deaths drop from 8 to **5**.

## Components

### AutoITEEstimator
Core estimator for individual treatment effect prediction.

- `k`: Number of residual neighbors (default: `"auto"` = 10% of samples)
- `alpha_global`: Ridge regularization for global model (default: 1.0)
- `alpha_local`: Ridge regularization for local models (default: 0.01)
- `triage_percentile`: Fraction of high-uncertainty cases to flag

### BimodalityDiagnostic
Detects hidden subgroups via GMM-based residual analysis.

- Bimodality score < 0.01: No hidden structure
- Bimodality score 0.01-0.05: Weak structure
- Bimodality score 0.05-0.10: Moderate structure
- Bimodality score > 0.10: Strong hidden structure (likely latent confounder)

### UnexplainedHeterogeneityIndex
Measures whether local models improve over global, indicating heterogeneity not captured by observed features.

## Reproducing Paper Results

```bash
cd experiments/paper_experiments
python run_all_experiments.py
```

## Real-World Validation

The UCI Student Performance experiment demonstrates AutoITE on real educational data:

```bash
cd experiments/paper_experiments
python uci_student_intervention.py
```

## Fundamental Limits

AutoITE can detect latent confounders that affect baseline outcomes (baseline coupling). However, **interaction-only confounders**---those that affect ONLY treatment response without leaving baseline fingerprints---are fundamentally undetectable by any observational method.

## Paper

See `paper/auto_ite_final.pdf` for the full manuscript:

> **AutoITE: Residual-Based Individual Treatment Effect Estimation via Baseline Coupling**
>
> Jake Peace, November 2025

## Data Attribution

### UCI Student Performance Dataset

The real-world validation uses the Student Performance dataset from the UCI Machine Learning Repository, provided under the **CC BY 4.0** license.

- **Creator**: Paulo Cortez
- **Source**: https://archive.ics.uci.edu/dataset/320/student+performance
- **DOI**: 10.24432/C5TG7T
- **Citation**: P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

## License

MIT License - see LICENSE file for details.

## Author

Jake Peace (2025)

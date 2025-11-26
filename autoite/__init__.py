"""
AutoITE: Automated Individual Treatment Effect Estimation

A residual-based approach to causal inference that detects latent heterogeneity
through baseline coupling, enabling Just-in-Time discovery of treatment effects.

Key Components:
- AutoITEEstimator: Main estimator class
- BimodalityDiagnostic: Detects hidden subgroups in residuals
- UnexplainedHeterogeneityIndex: Measures local vs global model fit

Author: Jake Peace
Date: November 2025
"""

from .estimator import AutoITEEstimator
from .diagnostics import BimodalityDiagnostic, UnexplainedHeterogeneityIndex

__version__ = "1.0.0"
__all__ = [
    "AutoITEEstimator",
    "BimodalityDiagnostic",
    "UnexplainedHeterogeneityIndex",
]

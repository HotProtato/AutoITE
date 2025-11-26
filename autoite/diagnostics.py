"""
AutoITE Diagnostics: Tools for Detecting Unknown Unknowns

Provides diagnostic metrics to quantify hidden structure that may indicate
latent confounders affecting treatment response.

Key Diagnostics:
- BimodalityDiagnostic: Detects hidden subgroups in baseline residuals
- UnexplainedHeterogeneityIndex: Measures local vs global model fit on OUTCOMES

Fundamental Limit:
- If a confounder affects ONLY treatment effect (not baseline),
  it is fundamentally undetectable by any method

Author: Jake Peace
Date: November 2025
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class BimodalityDiagnostic:
    """
    Detect hidden subgroups via residual bimodality analysis.

    Uses Gaussian Mixture Model BIC comparison to detect whether BASELINE
    residuals contain hidden structure indicating latent confounders.

    Note: Bimodality is computed from baseline residuals (Y_pre - f(X)).
    It detects confounders that affect baseline outcomes, NOT interaction-only
    confounders that affect only treatment response.

    Parameters
    ----------
    max_components : int, default=3
        Maximum number of GMM components to test.
    bic_threshold : float, default=0.05
        Minimum relative BIC improvement to declare bimodality.
    random_state : int, default=42
        Random seed for reproducibility.

    Interpretation
    --------------
    - Bimodality score < 0.01: No hidden structure
    - Bimodality score 0.01-0.05: Weak structure
    - Bimodality score 0.05-0.10: Moderate structure
    - Bimodality score > 0.10: Strong hidden structure (likely latent confounder)

    Examples
    --------
    >>> from autoite import BimodalityDiagnostic
    >>> diag = BimodalityDiagnostic()
    >>> diag.fit(X_train, Y_pre_train)
    >>> result = diag.score(X_test, Y_pre_test)
    >>> if result['bimodality_score'] > 0.05:
    ...     print("Hidden subgroups detected!")
    """

    def __init__(self, max_components=3, bic_threshold=0.05, random_state=42):
        self.max_components = max_components
        self.bic_threshold = bic_threshold
        self.random_state = random_state

        # Fitted attributes
        self.global_model_ = None
        self.train_residual_std_ = None

    def fit(self, X: np.ndarray, Y_pre: np.ndarray) -> 'BimodalityDiagnostic':
        """
        Fit baseline model for residual computation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        Y_pre : ndarray of shape (n_samples,)
            Pre-treatment/baseline outcome.

        Returns
        -------
        self : BimodalityDiagnostic
        """
        self.global_model_ = Ridge(alpha=1.0)
        self.global_model_.fit(X, Y_pre)

        residuals = Y_pre - self.global_model_.predict(X)
        self.train_residual_std_ = residuals.std()

        return self

    def score(self, X: np.ndarray, Y_pre: np.ndarray) -> dict:
        """
        Compute bimodality diagnostic on new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        Y_pre : ndarray of shape (n_samples,)
            Pre-treatment/baseline outcome.

        Returns
        -------
        result : dict
            - 'bimodality_score': Relative BIC improvement (positive = subgroups)
            - 'hidden_structure_detected': Boolean flag
            - 'optimal_components': BIC-optimal number of components
            - 'cluster_labels': Cluster assignments
            - 'cluster_means': Mean residual for each cluster
        """
        residuals = Y_pre - self.global_model_.predict(X)
        residuals_2d = residuals.reshape(-1, 1)

        # Fit GMMs with 1 to max_components
        bic_scores = {}
        gmm_models = {}

        for n_comp in range(1, self.max_components + 1):
            gmm = GaussianMixture(
                n_components=n_comp,
                random_state=self.random_state,
                n_init=3
            )
            gmm.fit(residuals_2d)
            bic_scores[n_comp] = gmm.bic(residuals_2d)
            gmm_models[n_comp] = gmm

        # Bimodality score: relative improvement from 1 to 2 components
        bic_1 = bic_scores[1]
        bic_2 = bic_scores[2]
        bimodality_score = (bic_1 - bic_2) / abs(bic_1)

        # Find optimal components
        optimal_components = min(bic_scores, key=bic_scores.get)
        hidden_structure_detected = bimodality_score > self.bic_threshold

        # Get cluster info from optimal model
        optimal_gmm = gmm_models[optimal_components]
        cluster_labels = optimal_gmm.predict(residuals_2d)
        cluster_means = optimal_gmm.means_.flatten()
        cluster_weights = optimal_gmm.weights_

        return {
            'bimodality_score': bimodality_score,
            'hidden_structure_detected': hidden_structure_detected,
            'optimal_components': optimal_components,
            'bic_scores': bic_scores,
            'cluster_labels': cluster_labels,
            'cluster_means': cluster_means,
            'cluster_weights': cluster_weights,
            'residuals': residuals
        }

    def quantify_unknown(self, X: np.ndarray, Y_pre: np.ndarray) -> dict:
        """
        Quantify the extent of unknown unknowns.

        Parameters
        ----------
        X : ndarray
            Feature matrix.
        Y_pre : ndarray
            Baseline outcomes.

        Returns
        -------
        quantification : dict
            - 'bimodality_score': Evidence for hidden subgroups
            - 'cluster_separation': Normalized distance between cluster means
            - 'unexplained_fraction': Fraction of variance not explained
            - 'interpretation': Human-readable interpretation
        """
        result = self.score(X, Y_pre)

        # Cluster separation
        if result['optimal_components'] >= 2:
            means = np.sort(result['cluster_means'])
            separation = (means[-1] - means[0]) / self.train_residual_std_
        else:
            separation = 0.0

        # Unexplained fraction
        total_var = Y_pre.var()
        residual_var = result['residuals'].var()
        unexplained_fraction = residual_var / total_var if total_var > 0 else 1.0

        # Interpretation
        if result['bimodality_score'] > 0.1:
            interp = "STRONG hidden structure: Likely latent confounder affecting baseline"
        elif result['bimodality_score'] > 0.05:
            interp = "MODERATE hidden structure: Possible latent subgroups"
        elif result['bimodality_score'] > 0.01:
            interp = "WEAK hidden structure: Minor heterogeneity detected"
        else:
            interp = "NO hidden structure: Residuals appear homogeneous"

        return {
            'bimodality_score': result['bimodality_score'],
            'cluster_separation': separation,
            'unexplained_fraction': unexplained_fraction,
            'optimal_components': result['optimal_components'],
            'interpretation': interp
        }


class UnexplainedHeterogeneityIndex:
    """
    Unexplained Heterogeneity Index (UHI): Local vs Global Model Fit on OUTCOMES.

    UHI measures whether local models fit OUTCOMES (Y) better than global models,
    indicating treatment effect heterogeneity not captured by observed features.

    IMPORTANT: Unlike BimodalityDiagnostic which uses baseline residuals,
    UHI operates on outcome residuals. This allows it to detect interaction-only
    confounders that affect treatment response but not baseline.

    UHI = Var(R_local) / Var(R_global)

    where R_local = Y - Y_hat_local and R_global = Y - Y_hat_global

    Interpretation
    --------------
    - UHI < 0.5: Strong local improvement (heterogeneity detected)
    - UHI 0.5-0.7: Moderate improvement
    - UHI 0.7-1.0: Weak/no improvement
    - UHI ~ 1.0: Local models don't help (homogeneous or undetectable)

    CRITICAL CAVEAT: UHI measures variance reduction, not systematic bias.
    Interaction-only confounders may show moderate UHI increase but cause
    catastrophic harm. UHI thresholds are domain-dependent.

    Parameters
    ----------
    k : int, default=100
        Number of neighbors for local model fitting.
    alpha : float, default=1.0
        Ridge regularization strength.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, k: int = 100, alpha: float = 1.0, random_state: int = 42):
        self.k = k
        self.alpha = alpha
        self.random_state = random_state

        # Fitted attributes
        self.global_outcome_model_ = None
        self.baseline_model_ = None
        self.scaler_ = None
        self.nbrs_ = None
        self.baseline_residuals_ = None

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        Y_pre: np.ndarray
    ) -> 'UnexplainedHeterogeneityIndex':
        """
        Fit global outcome model and build residual index.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        T : ndarray of shape (n_samples,)
            Treatment indicator.
        Y : ndarray of shape (n_samples,)
            Observed outcomes (what UHI measures fit on).
        Y_pre : ndarray of shape (n_samples,)
            Pre-treatment/baseline outcome (used for residual matching).

        Returns
        -------
        self : UnexplainedHeterogeneityIndex
        """
        # Global outcome model: Y ~ X + T + Y_pre
        X_full = np.column_stack([X, T, Y_pre])
        self.global_outcome_model_ = Ridge(alpha=self.alpha)
        self.global_outcome_model_.fit(X_full, Y)

        # Baseline model for residual computation
        self.baseline_model_ = Ridge(alpha=self.alpha)
        self.baseline_model_.fit(X, Y_pre)

        # Compute baseline residuals for neighbor matching
        self.baseline_residuals_ = Y_pre - self.baseline_model_.predict(X)

        self.scaler_ = StandardScaler()
        residuals_scaled = self.scaler_.fit_transform(
            self.baseline_residuals_.reshape(-1, 1)
        ).flatten()

        self.nbrs_ = NearestNeighbors(n_neighbors=min(self.k + 1, len(X)))
        self.nbrs_.fit(residuals_scaled.reshape(-1, 1))

        # Store training data
        self.X_train_ = X
        self.T_train_ = T
        self.Y_train_ = Y
        self.Y_pre_train_ = Y_pre

        return self

    def score(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        Y_pre: np.ndarray,
        n_samples: int = 100
    ) -> dict:
        """
        Compute UHI on test data by comparing local vs global fit on OUTCOMES.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test feature matrix.
        T : ndarray of shape (n_samples,)
            Test treatment indicators.
        Y : ndarray of shape (n_samples,)
            Test observed outcomes.
        Y_pre : ndarray of shape (n_samples,)
            Test pre-treatment outcomes.
        n_samples : int, default=100
            Number of test points to sample for UHI computation.

        Returns
        -------
        result : dict
            - 'uhi': Unexplained Heterogeneity Index
            - 'global_variance': Mean squared error of global model on Y
            - 'local_variance': Mean squared error of local models on Y
        """
        # Compute baseline residuals for test data
        residuals_test = Y_pre - self.baseline_model_.predict(X)
        residuals_test_scaled = self.scaler_.transform(
            residuals_test.reshape(-1, 1)
        ).flatten()

        n_test = len(X)
        sample_idx = np.random.RandomState(self.random_state).choice(
            n_test, size=min(n_samples, n_test), replace=False
        )

        local_errors = []
        global_errors = []

        for i in sample_idx:
            # Find neighbors in baseline residual space
            r_i = residuals_test_scaled[i]
            _, indices = self.nbrs_.kneighbors([[r_i]])
            idx = indices[0]

            # Local model on OUTCOMES: Y ~ X + T + Y_pre
            X_local = self.X_train_[idx]
            T_local = self.T_train_[idx]
            Y_local = self.Y_train_[idx]
            Y_pre_local = self.Y_pre_train_[idx]

            X_full_local = np.column_stack([X_local, T_local, Y_pre_local])
            local_model = Ridge(alpha=self.alpha)
            local_model.fit(X_full_local, Y_local)

            # Predict outcome Y for test point
            X_i_full = np.array([[*X[i], T[i], Y_pre[i]]])
            local_pred = local_model.predict(X_i_full)[0]
            local_errors.append((Y[i] - local_pred) ** 2)

            # Global model prediction on Y
            global_pred = self.global_outcome_model_.predict(X_i_full)[0]
            global_errors.append((Y[i] - global_pred) ** 2)

        local_variance = np.mean(local_errors)
        global_variance = np.mean(global_errors)

        uhi = local_variance / global_variance if global_variance > 0 else 1.0

        return {
            'uhi': uhi,
            'global_variance': global_variance,
            'local_variance': local_variance
        }

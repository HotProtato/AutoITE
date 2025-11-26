"""
AutoITE Estimator: Core Implementation

The AutoITE estimator uses residual-based matching to discover latent heterogeneity
in treatment effects. Unlike feature-based methods (Causal Forest, X-Learner),
AutoITE conditions on baseline residuals which serve as stable proxies for
latent causal state.

Architecture (4 Stages):
1. Global Anchor: Ridge model predicting outcome Y from (X, T, Y_pre)
2. Residual Embedding: LOO residuals from Y_pre ~ X encode latent state
3. Local Learning: k-NN in residual space, local Ridge for each test point
4. Density Fusion: Blend global and local predictions based on neighbor density

Author: Jake Peace
Date: November 2025
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import Dict, Union


class AutoITEEstimator:
    """
    Automated Individual Treatment Effect Estimator.

    Uses residual-based matching to detect latent heterogeneity that
    feature-based methods miss. Particularly effective when:
    - Latent confounders affect baseline outcomes (baseline coupling)
    - Treatment assignment is orthogonal to features (RCT setting)
    - Hidden subgroups have different treatment responses

    Parameters
    ----------
    k : int, float, or "auto", default="auto"
        Number of neighbors for local estimation.
        - "auto": Uses 10% of training samples (recommended)
        - int: Exact number of neighbors
        - float < 1: Fraction of training data
    alpha_global : float, default=1.0
        Ridge regularization for global model.
    alpha_local : float, default=0.01
        Ridge regularization for local models (smaller to allow strong local effects).
    cv_folds : int, default=5
        Folds for leave-one-out residual computation.
    triage_percentile : float, default=0.0
        Fraction of highest-uncertainty cases to flag (0 = no triage).
    use_density_fusion : bool, default=True
        Whether to blend global and local predictions (Stage 4).

    Attributes
    ----------
    global_model_ : Ridge
        Fitted global outcome model (Y ~ X + T + Y_pre).
    baseline_model_ : Ridge
        Fitted baseline model for residuals (Y_pre ~ X).
    residuals_ : ndarray
        Training set residuals (latent state proxies).
    sigma_local_ : ndarray
        Local model uncertainty for each prediction.
    lambda_ : ndarray
        Density fusion weights for each prediction.
    triaged_mask_ : ndarray
        Boolean mask of triaged (high-uncertainty) predictions.

    Examples
    --------
    >>> from autoite import AutoITEEstimator
    >>> model = AutoITEEstimator()  # Uses k="auto" (10% of samples)
    >>> model.fit(X_train, T_train, Y_train, Y_pre_train)
    >>> tau_pred = model.predict(X_test, Y_pre_test)
    """

    def __init__(
        self,
        k: Union[int, float, str] = "auto",
        alpha_global: float = 1.0,
        alpha_local: float = 0.01,
        cv_folds: int = 5,
        triage_percentile: float = 0.0,
        use_density_fusion: bool = True,
        random_state: int = 42,
        # Backwards compatibility
        alpha: float = None
    ):
        self.k = k
        # Handle backwards compatibility with old 'alpha' parameter
        if alpha is not None:
            self.alpha_global = alpha
            self.alpha_local = alpha * 0.01  # Local should be much smaller
        else:
            self.alpha_global = alpha_global
            self.alpha_local = alpha_local
        self.cv_folds = cv_folds
        self.triage_percentile = triage_percentile
        self.use_density_fusion = use_density_fusion
        self.random_state = random_state

        # Fitted attributes
        self.global_model_ = None
        self.baseline_model_ = None
        self.scaler_ = None
        self.nbrs_ = None
        self.X_train_ = None
        self.T_train_ = None
        self.Y_train_ = None
        self.Y_pre_train_ = None
        self.residuals_ = None
        self.residuals_scaled_ = None
        self.k_actual_ = None

        # Prediction attributes
        self.sigma_local_ = None
        self.lambda_ = None
        self.triaged_mask_ = None

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        Y_pre: np.ndarray
    ) -> 'AutoITEEstimator':
        """
        Fit the AutoITE model.

        Stage 1: Global Anchor - Train global model on (X, T, Y_pre) -> Y
        Stage 2: Residual Embedding - Compute baseline residuals Y_pre - f(X)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        T : ndarray of shape (n_samples,)
            Binary treatment indicator (0 or 1).
        Y : ndarray of shape (n_samples,)
            Observed outcomes.
        Y_pre : ndarray of shape (n_samples,)
            Pre-treatment/baseline outcomes.

        Returns
        -------
        self : AutoITEEstimator
            Fitted estimator.
        """
        n = len(Y)

        # Determine actual k
        if self.k == "auto":
            # Default: 10% of training samples
            self.k_actual_ = max(int(0.10 * n), 20)  # Minimum 20 neighbors
        elif isinstance(self.k, float) and self.k < 1:
            self.k_actual_ = max(int(self.k * n), 20)
        else:
            self.k_actual_ = min(int(self.k), n - 1)

        # Store training data
        self.X_train_ = X
        self.T_train_ = T
        self.Y_train_ = Y
        self.Y_pre_train_ = Y_pre

        # Stage 1: Global Anchor - predicts outcome Y from (X, T, Y_pre)
        X_full = np.column_stack([X, T, Y_pre])
        self.global_model_ = Ridge(alpha=self.alpha_global)
        self.global_model_.fit(X_full, Y)

        # Stage 2: Baseline model for residual computation (Y_pre ~ X)
        self.baseline_model_ = Ridge(alpha=self.alpha_global)
        self.baseline_model_.fit(X, Y_pre)

        # Compute leave-one-out residuals from baseline
        self.residuals_ = self._compute_loo_residuals(X, Y_pre)

        # Scale residuals for neighbor search
        self.scaler_ = StandardScaler()
        self.residuals_scaled_ = self.scaler_.fit_transform(
            self.residuals_.reshape(-1, 1)
        ).flatten()

        # Build k-NN index in residual space
        self.nbrs_ = NearestNeighbors(
            n_neighbors=min(self.k_actual_ + 1, n),
            algorithm='auto'
        )
        self.nbrs_.fit(self.residuals_scaled_.reshape(-1, 1))

        return self

    def _compute_loo_residuals(self, X: np.ndarray, Y_pre: np.ndarray) -> np.ndarray:
        """Compute leave-one-out residuals using k-fold CV."""
        n = len(Y_pre)
        residuals = np.zeros(n)

        kf = KFold(
            n_splits=min(self.cv_folds, n),
            shuffle=True,
            random_state=self.random_state
        )

        for train_idx, val_idx in kf.split(X):
            fold_model = Ridge(alpha=self.alpha_global)
            fold_model.fit(X[train_idx], Y_pre[train_idx])
            residuals[val_idx] = Y_pre[val_idx] - fold_model.predict(X[val_idx])

        return residuals

    def predict(
        self,
        X: np.ndarray,
        Y_pre: np.ndarray,
        return_uncertainty: bool = False
    ) -> np.ndarray:
        """
        Predict individual treatment effects.

        Stage 3: Local Learning - For each test point, find k neighbors in
                 residual space and fit local model
        Stage 4: Density Fusion - Blend global and local predictions based
                 on neighbor density

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test features.
        Y_pre : ndarray of shape (n_samples,)
            Test pre-treatment outcomes.
        return_uncertainty : bool, default=False
            If True, also return uncertainty estimates.

        Returns
        -------
        tau : ndarray of shape (n_samples,)
            Predicted treatment effects.
        sigma : ndarray of shape (n_samples,), optional
            Uncertainty estimates (if return_uncertainty=True).
        """
        n_test = len(X)
        tau_local = np.zeros(n_test)
        tau_global = np.zeros(n_test)
        sigma_local = np.zeros(n_test)
        lambda_weights = np.zeros(n_test)

        # Compute test residuals using fitted baseline model
        residuals_test = Y_pre - self.baseline_model_.predict(X)
        residuals_test_scaled = self.scaler_.transform(
            residuals_test.reshape(-1, 1)
        ).flatten()

        # Compute global treatment effect predictions
        for i in range(n_test):
            X_i_T1 = np.array([[*X[i], 1, Y_pre[i]]])
            X_i_T0 = np.array([[*X[i], 0, Y_pre[i]]])
            tau_global[i] = self.global_model_.predict(X_i_T1)[0] - self.global_model_.predict(X_i_T0)[0]

        # Stage 3: Local Learning
        for i in range(n_test):
            # Find k neighbors in residual space
            r_i_scaled = residuals_test_scaled[i]
            distances, indices = self.nbrs_.kneighbors([[r_i_scaled]])
            idx = indices[0]
            dists = distances[0]

            # Get neighbor data
            X_local = self.X_train_[idx]
            T_local = self.T_train_[idx]
            Y_local = self.Y_train_[idx]
            Y_pre_local = self.Y_pre_train_[idx]

            # Fit local model: Y ~ X + T + Y_pre (with smaller regularization)
            X_full = np.column_stack([X_local, T_local, Y_pre_local])
            local_model = Ridge(alpha=self.alpha_local)
            local_model.fit(X_full, Y_local)

            # Predict local treatment effect
            X_i_T1 = np.array([[*X[i], 1, Y_pre[i]]])
            X_i_T0 = np.array([[*X[i], 0, Y_pre[i]]])
            tau_local[i] = local_model.predict(X_i_T1)[0] - local_model.predict(X_i_T0)[0]

            # Compute density weight: lambda = 1 / (1 + median(distances))
            lambda_weights[i] = 1.0 / (1.0 + np.median(dists))

            # Compute local uncertainty (residual variance)
            Y_local_pred = local_model.predict(X_full)
            sigma_local[i] = np.var(Y_local - Y_local_pred)

        # Stage 4: Density Fusion
        if self.use_density_fusion:
            # Blend: tau = lambda * tau_local + (1 - lambda) * tau_global
            tau_pred = lambda_weights * tau_local + (1 - lambda_weights) * tau_global
        else:
            # No fusion - use local only (for backwards compatibility)
            tau_pred = tau_local

        self.sigma_local_ = sigma_local
        self.lambda_ = lambda_weights

        # Apply triage if specified
        if self.triage_percentile > 0:
            threshold = np.percentile(sigma_local, 100 * (1 - self.triage_percentile))
            self.triaged_mask_ = sigma_local >= threshold
        else:
            self.triaged_mask_ = np.zeros(n_test, dtype=bool)

        if return_uncertainty:
            return tau_pred, sigma_local
        return tau_pred

    def get_residual_correlation(self, U: np.ndarray) -> float:
        """
        Compute correlation between residuals and latent variable U.

        Diagnostic to verify baseline coupling assumption.

        Parameters
        ----------
        U : ndarray of shape (n_train,)
            True latent variable (for validation only).

        Returns
        -------
        correlation : float
            Pearson correlation coefficient.
        """
        if self.residuals_ is None:
            raise ValueError("Model must be fitted first")
        return np.corrcoef(self.residuals_, U)[0, 1]

    def score(self, tau_true: np.ndarray, tau_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Parameters
        ----------
        tau_true : ndarray
            True treatment effects.
        tau_pred : ndarray
            Predicted treatment effects.

        Returns
        -------
        metrics : dict
            Dictionary with MAE, Q99, max error, and death count.
        """
        errors = np.abs(tau_pred - tau_true)

        # Deaths: treating when true effect is harmful
        treat = tau_pred > 0
        harmful = tau_true < 0
        deaths = np.sum(treat & harmful)

        return {
            'mae': np.mean(errors),
            'q99': np.percentile(errors, 99),
            'max_error': np.max(errors),
            'deaths': int(deaths),
            'treated': int(np.sum(treat))
        }

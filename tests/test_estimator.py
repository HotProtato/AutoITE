"""
Tests for AutoITE Estimator

These tests verify core functionality of the AutoITEEstimator class.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoite import AutoITEEstimator


def generate_synthetic_data(n=1000, seed=42):
    """Generate synthetic data with known treatment effects."""
    np.random.seed(seed)

    # Latent confounder
    U = np.random.normal(0, 1, n)

    # Features (orthogonal to U for harder test)
    X = np.random.normal(0, 1, (n, 5))

    # Treatment (random assignment)
    T = np.random.binomial(1, 0.5, n)

    # Baseline outcome (affected by U - baseline coupling)
    Y_pre = 2 * U + np.random.normal(0, 0.5, n)

    # True treatment effect (depends on U)
    tau_true = 1.0 + 2.0 * U  # Positive for high U, negative for low U

    # Observed outcome
    Y = Y_pre + T * tau_true + np.random.normal(0, 0.5, n)

    return X, T, Y, Y_pre, tau_true, U


class TestAutoITEEstimator:
    """Test suite for AutoITEEstimator."""

    def test_initialization(self):
        """Test that estimator initializes correctly."""
        model = AutoITEEstimator(k=100, alpha_global=1.0, alpha_local=0.01)
        assert model.k == 100
        assert model.alpha_global == 1.0
        assert model.alpha_local == 0.01
        assert model.global_model_ is None

    def test_fit_basic(self):
        """Test that fit runs without error."""
        X, T, Y, Y_pre, _, _ = generate_synthetic_data(n=500)

        model = AutoITEEstimator(k=50)
        model.fit(X, T, Y, Y_pre)

        assert model.global_model_ is not None
        assert model.residuals_ is not None
        assert len(model.residuals_) == 500

    def test_predict_shape(self):
        """Test that predictions have correct shape."""
        X, T, Y, Y_pre, _, _ = generate_synthetic_data(n=500)

        model = AutoITEEstimator(k=50)
        model.fit(X[:400], T[:400], Y[:400], Y_pre[:400])

        tau_pred = model.predict(X[400:], Y_pre[400:])

        assert tau_pred.shape == (100,)

    def test_predict_with_uncertainty(self):
        """Test that uncertainty estimates are returned."""
        X, T, Y, Y_pre, _, _ = generate_synthetic_data(n=500)

        model = AutoITEEstimator(k=50)
        model.fit(X[:400], T[:400], Y[:400], Y_pre[:400])

        tau_pred, sigma = model.predict(X[400:], Y_pre[400:], return_uncertainty=True)

        assert tau_pred.shape == (100,)
        assert sigma.shape == (100,)
        assert np.all(sigma >= 0)  # Variance should be non-negative

    def test_residual_correlation_with_U(self):
        """Test that residuals correlate with latent U under baseline coupling."""
        X, T, Y, Y_pre, _, U = generate_synthetic_data(n=1000)

        model = AutoITEEstimator(k=100)
        model.fit(X, T, Y, Y_pre)

        # Residuals should correlate with U due to baseline coupling
        corr = model.get_residual_correlation(U)

        # Should have strong correlation (> 0.5 in absolute value)
        assert abs(corr) > 0.5, f"Residual-U correlation too weak: {corr}"

    def test_treatment_effect_recovery(self):
        """Test that AutoITE recovers treatment effects better than random."""
        X, T, Y, Y_pre, tau_true, _ = generate_synthetic_data(n=2000)

        # Train/test split
        model = AutoITEEstimator(k=200)
        model.fit(X[:1500], T[:1500], Y[:1500], Y_pre[:1500])

        tau_pred = model.predict(X[1500:], Y_pre[1500:])
        tau_test = tau_true[1500:]

        # Correlation between predicted and true effects
        corr = np.corrcoef(tau_pred, tau_test)[0, 1]

        # Should have positive correlation
        assert corr > 0.3, f"Treatment effect correlation too weak: {corr}"

    def test_k_as_fraction(self):
        """Test that k can be specified as fraction of data."""
        X, T, Y, Y_pre, _, _ = generate_synthetic_data(n=1000)

        model = AutoITEEstimator(k=0.1)  # 10% of data
        model.fit(X, T, Y, Y_pre)

        assert model.k_actual_ == 100  # 10% of 1000

    def test_triage(self):
        """Test that triage correctly flags high-uncertainty cases."""
        X, T, Y, Y_pre, _, _ = generate_synthetic_data(n=500)

        model = AutoITEEstimator(k=50, triage_percentile=0.15)
        model.fit(X[:400], T[:400], Y[:400], Y_pre[:400])

        tau_pred = model.predict(X[400:], Y_pre[400:])

        # Should flag approximately 15% of cases
        triaged_fraction = model.triaged_mask_.mean()
        assert 0.10 <= triaged_fraction <= 0.20, f"Triage fraction unexpected: {triaged_fraction}"

    def test_score_method(self):
        """Test the score method returns expected metrics."""
        tau_true = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
        tau_pred = np.array([0.8, -0.8, 0.6, -0.4, 0.1])

        model = AutoITEEstimator()
        metrics = model.score(tau_true, tau_pred)

        assert 'mae' in metrics
        assert 'q99' in metrics
        assert 'max_error' in metrics
        assert 'deaths' in metrics
        assert metrics['mae'] > 0
        assert metrics['deaths'] >= 0


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 3))
        T = np.random.binomial(1, 0.5, n)
        Y = np.random.normal(0, 1, n)
        Y_pre = np.random.normal(0, 1, n)

        model = AutoITEEstimator(k=10)
        model.fit(X, T, Y, Y_pre)

        tau_pred = model.predict(X[:10], Y_pre[:10])
        assert len(tau_pred) == 10

    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, (n, 1))
        T = np.random.binomial(1, 0.5, n)
        Y = np.random.normal(0, 1, n)
        Y_pre = np.random.normal(0, 1, n)

        model = AutoITEEstimator(k=20)
        model.fit(X, T, Y, Y_pre)

        tau_pred = model.predict(X[:20], Y_pre[:20])
        assert len(tau_pred) == 20

    def test_all_treated(self):
        """Test behavior when all training units are treated."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        T = np.ones(n, dtype=int)  # All treated
        Y = np.random.normal(0, 1, n)
        Y_pre = np.random.normal(0, 1, n)

        model = AutoITEEstimator(k=20)
        model.fit(X, T, Y, Y_pre)

        # Should still run, though estimates may be poor
        tau_pred = model.predict(X[:10], Y_pre[:10])
        assert len(tau_pred) == 10

    def test_deterministic_with_seed(self):
        """Test that results are reproducible with same seed."""
        X, T, Y, Y_pre, _, _ = generate_synthetic_data(n=500, seed=123)

        model1 = AutoITEEstimator(k=50, random_state=42)
        model1.fit(X[:400], T[:400], Y[:400], Y_pre[:400])
        tau1 = model1.predict(X[400:], Y_pre[400:])

        model2 = AutoITEEstimator(k=50, random_state=42)
        model2.fit(X[:400], T[:400], Y[:400], Y_pre[:400])
        tau2 = model2.predict(X[400:], Y_pre[400:])

        np.testing.assert_array_almost_equal(tau1, tau2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

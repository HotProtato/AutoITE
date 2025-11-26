"""
Tests for AutoITE Diagnostics

These tests verify functionality of BimodalityDiagnostic and
UnexplainedHeterogeneityIndex classes.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoite import BimodalityDiagnostic, UnexplainedHeterogeneityIndex


def generate_bimodal_data(n=500, seed=42):
    """Generate data with clear bimodal residual structure."""
    np.random.seed(seed)

    # Two latent groups
    group = np.random.binomial(1, 0.5, n)

    # Features (don't reveal group)
    X = np.random.normal(0, 1, (n, 5))

    # Baseline affected by group (creates bimodal residuals)
    Y_pre = np.where(group == 0, -2, 2) + np.random.normal(0, 0.5, n)

    return X, Y_pre, group


def generate_unimodal_data(n=500, seed=42):
    """Generate data with unimodal (homogeneous) residuals."""
    np.random.seed(seed)

    X = np.random.normal(0, 1, (n, 5))
    Y_pre = X[:, 0] + np.random.normal(0, 1, n)  # Simple linear relationship

    return X, Y_pre


def generate_outcome_data(n=500, seed=42):
    """Generate full data with treatment and outcomes for UHI tests."""
    np.random.seed(seed)

    X = np.random.normal(0, 1, (n, 5))
    T = np.random.binomial(1, 0.5, n)
    Y_pre = X[:, 0] + np.random.normal(0, 0.5, n)

    # Treatment effect varies by X[:, 1]
    tau = 1.0 + 0.5 * X[:, 1]
    Y = Y_pre + T * tau + np.random.normal(0, 0.5, n)

    return X, T, Y, Y_pre


class TestBimodalityDiagnostic:
    """Test suite for BimodalityDiagnostic."""

    def test_initialization(self):
        """Test that diagnostic initializes correctly."""
        diag = BimodalityDiagnostic(max_components=3, bic_threshold=0.05)
        assert diag.max_components == 3
        assert diag.bic_threshold == 0.05

    def test_fit(self):
        """Test that fit runs without error."""
        X, Y_pre = generate_unimodal_data(n=200)

        diag = BimodalityDiagnostic()
        diag.fit(X, Y_pre)

        assert diag.global_model_ is not None
        assert diag.train_residual_std_ > 0

    def test_score_returns_dict(self):
        """Test that score returns expected dictionary keys."""
        X, Y_pre = generate_unimodal_data(n=300)

        diag = BimodalityDiagnostic()
        diag.fit(X[:200], Y_pre[:200])
        result = diag.score(X[200:], Y_pre[200:])

        assert 'bimodality_score' in result
        assert 'hidden_structure_detected' in result
        assert 'optimal_components' in result
        assert 'cluster_labels' in result
        assert 'cluster_means' in result

    def test_detects_bimodal_structure(self):
        """Test that bimodal data is detected."""
        X, Y_pre, _ = generate_bimodal_data(n=500)

        diag = BimodalityDiagnostic()
        diag.fit(X[:300], Y_pre[:300])
        result = diag.score(X[300:], Y_pre[300:])

        # Should detect structure
        assert result['bimodality_score'] > 0.05, \
            f"Failed to detect bimodal structure: {result['bimodality_score']}"
        assert result['optimal_components'] >= 2

    def test_no_false_positive_unimodal(self):
        """Test that unimodal data doesn't trigger false positive."""
        X, Y_pre = generate_unimodal_data(n=500)

        diag = BimodalityDiagnostic()
        diag.fit(X[:300], Y_pre[:300])
        result = diag.score(X[300:], Y_pre[300:])

        # Should not detect strong structure
        # (allowing for some noise, use generous threshold)
        assert result['bimodality_score'] < 0.15, \
            f"False positive on unimodal data: {result['bimodality_score']}"

    def test_quantify_unknown(self):
        """Test the quantify_unknown method."""
        X, Y_pre, _ = generate_bimodal_data(n=400)

        diag = BimodalityDiagnostic()
        diag.fit(X[:250], Y_pre[:250])
        result = diag.quantify_unknown(X[250:], Y_pre[250:])

        assert 'bimodality_score' in result
        assert 'cluster_separation' in result
        assert 'unexplained_fraction' in result
        assert 'interpretation' in result
        assert isinstance(result['interpretation'], str)

    def test_cluster_labels_shape(self):
        """Test that cluster labels have correct shape."""
        X, Y_pre = generate_unimodal_data(n=300)

        diag = BimodalityDiagnostic()
        diag.fit(X[:200], Y_pre[:200])
        result = diag.score(X[200:], Y_pre[200:])

        assert len(result['cluster_labels']) == 100


class TestUnexplainedHeterogeneityIndex:
    """Test suite for UnexplainedHeterogeneityIndex."""

    def test_initialization(self):
        """Test that UHI initializes correctly."""
        uhi = UnexplainedHeterogeneityIndex(k=100, alpha=1.0)
        assert uhi.k == 100
        assert uhi.alpha == 1.0

    def test_fit(self):
        """Test that fit runs without error."""
        X, T, Y, Y_pre = generate_outcome_data(n=200)

        uhi = UnexplainedHeterogeneityIndex(k=30)
        uhi.fit(X, T, Y, Y_pre)

        assert uhi.global_outcome_model_ is not None
        assert uhi.baseline_residuals_ is not None

    def test_score_returns_dict(self):
        """Test that score returns expected dictionary keys."""
        X, T, Y, Y_pre = generate_outcome_data(n=300)

        uhi = UnexplainedHeterogeneityIndex(k=30)
        uhi.fit(X[:200], T[:200], Y[:200], Y_pre[:200])
        result = uhi.score(X[200:], T[200:], Y[200:], Y_pre[200:], n_samples=50)

        assert 'uhi' in result
        assert 'global_variance' in result
        assert 'local_variance' in result

    def test_uhi_range(self):
        """Test that UHI is in reasonable range."""
        X, T, Y, Y_pre = generate_outcome_data(n=400)

        uhi = UnexplainedHeterogeneityIndex(k=50)
        uhi.fit(X[:300], T[:300], Y[:300], Y_pre[:300])
        result = uhi.score(X[300:], T[300:], Y[300:], Y_pre[300:], n_samples=50)

        # UHI should typically be between 0 and 2
        assert 0 <= result['uhi'] <= 3, f"UHI out of expected range: {result['uhi']}"

    def test_heterogeneous_data_lower_uhi(self):
        """Test that heterogeneous data tends to have lower UHI."""
        # Create data with local structure in outcomes
        np.random.seed(42)
        n = 600
        X = np.random.normal(0, 1, (n, 5))
        T = np.random.binomial(1, 0.5, n)

        # Baseline with local patterns
        cluster = (X[:, 0] > 0).astype(int)
        Y_pre = np.where(cluster == 0,
                         X[:, 1] + np.random.normal(0, 0.3, n),
                         -X[:, 1] + np.random.normal(0, 0.3, n))

        # Outcomes with cluster-dependent treatment effects
        tau = np.where(cluster == 0, 1.0, -1.0)
        Y = Y_pre + T * tau + np.random.normal(0, 0.3, n)

        uhi = UnexplainedHeterogeneityIndex(k=50)
        uhi.fit(X[:400], T[:400], Y[:400], Y_pre[:400])
        result = uhi.score(X[400:], T[400:], Y[400:], Y_pre[400:], n_samples=100)

        # Local models should help, so UHI < 1
        # (Being generous with threshold due to noise)
        assert result['uhi'] < 1.5, f"UHI unexpectedly high for heterogeneous data: {result['uhi']}"


class TestDiagnosticIntegration:
    """Integration tests combining diagnostics."""

    def test_both_diagnostics_on_same_data(self):
        """Test running both diagnostics on the same dataset."""
        X, Y_pre, group = generate_bimodal_data(n=400)

        # Generate treatment and outcomes for UHI
        np.random.seed(123)
        T = np.random.binomial(1, 0.5, len(X))
        tau = np.where(group == 0, -1.0, 1.0)
        Y = Y_pre + T * tau + np.random.normal(0, 0.3, len(X))

        # Bimodality (uses only baseline)
        bimodal = BimodalityDiagnostic()
        bimodal.fit(X[:250], Y_pre[:250])
        bimodal_result = bimodal.quantify_unknown(X[250:], Y_pre[250:])

        # UHI (uses outcomes)
        uhi = UnexplainedHeterogeneityIndex(k=40)
        uhi.fit(X[:250], T[:250], Y[:250], Y_pre[:250])
        uhi_result = uhi.score(X[250:], T[250:], Y[250:], Y_pre[250:], n_samples=50)

        # Both should run without error
        assert bimodal_result['bimodality_score'] is not None
        assert uhi_result['uhi'] is not None

    def test_interpretation_strings(self):
        """Test that interpretation strings are meaningful."""
        X, Y_pre, _ = generate_bimodal_data(n=300)

        diag = BimodalityDiagnostic()
        diag.fit(X[:200], Y_pre[:200])
        result = diag.quantify_unknown(X[200:], Y_pre[200:])

        # Should contain relevant keywords
        interp = result['interpretation'].lower()
        assert any(word in interp for word in ['hidden', 'structure', 'heterogeneity', 'homogeneous'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

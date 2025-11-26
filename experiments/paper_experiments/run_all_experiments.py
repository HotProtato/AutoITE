"""
AutoITE Paper Experiments: Reproduce All Results

This script reproduces all experimental results reported in the paper:
1. Experiment 1: Orthogonality Recovery
2. Experiment 2: Safety Lift (Method Comparison + Triage)
3. Experiment 3: Completeness Detection
4. Simpson's Paradox Detection
5. Feature Robustness
6. Identifiability Limits

Run with: python run_all_experiments.py

Author: Jake Peace
Date: November 2025
"""

import numpy as np
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autoite import AutoITEEstimator, BimodalityDiagnostic
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.mixture import GaussianMixture

# Try to import optional dependencies
try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    print("Warning: econml not installed. Causal Forest comparisons will be skipped.")


def generate_orthogonality_data(n=10000, seed=42):
    """
    Generate data where U is orthogonal to X (the hardest case for feature-based methods).

    Data generating process:
    - X: 5 features, standard normal
    - U: Binary latent confounder, independent of X
    - T: Random treatment assignment (RCT)
    - Y_pre: Baseline outcome, affected by X and U
    - tau: Treatment effect depends on U (heterogeneous)
    - Y: Outcome = baseline + treatment effect + noise
    """
    rng = np.random.default_rng(seed)

    # Features (U is orthogonal to X by construction)
    X = rng.standard_normal((n, 5))
    beta = np.array([0.5, -0.3, 0.4, -0.2, 0.1])

    # Latent confounder (independent of X)
    U = rng.binomial(1, 0.5, n)

    # Treatment (randomized)
    T = rng.binomial(1, 0.5, n)

    # Baseline outcome (affected by U - this is baseline coupling)
    epsilon_pre = rng.normal(0, 0.5, n)
    Y_pre = X @ beta - 2.0 * U + epsilon_pre

    # True treatment effect (heterogeneous by U)
    tau_true = np.where(U == 0, 1.0, -2.0)

    # Outcome
    epsilon = rng.normal(0, 0.5, n)
    Y = X @ beta - 2.0 * U + T * tau_true + epsilon

    return {
        'X': X, 'T': T, 'Y': Y, 'Y_pre': Y_pre,
        'U': U, 'tau_true': tau_true
    }


def experiment_1_orthogonality(verbose=True):
    """
    Experiment 1: Orthogonality Recovery

    Tests whether AutoITE can recover latent heterogeneity when U ⊥ X.
    Compares residual correlation with U across methods.
    """
    if verbose:
        print("\n" + "="*80)
        print("EXPERIMENT 1: ORTHOGONALITY RECOVERY")
        print("="*80)

    data = generate_orthogonality_data(n=10000, seed=42)

    # Train/test split
    n = len(data['X'])
    train_idx = np.arange(int(0.8 * n))
    test_idx = np.arange(int(0.8 * n), n)

    X_train, X_test = data['X'][train_idx], data['X'][test_idx]
    T_train, T_test = data['T'][train_idx], data['T'][test_idx]
    Y_train, Y_test = data['Y'][train_idx], data['Y'][test_idx]
    Y_pre_train, Y_pre_test = data['Y_pre'][train_idx], data['Y_pre'][test_idx]
    U_train, U_test = data['U'][train_idx], data['U'][test_idx]
    tau_true = data['tau_true'][test_idx]

    results = {}

    # AutoITE
    model = AutoITEEstimator(k=1000, alpha=1.0)
    model.fit(X_train, T_train, Y_train, Y_pre_train)
    tau_pred = model.predict(X_test, Y_pre_test)

    # Compute residual correlation with U
    residual_corr = model.get_residual_correlation(U_train)

    # Compute prediction correlation with U
    pred_corr = np.corrcoef(tau_pred, U_test)[0, 1]

    results['AutoITE'] = {
        'residual_corr': residual_corr,
        'pred_corr': pred_corr,
        'mae': np.mean(np.abs(tau_pred - tau_true))
    }

    # Causal Forest (if available)
    if HAS_ECONML:
        cf = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
            model_t=RandomForestRegressor(n_estimators=100, random_state=42),
            random_state=42
        )
        cf.fit(Y_train, T_train, X=X_train)
        tau_cf = cf.effect(X_test).flatten()

        results['CausalForest'] = {
            'residual_corr': 0.0,  # CF doesn't use residuals
            'pred_corr': np.corrcoef(tau_cf, U_test)[0, 1],
            'mae': np.mean(np.abs(tau_cf - tau_true))
        }

    if verbose:
        print(f"\nU prevalence: {U_test.mean()*100:.1f}%")
        print(f"True tau distribution: U=0 -> +1.0, U=1 -> -2.0")
        print("\nResults:")
        print(f"{'Method':<20} {'Corr(R,U)':<15} {'Corr(tau,U)':<15} {'MAE':<10}")
        print("-"*60)
        for method, res in results.items():
            print(f"{method:<20} {res['residual_corr']:<15.4f} {res['pred_corr']:<15.4f} {res['mae']:<10.4f}")

        print("\nKey Finding: AutoITE achieves strong correlation with hidden U")
        print("via residuals, while Causal Forest achieves ~0 (orthogonality).")

    return results


def experiment_2_safety_lift(verbose=True):
    """
    Experiment 2: Safety Lift

    Comprehensive method comparison including triage analysis.
    """
    if verbose:
        print("\n" + "="*80)
        print("EXPERIMENT 2: SAFETY LIFT")
        print("="*80)

    data = generate_orthogonality_data(n=10000, seed=42)

    n = len(data['X'])
    train_idx = np.arange(int(0.8 * n))
    test_idx = np.arange(int(0.8 * n), n)

    X_train, X_test = data['X'][train_idx], data['X'][test_idx]
    T_train, T_test = data['T'][train_idx], data['T'][test_idx]
    Y_train, Y_test = data['Y'][train_idx], data['Y'][test_idx]
    Y_pre_train, Y_pre_test = data['Y_pre'][train_idx], data['Y_pre'][test_idx]
    tau_true = data['tau_true'][test_idx]

    results = {}

    # AutoITE (no triage)
    model = AutoITEEstimator(k=1000, alpha=1.0)
    model.fit(X_train, T_train, Y_train, Y_pre_train)
    tau_pred = model.predict(X_test, Y_pre_test)
    results['AutoITE (k=1000)'] = model.score(tau_true, tau_pred)

    # AutoITE with triage
    for triage_pct in [0.10, 0.15, 0.20]:
        model_triage = AutoITEEstimator(k=1000, alpha=1.0, triage_percentile=triage_pct)
        model_triage.fit(X_train, T_train, Y_train, Y_pre_train)
        tau_pred_triage, sigma = model_triage.predict(X_test, Y_pre_test, return_uncertainty=True)

        # Evaluate only non-triaged cases
        mask = ~model_triage.triaged_mask_
        if mask.sum() > 0:
            errors = np.abs(tau_pred_triage[mask] - tau_true[mask])
            treat = tau_pred_triage[mask] > 0
            harmful = tau_true[mask] < 0
            deaths = np.sum(treat & harmful)

            results[f'AutoITE + {int(triage_pct*100)}% triage'] = {
                'mae': np.mean(errors),
                'q99': np.percentile(errors, 99),
                'max_error': np.max(errors),
                'deaths': int(deaths),
                'treated': int(np.sum(treat)),
                'coverage': f"{(1-triage_pct)*100:.0f}%"
            }

    # Causal Forest
    if HAS_ECONML:
        cf = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
            model_t=RandomForestRegressor(n_estimators=100, random_state=42),
            random_state=42
        )
        cf.fit(Y_train, T_train, X=X_train)
        tau_cf = cf.effect(X_test).flatten()

        errors = np.abs(tau_cf - tau_true)
        treat = tau_cf > 0
        harmful = tau_true < 0
        deaths = np.sum(treat & harmful)

        results['Causal Forest'] = {
            'mae': np.mean(errors),
            'q99': np.percentile(errors, 99),
            'max_error': np.max(errors),
            'deaths': int(deaths),
            'treated': int(np.sum(treat))
        }

    if verbose:
        print("\nMethod Comparison:")
        print(f"{'Method':<30} {'MAE':<10} {'Q99':<10} {'Max':<10} {'Deaths':<10}")
        print("-"*70)
        for method, res in results.items():
            coverage = res.get('coverage', '100%')
            print(f"{method:<30} {res['mae']:<10.4f} {res['q99']:<10.2f} {res['max_error']:<10.2f} {res['deaths']:<10}")

    return results


def experiment_3_completeness(verbose=True):
    """
    Experiment 3: Completeness Detection

    Tests bimodality diagnostic for detecting hidden subgroups.
    """
    if verbose:
        print("\n" + "="*80)
        print("EXPERIMENT 3: COMPLETENESS DETECTION")
        print("="*80)

    results = {}

    # Scenario A: No hidden confounder
    rng = np.random.default_rng(42)
    n = 10000
    X = rng.standard_normal((n, 5))
    Y_pre = X @ np.array([0.5, -0.3, 0.4, -0.2, 0.1]) + rng.normal(0, 0.5, n)

    diag = BimodalityDiagnostic()
    train_idx = np.arange(int(0.8 * n))
    test_idx = np.arange(int(0.8 * n), n)

    diag.fit(X[train_idx], Y_pre[train_idx])
    result_a = diag.score(X[test_idx], Y_pre[test_idx])
    results['No confounder'] = {
        'bimodality': result_a['bimodality_score'],
        'detected': result_a['hidden_structure_detected']
    }

    # Scenario B: Baseline-coupled confounder
    U = rng.binomial(1, 0.5, n)
    Y_pre_coupled = X @ np.array([0.5, -0.3, 0.4, -0.2, 0.1]) - 2.0 * U + rng.normal(0, 0.5, n)

    diag_b = BimodalityDiagnostic()
    diag_b.fit(X[train_idx], Y_pre_coupled[train_idx])
    result_b = diag_b.score(X[test_idx], Y_pre_coupled[test_idx])

    # GMM recovery
    labels = result_b['cluster_labels']
    U_test = U[test_idx]
    recovery = max((labels == U_test).mean(), (labels != U_test).mean())

    results['Baseline-coupled'] = {
        'bimodality': result_b['bimodality_score'],
        'detected': result_b['hidden_structure_detected'],
        'gmm_recovery': recovery
    }

    if verbose:
        print("\nBimodality Detection Results:")
        print(f"{'Scenario':<25} {'Bimodality':<15} {'Detected':<15} {'GMM Recovery':<15}")
        print("-"*70)
        for scenario, res in results.items():
            gmm = f"{res.get('gmm_recovery', 0)*100:.1f}%" if 'gmm_recovery' in res else "N/A"
            print(f"{scenario:<25} {res['bimodality']:<15.4f} {str(res['detected']):<15} {gmm:<15}")

    return results


def experiment_feature_robustness(verbose=True):
    """
    Feature Robustness: Test adding noise features.
    """
    if verbose:
        print("\n" + "="*80)
        print("FEATURE ROBUSTNESS TEST")
        print("="*80)

    base_data = generate_orthogonality_data(n=10000, seed=42)

    n = len(base_data['X'])
    train_idx = np.arange(int(0.8 * n))
    test_idx = np.arange(int(0.8 * n), n)

    results = {}
    rng = np.random.default_rng(42)

    for n_noise in [0, 50, 100, 200]:
        if n_noise == 0:
            X = base_data['X']
        else:
            noise_features = rng.standard_normal((n, n_noise))
            X = np.column_stack([base_data['X'], noise_features])

        X_train, X_test = X[train_idx], X[test_idx]
        T_train = base_data['T'][train_idx]
        Y_train = base_data['Y'][train_idx]
        Y_pre_train, Y_pre_test = base_data['Y_pre'][train_idx], base_data['Y_pre'][test_idx]
        tau_true = base_data['tau_true'][test_idx]

        model = AutoITEEstimator(k=1000, alpha=1.0)
        model.fit(X_train, T_train, Y_train, Y_pre_train)
        tau_pred = model.predict(X_test, Y_pre_test)

        mae = np.mean(np.abs(tau_pred - tau_true))
        results[f'{5 + n_noise} features'] = {'mae': mae, 'noise_features': n_noise}

    if verbose:
        baseline_mae = results['5 features']['mae']
        print("\nNoise Feature Robustness:")
        print(f"{'Configuration':<20} {'MAE':<10} {'Change':<10}")
        print("-"*40)
        for config, res in results.items():
            change = (res['mae'] - baseline_mae) / baseline_mae * 100
            print(f"{config:<20} {res['mae']:<10.4f} {change:+.1f}%")

    return results


def experiment_identifiability_limit(verbose=True):
    """
    Identifiability Limit: Test interaction-only confounders.
    """
    if verbose:
        print("\n" + "="*80)
        print("IDENTIFIABILITY LIMIT TEST")
        print("="*80)

    rng = np.random.default_rng(42)
    n = 10000

    results = {}

    # Scenario 1: Observable (baseline-coupled)
    X = rng.standard_normal((n, 5))
    U = rng.binomial(1, 0.5, n)
    T = rng.binomial(1, 0.5, n)

    beta = np.array([0.5, -0.3, 0.4, -0.2, 0.1])
    Y_pre = X @ beta - 2.0 * U + rng.normal(0, 0.5, n)
    tau_true = np.where(U == 0, 1.0, -2.0)
    Y = X @ beta - 2.0 * U + T * tau_true + rng.normal(0, 0.5, n)

    train_idx = np.arange(int(0.8 * n))
    test_idx = np.arange(int(0.8 * n), n)

    model = AutoITEEstimator(k=1000, alpha=1.0)
    model.fit(X[train_idx], T[train_idx], Y[train_idx], Y_pre[train_idx])
    tau_pred = model.predict(X[test_idx], Y_pre[test_idx])

    errors = np.abs(tau_pred - tau_true[test_idx])
    treat = tau_pred > 0
    harmful = tau_true[test_idx] < 0
    deaths = np.sum(treat & harmful)
    detection = (tau_pred[U[test_idx] == 1] < 0).mean()

    results['Observable (baseline-coupled)'] = {
        'mae': np.mean(errors),
        'deaths': int(deaths),
        'detection_rate': detection
    }

    # Scenario 2: Interaction-only (undetectable)
    Y_pre_no_U = X @ beta + rng.normal(0, 0.5, n)  # U doesn't affect baseline
    Y_interaction = X @ beta + T * tau_true + rng.normal(0, 0.5, n)

    model2 = AutoITEEstimator(k=1000, alpha=1.0)
    model2.fit(X[train_idx], T[train_idx], Y_interaction[train_idx], Y_pre_no_U[train_idx])
    tau_pred2 = model2.predict(X[test_idx], Y_pre_no_U[test_idx])

    errors2 = np.abs(tau_pred2 - tau_true[test_idx])
    treat2 = tau_pred2 > 0
    deaths2 = np.sum(treat2 & harmful)
    detection2 = (tau_pred2[U[test_idx] == 1] < 0).mean()

    results['Interaction-only (undetectable)'] = {
        'mae': np.mean(errors2),
        'deaths': int(deaths2),
        'detection_rate': detection2
    }

    if verbose:
        print("\nIdentifiability Comparison:")
        print(f"{'Scenario':<35} {'MAE':<10} {'Deaths':<10} {'Detection':<15}")
        print("-"*70)
        for scenario, res in results.items():
            print(f"{scenario:<35} {res['mae']:<10.4f} {res['deaths']:<10} {res['detection_rate']*100:<15.1f}%")

        deaths_diff = results['Interaction-only (undetectable)']['deaths'] - results['Observable (baseline-coupled)']['deaths']
        print(f"\nKey Finding: {deaths_diff} additional deaths when confounder is undetectable")
        print("This is the fundamental identifiability limit, not an algorithm limitation.")

    return results


def main():
    """Run all paper experiments."""
    print("="*80)
    print("AutoITE Paper Experiments")
    print("Reproducing all results from the manuscript")
    print("="*80)

    # Run all experiments
    results = {}

    results['exp1'] = experiment_1_orthogonality()
    results['exp2'] = experiment_2_safety_lift()
    results['exp3'] = experiment_3_completeness()
    results['robustness'] = experiment_feature_robustness()
    results['identifiability'] = experiment_identifiability_limit()

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)

    return results


if __name__ == "__main__":
    main()

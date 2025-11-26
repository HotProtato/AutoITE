"""
Real-World Experiment: UCI Student Intervention Dataset

This experiment applies AutoITE to the UCI Student Performance dataset
to demonstrate effectiveness on real-world data.

Dataset: UCI Student Performance Data
- Source: https://archive.ics.uci.edu/dataset/320/student+performance
- Domain: Education
- Treatment: School support intervention
- Outcome: Final grade (G3)
- Baseline: First period grade (G1) or second period grade (G2)

The key question: Can AutoITE identify students who would benefit from
intervention vs those who would not, using baseline academic performance
as the proxy for latent learning ability?

Dataset Attribution (CC BY 4.0 License):
    Creator: Paulo Cortez
    DOI: 10.24432/C5TG7T
    Citation: P. Cortez and A. Silva. Using Data Mining to Predict Secondary
              School Student Performance. In A. Brito and J. Teixeira Eds.,
              Proceedings of 5th FUture BUsiness TEChnology Conference
              (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS,
              ISBN 978-9077381-39-7.

Author: Jake Peace
Date: November 2025
"""

import numpy as np
import pandas as pd
import sys
import os
from urllib.request import urlretrieve
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autoite import AutoITEEstimator, BimodalityDiagnostic

# Causal Forest for comparison
try:
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    print("Warning: econml not installed. Causal Forest comparison will be skipped.")


def download_student_data():
    """Download UCI Student Performance dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "student.zip")
    mat_path = os.path.join(data_dir, "student-mat.csv")

    if not os.path.exists(mat_path):
        print("Downloading UCI Student Performance dataset...")
        try:
            urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
            print("Download complete.")
        except Exception as e:
            print(f"Could not download dataset: {e}")
            print("Please download manually from:")
            print("https://archive.ics.uci.edu/ml/datasets/Student+Performance")
            return None

    return mat_path


def load_and_prepare_data(filepath):
    """
    Load and prepare student data for causal analysis.

    Treatment construction:
    - We use 'schoolsup' (extra educational support) as treatment
    - This is a natural intervention that varies across students

    Outcome:
    - G3: Final grade (0-20)

    Baseline:
    - G1: First period grade (proxy for latent ability)

    Features:
    - Demographics, family, and school-related variables
    """
    df = pd.read_csv(filepath, sep=';')

    print(f"Dataset size: {len(df)} students")
    print(f"Treatment (schoolsup) distribution: {df['schoolsup'].value_counts().to_dict()}")

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=[
        'school', 'sex', 'address', 'famsize', 'Pstatus',
        'Mjob', 'Fjob', 'reason', 'guardian', 'famsup',
        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'
    ], drop_first=True)

    # Treatment: school support (1 = yes, 0 = no)
    T = (df['schoolsup'] == 'yes').astype(int).values

    # Outcome: Final grade
    Y = df['G3'].values.astype(float)

    # Baseline: First period grade (proxy for latent ability)
    Y_pre = df['G1'].values.astype(float)

    # Features: All other variables (excluding G1, G2, G3, schoolsup)
    exclude_cols = ['G1', 'G2', 'G3', 'schoolsup_yes', 'schoolsup_no']
    feature_cols = [c for c in df_encoded.columns if c not in exclude_cols and 'schoolsup' not in c]
    X = df_encoded[feature_cols].values.astype(float)

    print(f"Features: {X.shape[1]}")
    print(f"Treatment rate: {T.mean()*100:.1f}%")
    print(f"Outcome (G3) mean: {Y.mean():.2f}, std: {Y.std():.2f}")
    print(f"Baseline (G1) mean: {Y_pre.mean():.2f}, std: {Y_pre.std():.2f}")

    return X, T, Y, Y_pre, df


def run_student_experiment(X, T, Y, Y_pre, k_fraction=0.10):
    """
    Run AutoITE analysis on student data with Causal Forest comparison.

    Since we don't have true treatment effects, we:
    1. Estimate effects using AutoITE and Causal Forest
    2. Analyze heterogeneity patterns
    3. Check for hidden subgroups via bimodality
    4. Compare subgroup outcomes between methods
    """
    n = len(Y)
    k = int(k_fraction * n)

    print(f"\nUsing k = {k} ({k_fraction*100:.0f}% of n={n})")

    # Train/test split (stratified by treatment)
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    T_train, T_test = T[train_idx], T[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    Y_pre_train, Y_pre_test = Y_pre[train_idx], Y_pre[test_idx]

    print(f"\nTrain size: {len(train_idx)}, Test size: {len(test_idx)}")
    print(f"Train treatment rate: {T_train.mean()*100:.1f}%")
    print(f"Test treatment rate: {T_test.mean()*100:.1f}%")

    # =====================================================================
    # FIT AUTOITE
    # =====================================================================
    print("\nFitting AutoITE...")
    model = AutoITEEstimator(k=k, alpha=1.0)
    model.fit(X_train, T_train, Y_train, Y_pre_train)

    # Predict treatment effects
    tau_pred, sigma = model.predict(X_test, Y_pre_test, return_uncertainty=True)

    # =====================================================================
    # FIT CAUSAL FOREST (for comparison)
    # =====================================================================
    tau_cf = None
    if HAS_ECONML:
        print("Fitting Causal Forest...")
        # Include baseline as feature for fair comparison
        X_train_full = np.column_stack([X_train, Y_pre_train])
        X_test_full = np.column_stack([X_test, Y_pre_test])

        cf = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3),  # Use regressor for binary T
            discrete_treatment=True,
            n_estimators=200,
            random_state=42
        )
        cf.fit(Y_train, T_train, X=X_train_full)
        tau_cf = cf.effect(X_test_full).flatten()

    print("\n" + "="*60)
    print("AUTOITE RESULTS")
    print("="*60)

    print(f"\nPredicted Treatment Effect Distribution:")
    print(f"  Mean: {tau_pred.mean():.3f}")
    print(f"  Std:  {tau_pred.std():.3f}")
    print(f"  Min:  {tau_pred.min():.3f}")
    print(f"  Max:  {tau_pred.max():.3f}")

    # Heterogeneity analysis
    positive_effect = tau_pred > 0
    print(f"\nHeterogeneity Analysis:")
    print(f"  Students predicted to benefit: {positive_effect.sum()} ({positive_effect.mean()*100:.1f}%)")
    print(f"  Students predicted to be harmed: {(~positive_effect).sum()} ({(~positive_effect).mean()*100:.1f}%)")

    # Effect by baseline performance
    low_baseline = Y_pre_test < np.median(Y_pre_test)
    high_baseline = ~low_baseline

    print(f"\nEffect by Baseline Performance:")
    print(f"  Low baseline (G1 < median): mean effect = {tau_pred[low_baseline].mean():.3f}")
    print(f"  High baseline (G1 >= median): mean effect = {tau_pred[high_baseline].mean():.3f}")

    # Bimodality diagnostic
    print("\n" + "="*60)
    print("BIMODALITY DIAGNOSTIC")
    print("="*60)

    bimodal = BimodalityDiagnostic()
    bimodal.fit(X_train, Y_pre_train)
    bimodal_result = bimodal.quantify_unknown(X_test, Y_pre_test)

    print(f"\nBimodality Score: {bimodal_result['bimodality_score']:.4f}")
    print(f"Cluster Separation: {bimodal_result['cluster_separation']:.4f}")
    print(f"Interpretation: {bimodal_result['interpretation']}")

    # Compare treated vs control outcomes
    print("\n" + "="*60)
    print("OBSERVED OUTCOMES ANALYSIS")
    print("="*60)

    treated_test = T_test == 1
    control_test = T_test == 0

    if treated_test.sum() > 0 and control_test.sum() > 0:
        print(f"\nObserved Outcomes (Test Set):")
        print(f"  Treated: mean G3 = {Y_test[treated_test].mean():.2f} (n={treated_test.sum()})")
        print(f"  Control: mean G3 = {Y_test[control_test].mean():.2f} (n={control_test.sum()})")
        print(f"  Naive ATE: {Y_test[treated_test].mean() - Y_test[control_test].mean():.3f}")

        # AutoITE subgroup analysis
        print(f"\nAutoITE Subgroup Analysis:")

        # Among predicted benefiters
        benefiters = positive_effect
        if (benefiters & treated_test).sum() > 0 and (benefiters & control_test).sum() > 0:
            treated_benefit = Y_test[benefiters & treated_test].mean()
            control_benefit = Y_test[benefiters & control_test].mean()
            print(f"  Predicted benefiters (n={benefiters.sum()}):")
            print(f"    Treated: {treated_benefit:.2f}, Control: {control_benefit:.2f}")
            print(f"    Observed effect: {treated_benefit - control_benefit:.3f}")

        # Among predicted non-benefiters
        non_benefiters = ~positive_effect
        if (non_benefiters & treated_test).sum() > 0 and (non_benefiters & control_test).sum() > 0:
            treated_nonbenefit = Y_test[non_benefiters & treated_test].mean()
            control_nonbenefit = Y_test[non_benefiters & control_test].mean()
            print(f"  Predicted non-benefiters (n={non_benefiters.sum()}):")
            print(f"    Treated: {treated_nonbenefit:.2f}, Control: {control_nonbenefit:.2f}")
            print(f"    Observed effect: {treated_nonbenefit - control_nonbenefit:.3f}")

    # Uncertainty analysis
    print("\n" + "="*60)
    print("UNCERTAINTY ANALYSIS")
    print("="*60)

    print(f"\nLocal Model Uncertainty (sigma):")
    print(f"  Mean: {sigma.mean():.4f}")
    print(f"  Std:  {sigma.std():.4f}")

    # High uncertainty cases
    high_uncertainty = sigma > np.percentile(sigma, 85)
    print(f"\nHigh-uncertainty cases (top 15%): {high_uncertainty.sum()}")
    print(f"  These would be triaged for expert review")

    # =====================================================================
    # CAUSAL FOREST COMPARISON
    # =====================================================================
    if tau_cf is not None:
        print("\n" + "="*60)
        print("CAUSAL FOREST COMPARISON")
        print("="*60)

        print(f"\nCausal Forest Predicted Effects:")
        print(f"  Mean: {tau_cf.mean():.3f}")
        print(f"  Std:  {tau_cf.std():.3f}")
        print(f"  Min:  {tau_cf.min():.3f}")
        print(f"  Max:  {tau_cf.max():.3f}")

        cf_positive = tau_cf > 0
        print(f"\nCausal Forest Heterogeneity:")
        print(f"  Students predicted to benefit: {cf_positive.sum()} ({cf_positive.mean()*100:.1f}%)")
        print(f"  Students predicted to be harmed: {(~cf_positive).sum()} ({(~cf_positive).mean()*100:.1f}%)")

        # Subgroup validation for Causal Forest
        print(f"\nCausal Forest Subgroup Analysis:")
        cf_benefiters = cf_positive
        if (cf_benefiters & treated_test).sum() > 0 and (cf_benefiters & control_test).sum() > 0:
            cf_treated_benefit = Y_test[cf_benefiters & treated_test].mean()
            cf_control_benefit = Y_test[cf_benefiters & control_test].mean()
            print(f"  CF Predicted benefiters (n={cf_benefiters.sum()}):")
            print(f"    Treated: {cf_treated_benefit:.2f}, Control: {cf_control_benefit:.2f}")
            print(f"    Observed effect: {cf_treated_benefit - cf_control_benefit:.3f}")

        cf_non_benefiters = ~cf_positive
        if (cf_non_benefiters & treated_test).sum() > 0 and (cf_non_benefiters & control_test).sum() > 0:
            cf_treated_nonbenefit = Y_test[cf_non_benefiters & treated_test].mean()
            cf_control_nonbenefit = Y_test[cf_non_benefiters & control_test].mean()
            print(f"  CF Predicted non-benefiters (n={cf_non_benefiters.sum()}):")
            print(f"    Treated: {cf_treated_nonbenefit:.2f}, Control: {cf_control_nonbenefit:.2f}")
            print(f"    Observed effect: {cf_treated_nonbenefit - cf_control_nonbenefit:.3f}")

        # =====================================================================
        # HEAD-TO-HEAD COMPARISON
        # =====================================================================
        print("\n" + "="*60)
        print("HEAD-TO-HEAD COMPARISON: AutoITE vs Causal Forest")
        print("="*60)

        # Subgroup separation metric: difference in observed effects
        autoite_benefiters = positive_effect
        autoite_separation = None
        if (autoite_benefiters & treated_test).sum() > 0 and (autoite_benefiters & control_test).sum() > 0:
            if ((~autoite_benefiters) & treated_test).sum() > 0 and ((~autoite_benefiters) & control_test).sum() > 0:
                auto_benefit_effect = Y_test[autoite_benefiters & treated_test].mean() - Y_test[autoite_benefiters & control_test].mean()
                auto_harm_effect = Y_test[(~autoite_benefiters) & treated_test].mean() - Y_test[(~autoite_benefiters) & control_test].mean()
                autoite_separation = auto_benefit_effect - auto_harm_effect

        cf_separation = None
        if (cf_benefiters & treated_test).sum() > 0 and (cf_benefiters & control_test).sum() > 0:
            if ((~cf_benefiters) & treated_test).sum() > 0 and ((~cf_benefiters) & control_test).sum() > 0:
                cf_benefit_effect = Y_test[cf_benefiters & treated_test].mean() - Y_test[cf_benefiters & control_test].mean()
                cf_harm_effect = Y_test[(~cf_benefiters) & treated_test].mean() - Y_test[(~cf_benefiters) & control_test].mean()
                cf_separation = cf_benefit_effect - cf_harm_effect

        print(f"\nSubgroup Separation (benefit effect - harm effect):")
        print(f"  AutoITE:      {autoite_separation:.3f}" if autoite_separation else "  AutoITE:      N/A")
        print(f"  Causal Forest: {cf_separation:.3f}" if cf_separation else "  Causal Forest: N/A")

        if autoite_separation and cf_separation:
            if autoite_separation > cf_separation:
                print(f"\n  >>> AutoITE achieves {autoite_separation - cf_separation:.3f} better separation")
            else:
                print(f"\n  >>> Causal Forest achieves {cf_separation - autoite_separation:.3f} better separation")

        # Agreement analysis
        agreement = (positive_effect == cf_positive).mean()
        print(f"\nMethod Agreement: {agreement*100:.1f}%")

        # Disagreement analysis
        autoite_only_benefit = positive_effect & (~cf_positive)
        cf_only_benefit = (~positive_effect) & cf_positive
        print(f"  AutoITE predicts benefit, CF predicts harm: {autoite_only_benefit.sum()}")
        print(f"  CF predicts benefit, AutoITE predicts harm: {cf_only_benefit.sum()}")

    return {
        'tau_pred': tau_pred,
        'tau_cf': tau_cf,
        'sigma': sigma,
        'bimodality': bimodal_result,
        'model': model
    }


def analyze_environments(X, T, Y, Y_pre, tau_pred, tau_cf, df, test_idx):
    """
    Analyze what characteristics distinguish predicted benefiters from non-benefiters.
    This reveals what "latent environments" AutoITE has discovered.
    """
    print("\n" + "="*80)
    print("ENVIRONMENT ANALYSIS: What Distinguishes Benefiters from Non-Benefiters?")
    print("="*80)

    # Get test set data
    df_test = df.iloc[test_idx].copy()
    df_test['tau_autoite'] = tau_pred
    df_test['tau_cf'] = tau_cf if tau_cf is not None else np.nan
    df_test['autoite_benefit'] = tau_pred > 0
    df_test['cf_benefit'] = tau_cf > 0 if tau_cf is not None else False

    # Key variables to analyze
    continuous_vars = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                       'failures', 'famrel', 'freetime', 'goout', 'Dalc',
                       'Walc', 'health', 'absences', 'G1', 'G2']

    categorical_vars = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                        'Mjob', 'Fjob', 'reason', 'guardian', 'famsup',
                        'paid', 'activities', 'nursery', 'higher',
                        'internet', 'romantic']

    # =========================================================================
    # AUTOITE ENVIRONMENT ANALYSIS
    # =========================================================================
    print("\n" + "-"*60)
    print("AutoITE: Characteristics of Predicted Benefiters vs Non-Benefiters")
    print("-"*60)

    benefiters = df_test[df_test['autoite_benefit']]
    non_benefiters = df_test[~df_test['autoite_benefit']]

    print(f"\nBenefiters: n={len(benefiters)}, Non-benefiters: n={len(non_benefiters)}")

    # Continuous variables
    print("\n[Continuous Variables - Mean Comparison]")
    print(f"{'Variable':<15} {'Benefiters':>12} {'Non-Benefiters':>15} {'Difference':>12}")
    print("-" * 56)

    significant_continuous = []
    for var in continuous_vars:
        if var in df_test.columns:
            mean_benefit = benefiters[var].mean()
            mean_non = non_benefiters[var].mean()
            diff = mean_benefit - mean_non
            # Simple effect size
            pooled_std = df_test[var].std()
            effect_size = diff / pooled_std if pooled_std > 0 else 0

            if abs(effect_size) > 0.3:  # Medium effect size threshold
                significant_continuous.append((var, mean_benefit, mean_non, diff, effect_size))
            print(f"{var:<15} {mean_benefit:>12.2f} {mean_non:>15.2f} {diff:>+12.2f}")

    # Categorical variables
    print("\n[Categorical Variables - Proportion Comparison]")

    significant_categorical = []
    for var in categorical_vars:
        if var in df_test.columns:
            # Get value counts as proportions
            benefit_props = benefiters[var].value_counts(normalize=True)
            non_benefit_props = non_benefiters[var].value_counts(normalize=True)

            # Find largest difference
            all_values = set(benefit_props.index) | set(non_benefit_props.index)
            max_diff = 0
            max_val = None
            for val in all_values:
                p1 = benefit_props.get(val, 0)
                p2 = non_benefit_props.get(val, 0)
                if abs(p1 - p2) > max_diff:
                    max_diff = abs(p1 - p2)
                    max_val = val
                    max_p1, max_p2 = p1, p2

            if max_diff > 0.15:  # 15% difference threshold
                significant_categorical.append((var, max_val, max_p1, max_p2, max_diff))
                print(f"\n{var}: '{max_val}'")
                print(f"  Benefiters: {max_p1*100:.1f}%, Non-benefiters: {max_p2*100:.1f}% (diff: {max_diff*100:+.1f}%)")

    # =========================================================================
    # KEY FINDINGS SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("KEY ENVIRONMENT CHARACTERISTICS DISCOVERED BY AUTOITE")
    print("="*60)

    if significant_continuous:
        print("\n[Strong Continuous Predictors of Benefit]")
        for var, m1, m2, diff, es in sorted(significant_continuous, key=lambda x: -abs(x[4])):
            direction = "higher" if diff > 0 else "lower"
            print(f"  - {var}: Benefiters have {direction} values (effect size: {es:.2f})")

    if significant_categorical:
        print("\n[Strong Categorical Predictors of Benefit]")
        for var, val, p1, p2, diff in sorted(significant_categorical, key=lambda x: -x[4]):
            if p1 > p2:
                print(f"  - {var}='{val}': More common among benefiters ({p1*100:.0f}% vs {p2*100:.0f}%)")
            else:
                print(f"  - {var}='{val}': Less common among benefiters ({p1*100:.0f}% vs {p2*100:.0f}%)")

    # =========================================================================
    # RESIDUAL-BASED ENVIRONMENT
    # =========================================================================
    print("\n" + "-"*60)
    print("Residual-Based Environment (AutoITE's Core Signal)")
    print("-"*60)

    # The residual from predicting G1 from features represents "latent ability"
    from sklearn.linear_model import Ridge

    # Get feature columns (exclude grades and treatment)
    feature_cols = [c for c in df_test.columns if c not in
                    ['G1', 'G2', 'G3', 'schoolsup', 'tau_autoite', 'tau_cf',
                     'autoite_benefit', 'cf_benefit']]

    # Encode categoricals for the test set
    df_encoded = pd.get_dummies(df_test[feature_cols], drop_first=True)

    # Compute residuals
    X_test_env = df_encoded.values.astype(float)
    Y_pre_test = df_test['G1'].values

    model = Ridge(alpha=1.0)
    model.fit(X_test_env, Y_pre_test)
    residuals = Y_pre_test - model.predict(X_test_env)

    df_test['residual'] = residuals

    print(f"\nResidual (G1 - predicted G1) represents 'latent academic factor'")
    print(f"  Benefiters mean residual:     {df_test[df_test['autoite_benefit']]['residual'].mean():+.3f}")
    print(f"  Non-benefiters mean residual: {df_test[~df_test['autoite_benefit']]['residual'].mean():+.3f}")

    resid_diff = (df_test[df_test['autoite_benefit']]['residual'].mean() -
                  df_test[~df_test['autoite_benefit']]['residual'].mean())

    if resid_diff > 0:
        print(f"\n  >>> Benefiters have HIGHER residuals (overperform baseline prediction)")
        print(f"      Interpretation: Students who do better than expected benefit from support")
    else:
        print(f"\n  >>> Benefiters have LOWER residuals (underperform baseline prediction)")
        print(f"      Interpretation: Students who struggle more than expected benefit from support")

    # =========================================================================
    # CAUSAL FOREST COMPARISON
    # =========================================================================
    if tau_cf is not None:
        print("\n" + "-"*60)
        print("Causal Forest Environment Comparison")
        print("-"*60)

        cf_benefiters = df_test[df_test['cf_benefit']]
        cf_non_benefiters = df_test[~df_test['cf_benefit']]

        print(f"\nCF Benefiters: n={len(cf_benefiters)}, CF Non-benefiters: n={len(cf_non_benefiters)}")

        # Compare key variables
        print("\n[Key Variables - CF vs AutoITE]")
        key_vars = ['G1', 'failures', 'absences', 'studytime', 'Medu', 'Fedu']

        print(f"\n{'Variable':<12} {'AutoITE Benefit':>16} {'CF Benefit':>12} {'Overlap':>10}")
        print("-" * 52)

        for var in key_vars:
            if var in df_test.columns:
                auto_mean = benefiters[var].mean()
                cf_mean = cf_benefiters[var].mean() if len(cf_benefiters) > 0 else np.nan
                print(f"{var:<12} {auto_mean:>16.2f} {cf_mean:>12.2f}")

        # Disagreement analysis
        print("\n[Disagreement Cases]")
        autoite_only = df_test[df_test['autoite_benefit'] & ~df_test['cf_benefit']]
        cf_only = df_test[~df_test['autoite_benefit'] & df_test['cf_benefit']]

        if len(autoite_only) > 0:
            print(f"\nAutoITE says benefit, CF says harm (n={len(autoite_only)}):")
            print(f"  Mean G1: {autoite_only['G1'].mean():.2f}")
            print(f"  Mean failures: {autoite_only['failures'].mean():.2f}")
            print(f"  Mean residual: {autoite_only['residual'].mean():+.3f}")

        if len(cf_only) > 0:
            print(f"\nCF says benefit, AutoITE says harm (n={len(cf_only)}):")
            print(f"  Mean G1: {cf_only['G1'].mean():.2f}")
            print(f"  Mean failures: {cf_only['failures'].mean():.2f}")
            print(f"  Mean residual: {cf_only['residual'].mean():+.3f}")

    return df_test


def main():
    """Run UCI student intervention experiment."""
    print("="*80)
    print("UCI STUDENT INTERVENTION EXPERIMENT")
    print("Real-World Application of AutoITE")
    print("="*80)

    # Download/load data
    filepath = download_student_data()
    if filepath is None:
        # Create synthetic student-like data for demonstration
        print("\nUsing synthetic student data for demonstration...")
        np.random.seed(42)
        n = 395  # Same size as real dataset

        # Latent ability
        ability = np.random.normal(10, 3, n)

        # Features
        X = np.column_stack([
            np.random.normal(0, 1, (n, 10)),  # Continuous features
            np.random.binomial(1, 0.5, (n, 5))  # Binary features
        ])

        # Treatment: school support
        T = np.random.binomial(1, 0.3, n)

        # Baseline: G1 (affected by ability)
        Y_pre = ability + np.random.normal(0, 2, n)

        # Treatment effect: varies by ability
        tau = np.where(ability < 10, 1.5, -0.5)  # Low ability benefits, high doesn't

        # Outcome: G3
        Y = ability + T * tau + np.random.normal(0, 2, n)

        # Clip to valid grade range
        Y_pre = np.clip(Y_pre, 0, 20)
        Y = np.clip(Y, 0, 20)

        df = None
    else:
        X, T, Y, Y_pre, df = load_and_prepare_data(filepath)

    # Run experiment
    results = run_student_experiment(X, T, Y, Y_pre, k_fraction=0.10)

    # Analyze discovered environments
    if df is not None:
        # Get test indices (same as in run_student_experiment)
        np.random.seed(42)
        n = len(Y)
        indices = np.random.permutation(n)
        train_size = int(0.8 * n)
        test_idx = indices[train_size:]

        df_analysis = analyze_environments(
            X, T, Y, Y_pre,
            results['tau_pred'],
            results.get('tau_cf'),
            df,
            test_idx
        )
        results['df_analysis'] = df_analysis

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

    return results


if __name__ == "__main__":
    main()

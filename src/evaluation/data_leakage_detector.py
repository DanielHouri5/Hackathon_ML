"""Data leakage detection utilities using mutual information (MI).

This module computes mutual information between each feature in X and
the target y, prints a compact report highlighting influential
features (MI > 0), and returns a DataFrame sorted by MI score. Features
with zero MI are summarized at the end which is useful for quick
feature-cleaning during hackathons.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def detect_data_leakage(X, y, task="classification"):
    """Compute MI per feature and print a formatted relevance report.

    Parameters
    - X: pandas.DataFrame or numpy.ndarray of input features.
    - y: array-like target values.
    - task: str, either "classification" or "regression" to select the
      appropriate mutual information estimator.

    Behavior
    - Converts a numpy array to DataFrame if necessary.
    - Computes MI per feature using scikit-learn's estimators.
    - Builds and returns a DataFrame sorted by the MI score (descending).
    - Prints a human-readable report listing influential features and
      summarizing features with zero MI.

    Returns
    - pd.DataFrame: columns ['feature', 'mutual_info'] sorted by 'mutual_info'.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    results = []
    for col in X.columns:
        try:
            if task == "classification":
                mi_val = mutual_info_classif(X[[col]], y, discrete_features='auto', random_state=42)
            else:
                mi_val = mutual_info_regression(X[[col]], y, random_state=42)
            mi_val = mi_val[0] if len(mi_val) > 0 else 0
        except Exception:
            mi_val = 0
        results.append({'feature': col, 'mutual_info': mi_val})

    df_results = pd.DataFrame(results).sort_values(by='mutual_info', ascending=False)
    
    # Split results: features with influence and those without
    influential_features = df_results[df_results['mutual_info'] > 0]
    zero_features_count = len(df_results[df_results['mutual_info'] <= 0])

    # --- Print formatted report ---
    print("\n" + "!"*50)
    print(f"🔍 DATA LEAKAGE & FEATURE RELEVANCE ({task.upper()})")
    print("!"*50)
    print(f"{'Feature Name':<35} | {'MI Score':<10}")
    print("-" * 50)

    for _, row in influential_features.iterrows():
        alert = "🚨 HIGH" if row['mutual_info'] > 0.5 else "✅"
        print(f"{row['feature'][:35]:<35} | {row['mutual_info']:.4f}  {alert}")

    if zero_features_count > 0:
        print("-" * 50)
        print(f"... and {zero_features_count} other features with 0.0000 MI score.")
    
    print("!"*50 + "\n")

    return df_results
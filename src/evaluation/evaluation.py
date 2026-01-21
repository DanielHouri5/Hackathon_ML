import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, mean_squared_error)
import src.config as config

# --- 1. DATA LEAKAGE & FEATURE RELEVANCE ---
def detect_data_leakage(X, y, task="classification"):
    """
    Computes Mutual Information to find highly influential features or potential leakage.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    results = []
    for col in X.columns:
        try:
            # Mutual Info is robust but needs 2D input
            x_col = X[[col]]
            if task == "classification":
                mi_val = mutual_info_classif(x_col, y, random_state=42)[0]
            else:
                mi_val = mutual_info_regression(x_col, y, random_state=42)[0]
        except Exception:
            mi_val = 0
        results.append({'feature': col, 'mutual_info': mi_val})

    df_results = pd.DataFrame(results).sort_values(by='mutual_info', ascending=False)
    influential_features = df_results[df_results['mutual_info'] > 0]
    
    print("\n" + "!"*50)
    print(f"🔍 DATA LEAKAGE & FEATURE RELEVANCE ({task.upper()})")
    print("!"*50)
    print(f"{'Feature Name':<35} | {'MI Score':<10}")
    print("-" * 50)

    for _, row in influential_features.iterrows():
        # MI > 0.5 is often a red flag for leakage in many hackathon datasets
        alert = "🚨 HIGH" if row['mutual_info'] > 0.5 else "✅"
        print(f"{row['feature'][:35]:<35} | {row['mutual_info']:.4f}  {alert}")

    zero_count = len(df_results) - len(influential_features)
    if zero_count > 0:
        print("-" * 50)
        print(f"... and {zero_count} other features with 0.0000 MI score.")
    print("!"*50 + "\n")

    return df_results

# --- 2. MODEL EXPLAINABILITY (SHAP) ---
def shap_tree_summary(model, X, max_features=10, title="SHAP Feature Impact"):
    """
    Generates a SHAP summary plot. Handles different output formats of SHAP values.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    # TreeExplainer is optimized for XGBoost, LightGBM, CatBoost, and RandomForest
    explainer = shap.TreeExplainer(model)
    
    # Check for large datasets to avoid hang-ups during a demo
    if len(X) > 1000:
        print(f"Sampling 1000 rows for SHAP calculation...")
        X_sample = X.sample(1000, random_state=42)
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # Handle various SHAP output formats (Binary classification, Multi-class, Regression)
    if isinstance(shap_values, list):
        # Scikit-learn Random Forest returns a list for each class
        shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif len(shap_values.shape) == 3:
        # Some XGBoost versions return (rows, features, classes)
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values_to_plot, X_sample, max_display=max_features, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODELS_PLOTS_DIR, "shap_summary.png"))
    plt.close()

# --- 3. CROSS-VALIDATION STABILITY ---
def cross_validation_stability(model, X, y, task="classification", n_splits=5):
    """
    Runs CV and reports on the stability (standard deviation) of the results.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    
    y = pd.Series(y) if not isinstance(y, pd.Series) else y

    scores_map = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'rmse': []}

    if task == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        # Assuming the minority class is the target for Precision/Recall/F1
        pos_label = y.value_counts().index[-1]
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "classification":
            scores_map['accuracy'].append(accuracy_score(y_test, y_pred))
            scores_map['precision'].append(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
            scores_map['recall'].append(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
            scores_map['f1'].append(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            scores_map['rmse'].append(rmse)

    # --- Print Report ---
    print("\n" + "="*45)
    print(f"📊 CROSS-VALIDATION STABILITY ({task.upper()})")
    print("="*45)

    if task == "classification":
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for m in metrics:
            mean_s = np.mean(scores_map[m])
            std_s = np.std(scores_map[m])
            print(f"{m.capitalize():<12} : {mean_s:.4f} (±{std_s:.4f})")
        print(f"{'Target Class':<12} : '{pos_label}'")
    else:
        print(f"RMSE Mean    : {np.mean(scores_map['rmse']):.4f}")
        print(f"RMSE Std     : {np.std(scores_map['rmse']):.4f}")

    print("="*45 + "\n")

    return {k: (np.mean(v) if v else None) for k, v in scores_map.items()}

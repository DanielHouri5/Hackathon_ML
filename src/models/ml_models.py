# models_optimized.py
import time
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

model_dir = "../outputs/trained_models/"

# ----------------------------
# Helper functions
# ----------------------------

def compute_metrics(y_true, y_pred, y_proba=None, task_type="classification"):
    metrics = {}
    if task_type == "classification":
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except:
                metrics['roc_auc'] = None
    else:  # regression
        metrics['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
    return metrics


def save_model(model, model_name, model_dir="saved_models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filepath = os.path.join(model_dir, f"{model_name}.joblib")
    if hasattr(model, 'save_model'):  # CatBoost
        model.save_model(filepath)
    else:
        joblib.dump(model, filepath)
    return filepath


def get_feature_importance(model, feature_names=None):
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_.flatten()
    if importance is not None and feature_names is not None:
        importance = dict(zip(feature_names, importance))
    elif importance is not None:
        importance = dict(enumerate(importance))
    return importance

# ----------------------------
# Print results helper
# ----------------------------
def print_model_results(results, top_n_features=10):
    """
    מדפיס בצורה מסודרת את ה-metrics וה-feature importance
    results: dict שהפונקציה train_model מחזירה
    top_n_features: כמה תכונות חשובות להציג (ל-classification/regression)
    """
    import pandas as pd

    metrics = results.get("metrics", {})
    feature_importance = results.get("feature_importance", None)
    model_name = metrics.get("train", {}).get("model_name", "MODEL")

    print(f"\n===== RESULTS FOR {model_name.upper()} =====\n")

    # Metrics table
    metric_rows = []
    for split in ["train", "val", "test"]:
        if split in metrics:
            row = metrics[split].copy()
            row["split"] = split
            metric_rows.append(row)
    if metric_rows:
        df_metrics = pd.DataFrame(metric_rows)
        df_metrics = df_metrics.set_index("split")
        print("Metrics:")
        print(df_metrics.round(4))
        print()

    # Feature importance
    if feature_importance:
        print(f"Top {top_n_features} Feature Importances:")
        # ממיינים לפי ערך מוחלט, יורד
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, importance in sorted_features[:top_n_features]:
            print(f"{feature}: {importance:.4f}")
    else:
        print("No feature importance available for this model.")

# ----------------------------
# General training function
# ----------------------------

def train_model(model_type,
                X_train_le, X_val_le, X_test_le,
                y_train, y_val, y_test,
                task_type="classification",
                save_model_flag=False,
                model_dir=model_dir,
                **params):
    
    model_name = model_type.lower()
    
    # Initialize model
    if model_type == "logistic_regression":
        model = LogisticRegression(**params)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(**params) if task_type=="classification" else DecisionTreeRegressor(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params) if task_type=="classification" else RandomForestRegressor(**params)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(**params) if task_type=="classification" else xgb.XGBRegressor(**params)
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(**params) if task_type=="classification" else lgb.LGBMRegressor(**params)
    elif model_type == "catboost":
        model = cb.CatBoostClassifier(**params) if task_type=="classification" else cb.CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train
    start_time = time.time()
    if model_type == "catboost":
        model.fit(X_train_le, y_train, eval_set=(X_val_le, y_val), verbose=0)
    else:
        model.fit(X_train_le, y_train)
    train_time = time.time() - start_time

    # Predictions
    def predict_all(X):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if task_type=="classification" else None
        return y_pred, y_proba

    y_pred_train, y_proba_train = predict_all(X_train_le)
    y_pred_val, y_proba_val = predict_all(X_val_le)
    y_pred_test, y_proba_test = predict_all(X_test_le)

    # Compute metrics
    train_metrics = compute_metrics(y_train, y_pred_train, y_proba_train, task_type)
    val_metrics = compute_metrics(y_val, y_pred_val, y_proba_val, task_type)
    test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test, task_type)

    # Add metadata
    for m in [train_metrics, val_metrics, test_metrics]:
        m.update({
            "train_time": train_time,
            "n_samples": X_train_le.shape[0],
            "n_features": X_train_le.shape[1],
            "model_name": model_name,
            "task_type": task_type
        })

    # Feature importance
    feature_importance = get_feature_importance(model, feature_names=X_train_le.columns if isinstance(X_train_le, pd.DataFrame) else None)

    # Print summary
    print(f"\n===== {model_name.upper()} SUMMARY =====")
    print(f"Task type: {task_type}")
    if model_type in ["decision_tree", "random_forest"]:
        if hasattr(model, "max_depth"):
            print(f"Max depth: {model.max_depth}")
        if hasattr(model, "n_estimators"):
            print(f"Number of estimators: {getattr(model, 'n_estimators', 'N/A')}")
    print(f"Training time: {train_time:.3f}s")
    print("Train metrics:", train_metrics)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    if feature_importance is not None:
        print("Feature importance:", feature_importance)

    # Save model if requested
    if save_model_flag:
        save_model(model, model_name, model_dir=model_dir)

    # Return everything
    return {
        "model": model,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics
        },
        "feature_importance": feature_importance,
        "predictions": {
            "train": y_pred_train,
            "val": y_pred_val,
            "test": y_pred_test,
            "train_proba": y_proba_train,
            "val_proba": y_proba_val,
            "test_proba": y_proba_test
        }
    }



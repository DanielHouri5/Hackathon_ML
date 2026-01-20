# MODELS.PY
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
# General model template
# ----------------------------

def train_logreg(X_train, y_train, X_test, y_test,
                 task_type="classification", 
                 save_model_flag=False,
                 model_dir=model_dir,
                 **params):
    model_name = "logistic_regression"
    model = LogisticRegression(**params)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test) if task_type=="classification" else None

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba=model.predict_proba(X_train) if task_type=="classification" else None, task_type=task_type)
    train_metrics["train_time"] = train_time
    train_metrics["n_samples"] = X_train.shape[0]
    train_metrics["n_features"] = X_train.shape[1]
    train_metrics["model_name"] = model_name
    train_metrics["task_type"] = task_type

    test_metrics = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test, task_type=task_type)
    feature_importance = get_feature_importance(model, feature_names=X_train.columns if isinstance(X_train, pd.DataFrame) else None)

    if save_model_flag:
        save_model(model, model_name, model_dir=model_dir)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "predictions": {
            "train": y_pred_train,
            "test": y_pred_test,
            "test_proba": y_proba_test
        }
    }


# ----------------------------
# Decision Tree
# ----------------------------

def train_dt(X_train, y_train, X_test, y_test,
             task_type="classification", 
             save_model_flag=False,
             model_dir=model_dir,
             **params):
    model_name = "decision_tree"
    model = DecisionTreeClassifier(**params) if task_type=="classification" else DecisionTreeRegressor(**params)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test) if task_type=="classification" else None

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba=model.predict_proba(X_train) if task_type=="classification" else None, task_type=task_type)
    train_metrics["train_time"] = train_time
    train_metrics["n_samples"] = X_train.shape[0]
    train_metrics["n_features"] = X_train.shape[1]
    train_metrics["model_name"] = model_name
    train_metrics["task_type"] = task_type

    test_metrics = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test, task_type=task_type)
    feature_importance = get_feature_importance(model, feature_names=X_train.columns if isinstance(X_train, pd.DataFrame) else None)

    if save_model_flag:
        save_model(model, model_name, model_dir=model_dir)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "predictions": {
            "train": y_pred_train,
            "test": y_pred_test,
            "test_proba": y_proba_test
        }
    }


# ----------------------------
# Random Forest
# ----------------------------

def train_rf(X_train, y_train, X_test, y_test,
             task_type="classification", 
             save_model_flag=False,
             model_dir=model_dir,
             **params):
    model_name = "random_forest"
    model = RandomForestClassifier(**params) if task_type=="classification" else RandomForestRegressor(**params)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test) if task_type=="classification" else None

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba=model.predict_proba(X_train) if task_type=="classification" else None, task_type=task_type)
    train_metrics["train_time"] = train_time
    train_metrics["n_samples"] = X_train.shape[0]
    train_metrics["n_features"] = X_train.shape[1]
    train_metrics["model_name"] = model_name
    train_metrics["task_type"] = task_type

    test_metrics = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test, task_type=task_type)
    feature_importance = get_feature_importance(model, feature_names=X_train.columns if isinstance(X_train, pd.DataFrame) else None)

    if save_model_flag:
        save_model(model, model_name, model_dir=model_dir)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "predictions": {
            "train": y_pred_train,
            "test": y_pred_test,
            "test_proba": y_proba_test
        }
    }


# ----------------------------
# XGBoost
# ----------------------------

def train_xgb(X_train, y_train, X_test, y_test,
              task_type="classification",
              save_model_flag=False,
              model_dir=model_dir,
              **params):
    model_name = "xgboost"
    model = xgb.XGBClassifier(**params) if task_type=="classification" else xgb.XGBRegressor(**params)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test) if task_type=="classification" else None

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba=model.predict_proba(X_train) if task_type=="classification" else None, task_type=task_type)
    train_metrics["train_time"] = train_time
    train_metrics["n_samples"] = X_train.shape[0]
    train_metrics["n_features"] = X_train.shape[1]
    train_metrics["model_name"] = model_name
    train_metrics["task_type"] = task_type

    test_metrics = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test, task_type=task_type)
    feature_importance = get_feature_importance(model, feature_names=X_train.columns if isinstance(X_train, pd.DataFrame) else None)

    if save_model_flag:
        save_model(model, model_name, model_dir=model_dir)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "predictions": {
            "train": y_pred_train,
            "test": y_pred_test,
            "test_proba": y_proba_test
        }
    }


# ----------------------------
# LightGBM
# ----------------------------

def train_lgbm(X_train, y_train, X_test, y_test,
               task_type="classification",
               save_model_flag=False,
               model_dir=model_dir,
               **params):
    model_name = "lightgbm"
    model = lgb.LGBMClassifier(**params) if task_type=="classification" else lgb.LGBMRegressor(**params)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test) if task_type=="classification" else None

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba=model.predict_proba(X_train) if task_type=="classification" else None, task_type=task_type)
    train_metrics["train_time"] = train_time
    train_metrics["n_samples"] = X_train.shape[0]
    train_metrics["n_features"] = X_train.shape[1]
    train_metrics["model_name"] = model_name
    train_metrics["task_type"] = task_type

    test_metrics = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test, task_type=task_type)
    feature_importance = get_feature_importance(model, feature_names=X_train.columns if isinstance(X_train, pd.DataFrame) else None)

    if save_model_flag:
        save_model(model, model_name, model_dir=model_dir)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "predictions": {
            "train": y_pred_train,
            "test": y_pred_test,
            "test_proba": y_proba_test
        }
    }


# ----------------------------
# CatBoost
# ----------------------------

def train_catboost(X_train, y_train, X_test, y_test,
                   task_type="classification",
                   save_model_flag=False,
                   model_dir=model_dir,
                   **params):
    model_name = "catboost"
    model = cb.CatBoostClassifier(**params) if task_type=="classification" else cb.CatBoostRegressor(**params)

    start_time = time.time()
    model.fit(X_train, y_train, verbose=0)
    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test) if task_type=="classification" else None

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba=model.predict_proba(X_train) if task_type=="classification" else None, task_type=task_type)
    train_metrics["train_time"] = train_time
    train_metrics["n_samples"] = X_train.shape[0]
    train_metrics["n_features"] = X_train.shape[1]
    train_metrics["model_name"] = model_name
    train_metrics["task_type"] = task_type

    test_metrics = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test, task_type=task_type)
    feature_importance = get_feature_importance(model, feature_names=X_train.columns if isinstance(X_train, pd.DataFrame) else None)

    if save_model_flag:
        save_model(model, model_name, model_dir=model_dir)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "predictions": {
            "train": y_pred_train,
            "test": y_pred_test,
            "test_proba": y_proba_test
        }
    }


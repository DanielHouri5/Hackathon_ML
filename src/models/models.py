import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid, ParameterSampler, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostClassifier, CatBoostRegressor
from .parm_configs import DEFAULT_PARAMS, PARAM_GRIDS
from src.pre_processing.pre_processing import run_preprocessing_pipeline
import warnings

# ==================== Model Builders ====================
def build_logistic_regression(task_type="classification", **params):
    params.setdefault('max_iter', 1000)
    return LogisticRegression(**params)

def build_decision_tree(task_type="classification", **params):
    return DecisionTreeClassifier(**params) if task_type == "classification" else DecisionTreeRegressor(**params)

def build_random_forest(task_type="classification", **params):
    return RandomForestClassifier(**params) if task_type == "classification" else RandomForestRegressor(**params)

def build_adaboost(task_type="classification", **params):
    return AdaBoostClassifier(**params) if task_type == "classification" else AdaBoostRegressor(**params)

def build_gradient_boosting(task_type="classification", **params):
    return GradientBoostingClassifier(**params) if task_type == "classification" else GradientBoostingRegressor(**params)

def build_xgboost(task_type="classification", **params):
    return XGBClassifier(**params) if task_type == "classification" else XGBRegressor(**params)

def build_lightgbm(task_type="classification", **params):
    return LGBMClassifier(verbosity=-1, **params) if task_type == "classification" else LGBMRegressor(**params)

def build_catboost(task_type="classification", **params):
    params.setdefault('verbose', 0)
    return CatBoostClassifier(**params) if task_type == "classification" else CatBoostRegressor(**params)

# ==================== Hyperparameter Tuning ====================

def tune_and_report(model_key, X_train, y_train, task_type="classification"):
    param_dist = PARAM_GRIDS.get(model_key, {})
    
    base_model = globals()[f"build_{model_key}"](task_type=task_type)
    
    if task_type == "classification":
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'auc': 'roc_auc_ovr'
        }
        refit_metric = 'f1'
    else:
        scoring = {'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
        refit_metric = 'rmse'

    if isinstance(param_dist, list):
        total_combos = sum(len(ParameterGrid(p)) for p in param_dist)
    else:
        total_combos = len(ParameterGrid(param_dist))
    
    actual_n_iter = min(10, total_combos)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=actual_n_iter,
        cv=3,
        scoring=scoring,
        refit=refit_metric, 
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"\n{model_key.replace('_', ' ').title()}:")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        random_search.fit(X_train, y_train)
    
    results_df = pd.DataFrame(random_search.cv_results_)
    
    param_cols = [c for c in results_df.columns if 'param_' in c]
    metric_cols = [f'mean_test_{m}' for m in scoring.keys()]
    
    summary_table = results_df[param_cols + metric_cols].copy()
    
    new_names = {c: c.replace('param_', '') for c in param_cols}
    new_names.update({f'mean_test_{m}': m for m in scoring.keys()})
    summary_table = summary_table.rename(columns=new_names)
    
    summary_table = summary_table.sort_values(by=refit_metric, ascending=False).head(5)
    
    print(summary_table.to_string(index=True))
    print("-" * 30)
    
    return random_search.best_estimator_

# ==================== Metric Engine ====================
def compute_metrics(y_true, y_pred, y_proba=None, task_type="classification"):
    if task_type == "classification":
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'roc_auc': (roc_auc_score(y_true, y_proba, multi_class='ovr') 
                        if y_proba is not None else None)
        }
    return {
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

# ==================== Experiment Runner ====================
def get_fit_params(model_obj, X_val, y_val):
    if isinstance(model_obj, (CatBoostClassifier, CatBoostRegressor)):
        return {
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
            "verbose": 0
        }
    
    if isinstance(model_obj, (LGBMClassifier, LGBMRegressor)):
        return {
            "eval_set": [(X_val, y_val)],
            "callbacks": [early_stopping(stopping_rounds=50), log_evaluation(period=0)]
        }
        
    if isinstance(model_obj, (XGBClassifier, XGBRegressor)):
        return {
            "eval_set": [(X_val, y_val)],
            "verbose": False
        }
        
    return {} 

def train_model(model_obj, X_train, y_train, X_val=None, y_val=None):
    fit_params = get_fit_params(model_obj, X_val, y_val)
    
    model_obj.fit(X_train, y_train, **fit_params)
    return model_obj

def evaluate_model(model_obj, X_data, y_data, task_type="classification"):
    preds = model_obj.predict(X_data)
    
    proba = None
    if task_type == "classification":
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(X_data)[:, 1]
        elif hasattr(model_obj, "decision_function"):
            proba = model_obj.decision_function(X_data)
            
    return compute_metrics(y_data, preds, proba, task_type)

def run_model_comparison(models_to_run, X_train, y_train, X_val, y_val, task_type="classification"):
    all_experiments = {}
    
    for name, model_obj in models_to_run.items():
        print(f"Training {name}...")
        start_time = time.time()
        trained_model = train_model(model_obj, X_train, y_train, X_val, y_val)
        train_time = time.time() - start_time
        
        val_metrics = evaluate_model(trained_model, X_val, y_val, task_type)
        
        all_experiments[name] = {
            "model_object": trained_model,
            "metrics": {"val": val_metrics}, 
            "train_time": train_time,
            "params": trained_model.get_params()  
        }
        print(f"{name} finished (F1 Val: {val_metrics['f1']:.4f})")
        
    return all_experiments

# ==================== Artifacts & Importance ====================
def get_feature_importance(model, feature_names):
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_.flatten()) 
    if importance is not None:
        return pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    return None

def save_model_artifact(model, name, folder="../outputs/models/"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.joblib")
    joblib.dump(model, path)
    return path


# ==================== Results Printer ====================
def display_experiment_summary(all_experiments, split='val'):
    summary_data = []
    
    for model_name, results in all_experiments.items():
        metrics = results['metrics'][split].copy()
        metrics['Model'] = model_name
        metrics['Train Time (sec)'] = results['train_time']
        summary_data.append(metrics)
    
    df_summary = pd.DataFrame(summary_data)
    cols = ['Model'] + [c for c in df_summary.columns if c != 'Model']
    df_summary = df_summary[cols].set_index('Model')
    
    print(f"\n" + "="*30)
    print(f" FINAL SUMMARY ({split.upper()} SET) ")
    print("="*30)
    
    print(df_summary.round(4).to_string())
    
    return df_summary

if __name__ == "__main__":
    folder_path = "outputs/artifacts"
    file_path = os.path.join(folder_path, "processed_bundles.joblib")
    if os.path.exists(file_path):
        print(f"📦 Loading preprocessed data from cache: {file_path}")
        data_bundles = joblib.load(file_path)
    else:
        print("🔄 No cache found. Running preprocessing pipeline...")
        data_bundles = run_preprocessing_pipeline()

    X_train_le, X_val_le, X_test_le, y_train, y_val, y_test = data_bundles['trees_data']
    X_train_oh, X_val_oh, X_test_oh, _, _, _ = data_bundles['linear_data']

    tree_algo_keys = [
        "decision_tree", "random_forest", "adaboost", 
        "gradient_boosting", "xgboost", "lightgbm"
    ]
    linear_algo_keys = ["logistic_regression"]

    all_experiments = {}

    print("\nPHASE 1: TREE-BASED MODELS TUNING")
    for key in tree_algo_keys:
        best_model = tune_and_report(key, X_train_le, y_train)
        
        start_time = time.time()
        trained_model = train_model(best_model, X_train_le, y_train, X_val_le, y_val)
        train_time = time.time() - start_time
        
        val_metrics = evaluate_model(trained_model, X_val_le, y_val)
        all_experiments[key] = {
            "model_object": trained_model,
            "metrics": {"val": val_metrics},
            "train_time": train_time,
            "params": trained_model.get_params()
        }

    print("\n📈 PHASE 2: LINEAR MODELS TUNING")
    for key in linear_algo_keys:
        best_model = tune_and_report(key, X_train_oh, y_train)
        
        start_time = time.time()
        trained_model = train_model(best_model, X_train_oh, y_train, X_val_oh, y_val)
        train_time = time.time() - start_time
        
        val_metrics = evaluate_model(trained_model, X_val_oh, y_val)
        all_experiments[key] = {
            "model_object": trained_model,
            "metrics": {"val": val_metrics},
            "train_time": train_time,
            "params": trained_model.get_params()
        }

    final_summary = display_experiment_summary(all_experiments, split='val')

    best_name = final_summary['f1'].idxmax()
    winner_obj = all_experiments[best_name]['model_object']
    
    print(f"\n🏆 OVERALL WINNER: {best_name}")
    
    final_X = X_test_le if best_name in tree_algo_keys else X_test_oh
    test_metrics = evaluate_model(winner_obj, final_X, y_test)
    
    print("\n" + "="*40)
    print(f" 🏁 FINAL TEST PERFORMANCE: {best_name} ")
    print("="*40)
    for m, v in test_metrics.items(): print(f"{m.upper():<10}: {v:.4f}")

    save_model_artifact(winner_obj, f"best_tuned_{best_name}")

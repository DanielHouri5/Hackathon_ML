import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid, ParameterSampler, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostClassifier, CatBoostRegressor
from src.evaluation.evaluation import cross_validation_stability, detect_data_leakage, shap_tree_summary
from .blending_ensemble import BlendingEnsemble
from .stacking_ensemble import StackingEnsemble
from .parm_configs import DEFAULT_PARAMS, PARAM_GRIDS
from src.pre_processing.pre_processing import run_preprocessing_pipeline
from src.visualization.models_visualization import plot_model_comparison, plot_roc_curve, plot_feature_importance, plot_confusion_matrix, plot_n_trees_vs_f1
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 1. MODEL BUILDERS 
# =============================================================================
def build_logistic_regression(task_type="classification", **params):
    params.setdefault('max_iter', 1000) 
    params.setdefault('solver', 'lbfgs') 
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

# =============================================================================
# 2. HYPERPARAMETER TUNING 
# =============================================================================
def run_tuning(model_key, X_train, y_train, task_type="classification"):
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
    
    rs = RandomizedSearchCV(
        estimator=base_model, 
        param_distributions=param_dist,
        n_iter=min(10, total_combos), 
        cv=3, 
        scoring=scoring,
        refit=refit_metric, 
        random_state=42, 
        n_jobs=-1
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        rs.fit(X_train, y_train)
    return rs

def report_tuning_results(model_key, search_results):
    print(f"\n📌 {model_key.replace('_', ' ').title()} Hyperparameter Search Results:")
    
    df = pd.DataFrame(search_results.cv_results_)
    
    param_cols = [c for c in df.columns if 'param_' in c]
    metric_cols = [c for c in df.columns if 'mean_test_' in c]
    time_col = ['mean_fit_time'] 
    
    summary = df[param_cols + metric_cols + time_col].copy()
    
    new_names = {c: c.replace('param_', '').replace('mean_test_', '') for c in (param_cols + metric_cols)}
    new_names['mean_fit_time'] = 'train_time_avg'
    summary = summary.rename(columns=new_names)
    
    sort_col = 'f1' if 'f1' in summary.columns else 'rmse'
    print(summary.sort_values(by=sort_col, ascending=False).head(3).to_string(index=False))
    print("-" * 30)

# =============================================================================
# 3. TRAINING ENGINE 
# =============================================================================
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

# =============================================================================
# 4. EVALUATION & METRICS 
# =============================================================================
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

def evaluate_model(model_obj, X_data, y_data, task_type="classification"):
    if isinstance(model_obj, (BlendingEnsemble, StackingEnsemble)):
        X_tree, X_linear = X_data
        preds = model_obj.predict(X_tree, X_linear)
        proba = model_obj.predict_proba(X_tree, X_linear) if task_type == "classification" else None
    else:
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

# =============================================================================
# 5. EXPERIMENT PHASES 
# =============================================================================
def execute_phase(algo_keys, X_train, y_train, X_val, y_val, phase_name, task_type="classification"):
    print(f"\n🚀 STARTING PHASE: {phase_name}")
    phase_results = {}
    
    for key in algo_keys:
        # 1. Tuning
        search_obj = run_tuning(key, X_train, y_train, task_type)
        report_tuning_results(key, search_obj)
        
        # 2. Final Training (with Early Stopping)
        start_time = time.time()
        model = train_model(search_obj.best_estimator_, X_train, y_train, X_val, y_val)
        elapsed = time.time() - start_time
        
        # 3. Validation Evaluation
        search_obj = run_tuning(key, X_train, y_train, task_type)
        phase_results[key] = {
            "model_object": model,
            "search_object": search_obj,
            "metrics": {"val": evaluate_model(model, X_val, y_val, task_type)},
            "train_time": elapsed
        }
    return phase_results

def run_final_ensemble_test(all_exps, X_te_t, X_te_l, y_te, tree_keys):
    print("\n🏁 FINAL TEST PERFORMANCE (INCLUDING ENSEMBLES) ")
    test_rows = []
    for name, exp in all_exps.items():
        model = exp['model_object']
        
        if isinstance(model, (BlendingEnsemble, StackingEnsemble)):
            y_pred = model.predict(X_te_t, X_te_l)
            y_proba = model.predict_proba(X_te_t, X_te_l)
        else:
            X_test = X_te_t if name in tree_keys else X_te_l
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
        m = compute_metrics(y_te, y_pred, y_proba)
        m['Model'] = name
        test_rows.append(m)
    
    summary_df = pd.DataFrame(test_rows).set_index('Model').sort_values(by='f1', ascending=False)
    print(summary_df.round(4).to_string())
    return summary_df

# =============================================================================
# 6. UTILITIES & ARTIFACTS 
# =============================================================================
def get_feature_importance(model, feature_names):
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_.flatten()) 
    if importance is not None:
        return pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    return None

def save_model_artifact(model, name, folder="outputs/trained_models/"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.joblib")
    joblib.dump(model, path)
    return path

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

def analyze_tuning_path(model_key, search_obj):
    df_results = pd.DataFrame(search_obj.cv_results_)
   
    param_col = [c for c in df_results.columns if 'n_estimators' in c]
    
    if not param_col:
        print(f"No n_estimators parameter found in tuning results for {model_key}")
        return

    plot_df = pd.DataFrame({
        'n_estimators': df_results[param_col[0]],
        'mean_test_score': df_results['mean_test_f1'] if 'mean_test_f1' in df_results else df_results['mean_test_score']
    })
    
    plot_n_trees_vs_f1(plot_df, model_key)
# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================
def main():
    # 1. Data Loading
    file_path = "outputs/artifacts/processed_bundles.joblib"
    data_bundles = run_preprocessing_pipeline() if not os.path.exists(file_path) else joblib.load(file_path)
    X_tr_t, X_val_t, X_te_t, y_tr, y_val, y_te = data_bundles['trees_data']
    X_tr_l, X_val_l, X_te_l, _, _, _ = data_bundles['linear_data']
    feature_names = X_tr_t.columns.tolist()
    print("🛡️ Running Pre-flight Leakage Check...")
    detect_data_leakage(X_tr_t, y_tr, task="classification")
    
    # 2. Base Models
    tree_keys = ["decision_tree", "random_forest", "adaboost", "gradient_boosting", "xgboost", "lightgbm"]
    linear_keys = ["logistic_regression"]
    all_experiments = {}
    all_experiments.update(execute_phase(tree_keys, X_tr_t, y_tr, X_val_t, y_val, "Tree Models"))
    all_experiments.update(execute_phase(linear_keys, X_tr_l, y_tr, X_val_l, y_val, "Linear Models"))

    # 3. Ensemble (Blending & Stacking)
    top_3_keys = ['logistic_regression', 'lightgbm', 'random_forest']
        
    # Blending
    blender = BlendingEnsemble({k: all_experiments[k]['model_object'] for k in top_3_keys}, tree_keys=tree_keys)
    start_blend = time.time()
    blender.fit(X_val_t, X_val_l, y_val)
    all_experiments["blending_ensemble"] = {
        "model_object": blender,
        "metrics": {"val": evaluate_model(blender, (X_val_t, X_val_l), y_val)},
        "train_time": time.time() - start_blend
    }

    # Stacking
    stacker = StackingEnsemble({k: clone(all_experiments[k]['model_object']) for k in top_3_keys}, n_splits=5, tree_keys=tree_keys)
    start_stack = time.time()
    stacker.fit(X_tr_t, X_tr_l, y_tr)
    all_experiments["stacking_ensemble"] = {
        "model_object": stacker,
        "metrics": {"val": evaluate_model(stacker, (X_val_t, X_val_l), y_val)},
        "train_time": time.time() - start_stack
    }
    # 4. Visualizations
    plot_model_comparison(all_experiments, split='val')
    plot_roc_curve(all_experiments, X_val_t, X_val_l, y_val, tree_keys)

    # 5. Final Summary & Test Evaluation
    display_experiment_summary(all_experiments, split='val')
    test_summary = run_final_ensemble_test(all_experiments, X_te_t, X_te_l, y_te, tree_keys)

    # 6. Save & Deep Analysis of Best Model
    best_model_name = test_summary.index[0]
    best_exp = all_experiments[best_model_name]
    best_model_obj = best_exp['model_object']

    print(f"\n🏆 Overall Winner: {best_model_name}")
    save_model_artifact(best_model_obj, f"best_model_{best_model_name}")

    if isinstance(best_model_obj, (BlendingEnsemble, StackingEnsemble)):
        X_val_winner = (X_val_t, X_val_l)
    else:
        print(f"⚖️ Checking stability for {best_model_name}...")
        cross_validation_stability(best_model_obj, X_tr_t, y_tr)
        X_val_winner = X_val_t if best_model_name in tree_keys else X_val_l

    plot_confusion_matrix(best_model_obj, X_val_winner, y_val, class_names=['Negative', 'Positive'])

    if not isinstance(best_model_obj, (BlendingEnsemble, StackingEnsemble)):
        plot_feature_importance(best_model_obj, feature_names)
    
    best_search_obj = all_experiments[best_model_name].get('search_object')
    if best_search_obj:
        analyze_tuning_path(best_model_name, best_search_obj)

    tree_model = all_experiments["lightgbm"]['model_object']

    print(f"🧠 Generating SHAP explanations for lightgbm...")
    try:
        shap_tree_summary(tree_model, X_val_t, title=f"SHAP Impact: LightGBM on Validation Set")
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")

if __name__ == "__main__":
    main()

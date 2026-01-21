import os
import joblib
import time
from sklearn import clone
from src.pre_processing.pre_processing import run_preprocessing_pipeline
from src.models.models import (execute_phase, evaluate_model, display_experiment_summary, run_final_ensemble_test, save_model_artifact, analyze_tuning_path)
from src.models.blending_ensemble import BlendingEnsemble
from src.models.stacking_ensemble import StackingEnsemble
from src.evaluation.evaluation import detect_data_leakage, cross_validation_stability, shap_tree_summary
from src.visualization.models_visualization import (plot_model_comparison, plot_roc_curve, plot_confusion_matrix, plot_feature_importance)

def data_preparation():
    file_path = "outputs/artifacts/processed_bundles.joblib"
    data_bundles = run_preprocessing_pipeline() if not os.path.exists(file_path) else joblib.load(file_path)
    
    X_tr_t, X_val_t, X_te_t, y_tr, y_val, y_te = data_bundles['trees_data']
    X_tr_l, X_val_l, X_te_l, _, _, _ = data_bundles['linear_data']
    feature_names = X_tr_t.columns.tolist()
    
    print("🛡️ Running Pre-flight Leakage Check...")
    detect_data_leakage(X_tr_t, y_tr, task="classification")
    
    return X_tr_t, X_val_t, X_te_t, X_tr_l, X_val_l, X_te_l, y_tr, y_val, y_te, feature_names

def train_base_models(X_tr_t, X_val_t, X_tr_l, X_val_l, y_tr, y_val):    
    tree_keys = ["decision_tree", "random_forest", "adaboost", "gradient_boosting", "xgboost", "lightgbm"]
    linear_keys = ["logistic_regression"]
    
    all_experiments = {}
    all_experiments.update(execute_phase(tree_keys, X_tr_t, y_tr, X_val_t, y_val, "Tree Models"))
    all_experiments.update(execute_phase(linear_keys, X_tr_l, y_tr, X_val_l, y_val, "Linear Models"))
    
    return all_experiments, tree_keys

def stage_3_train_ensembles(all_experiments, tree_keys, X_tr_t, X_tr_l, X_val_t, X_val_l, y_tr, y_val):
    top_3_keys = ['logistic_regression', 'lightgbm', 'random_forest']
    
    # --- Blending ---
    blender = BlendingEnsemble({k: all_experiments[k]['model_object'] for k in top_3_keys}, tree_keys=tree_keys)
    start_blend = time.time()
    blender.fit(X_val_t, X_val_l, y_val)
    all_experiments["blending_ensemble"] = {
        "model_object": blender,
        "metrics": {"val": evaluate_model(blender, (X_val_t, X_val_l), y_val)},
        "train_time": time.time() - start_blend
    }

    # --- Stacking ---
    stacker = StackingEnsemble({k: clone(all_experiments[k]['model_object']) for k in top_3_keys}, n_splits=5, tree_keys=tree_keys)
    start_stack = time.time()
    stacker.fit(X_tr_t, X_tr_l, y_tr)
    all_experiments["stacking_ensemble"] = {
        "model_object": stacker,
        "metrics": {"val": evaluate_model(stacker, (X_val_t, X_val_l), y_val)},
        "train_time": time.time() - start_stack
    }
    
    return all_experiments

def visualizations(all_experiments, X_val_t, X_val_l, y_val, tree_keys):
    plot_model_comparison(all_experiments, split='val')
    plot_roc_curve(all_experiments, X_val_t, X_val_l, y_val, tree_keys)
    
def test_evaluation(all_experiments, X_te_t, X_te_l, y_te, tree_keys):
    display_experiment_summary(all_experiments, split='val')
    
    test_summary = run_final_ensemble_test(all_experiments, X_te_t, X_te_l, y_te, tree_keys)
    best_name = test_summary.index[0]
    save_model_artifact(all_experiments[best_name]['model_object'], f"best_model_{best_name}")
    
    return best_name

def deep_analysis(best_name, all_experiments, X_tr_t, X_val_t, X_val_l, y_tr, y_val, tree_keys, feature_names):
    best_exp = all_experiments[best_name]
    best_obj = best_exp['model_object']

    if isinstance(best_obj, (BlendingEnsemble, StackingEnsemble)):
        X_val_winner = (X_val_t, X_val_l)
    else:
        print(f"⚖️ Checking stability for {best_name}...")
        cross_validation_stability(best_obj, X_tr_t, y_tr)
        X_val_winner = X_val_t if best_name in tree_keys else X_val_l
        plot_feature_importance(best_obj, feature_names)

    plot_confusion_matrix(best_obj, X_val_winner, y_val)
    if best_exp.get('search_object'):
        analyze_tuning_path(best_name, best_exp['search_object'])

    try:
        print(f"🧠 Generating SHAP explanations...")
        shap_tree_summary(all_experiments["lightgbm"]['model_object'], X_val_t)
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")

def main():
    # 1. Data Loading & Preparation
    (X_tr_t, X_val_t, X_te_t, X_tr_l, X_val_l, X_te_l, y_tr, y_val, y_te, feature_names) = data_preparation()

    # 2. Base Models Training
    all_experiments, tree_keys = train_base_models(X_tr_t, X_val_t, X_tr_l, X_val_l, y_tr, y_val)

    # 3. Ensembles
    all_experiments = stage_3_train_ensembles(all_experiments, tree_keys, X_tr_t, X_tr_l, X_val_t, X_val_l, y_tr, y_val)

    # 4. Visualizations
    visualizations(all_experiments, X_val_t, X_val_l, y_val, tree_keys)

    # 5. Final Test
    best_model_name = test_evaluation(all_experiments, X_te_t, X_te_l, y_te, tree_keys)

    # 6. Deep Analysis
    deep_analysis(best_model_name, all_experiments, X_tr_t, X_val_t, X_val_l, y_tr, y_val, tree_keys, feature_names)
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
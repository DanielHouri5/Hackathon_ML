import os
import time
import joblib
import pandas as pd
from src.pre_processing.pre_processing import run_preprocessing_pipeline
from src.models.models import (
    tune_and_report, train_model, evaluate_model, 
    display_experiment_summary, save_model_artifact
)
from src.visualization.visualizations import (
    plot_model_comparison, plot_roc_curve, 
    plot_feature_importance, plot_confusion_matrix, 
    plot_n_trees_vs_f1, plot_correlation_heatmap
)
import src.config as config
from src.models.parm_configs import DEFAULT_PARAMS, PARAM_GRIDS

def main():
    # --- שלב 1: Pre-processing ---
    # נבדוק אם יש כבר נתונים מעובדים כדי לחסוך זמן בהרצות חוזרות
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
            "params": trained_model.get_params(),
            "type": "tree"  # <--- הוסף את השורה הזו
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
            "params": trained_model.get_params(),
            "type": "tree"  # <--- הוסף את השורה הזו
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

    # --- שלב 3: סיכום תוצאות ובחירת מנצח ---
    target_names = data_bundles.get('target_classes', None)
    summary_df = display_experiment_summary(all_experiments, split='val')
    best_model_name = summary_df['f1'].idxmax()
    winner_entry = all_experiments[best_model_name]
    winner_obj = winner_entry['model_object']
    
    print(f"\n🏆 The winner is: {best_model_name} with F1-Score: {winner_entry['metrics']['val']['f1']:.4f}")
    # --- שלב 4: ויזואליזציות ---
    print("\n📊 Generating Visualizations...")

    # 1. השוואת מודלים (Bar Chart)
    plot_model_comparison(all_experiments, split='val')

    # 2. עקומת ROC לכל המודלים
    plot_roc_curve(all_experiments, X_val_le, X_val_oh, y_val, tree_algo_keys)

    # 3. מטריצת בלבול למודל המנצח (על ה-Test Set)
    X_test_final = X_test_le if winner_entry['type'] == 'tree' else X_test_oh
    plot_confusion_matrix(winner_obj, X_test_final, y_test, class_names=target_names)
    

    # 4. חשיבות פיצ'רים למודל המנצח
    feat_names = X_train_le.columns if winner_entry['type'] == 'tree' else X_train_oh.columns
    plot_feature_importance(winner_obj, feat_names)
    

    # 5. גרף השוואת כמות עצים (עבור ה-Random Forest למשל)
    if "random_forest" in all_experiments:
        plot_n_trees_vs_f1("random_forest", X_train_le, y_train, X_val_le, y_val)

    # 6. מפת חום של קורלציות (על ה-Train הראשוני)
    plot_correlation_heatmap(X_train_le)

    # --- שלב 5: שמירת המודל המנצח ---
    save_path = save_model_artifact(winner_obj, f"final_best_model_{best_model_name}")
    print(f"\n✅ Pipeline complete! Best model saved at: {save_path}")

if __name__ == "__main__":
    main()
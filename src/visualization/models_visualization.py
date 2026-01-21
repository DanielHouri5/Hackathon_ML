import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import src.config as config

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

def plot_model_comparison(all_experiments, split='val'):
    set_style()
    
    data_list = []
    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1']
    
    for model_name, results in all_experiments.items():
        model_metrics = results['metrics'][split]
        for m_name in metrics_to_show:
            if m_name in model_metrics:
                data_list.append({
                    'Model': model_name,
                    'Metric': m_name.capitalize(),
                    'Score': model_metrics[m_name]
                })
    
    df_plot = pd.DataFrame(data_list)
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_plot, palette='tab10')
    
    plt.title(f'Model Comparison - Detailed Performance ({split.upper()} Set)', fontsize=15)
    plt.ylim(0, 1.1) 
    plt.ylabel('Score')
    plt.xlabel('Models')
    plt.legend(title='Metrics', loc='upper right')
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=9)

    plt.tight_layout()
    
    save_path = os.path.join(config.MODELS_PLOTS_DIR, 'model_comparison_detailed.png')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(all_experiments, X_val_t, X_val_l, y_val, tree_keys):
    set_style()
    plt.figure(figsize=(10, 8))
    
    for name, exp in all_experiments.items():
        model = exp['model_object']
        
        try:
            if name in ['blending_ensemble', 'stacking_ensemble']:
                y_proba = model.predict_proba(X_val_t, X_val_l)
                if len(y_proba.shape) > 1: y_proba = y_proba[:, 1]
            
            else:
                X_target = X_val_t if name in tree_keys else X_val_l
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_target)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_proba = model.decision_function(X_target)
                else:
                    continue 
            
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            
        except Exception as e:
            print(f"⚠️ Could not plot ROC for {name}: {e}")
            
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc='lower right')
    
    save_path = os.path.join(config.MODELS_PLOTS_DIR, 'roc_curve_all_models.png')
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, top_n=15):
    set_style()
    importance = None
    
    model_name = model.__class__.__name__
    
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_.flatten())

    if importance is not None:
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=fi_df, hue='Feature', palette='magma', legend=False)
        
        plt.title(f'Top {top_n} Feature Importances\n(Model: {model_name})', fontsize=14)
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(config.MODELS_PLOTS_DIR, f'feature_importance_{model_name}.png')
        plt.savefig(save_path)
        plt.close()
    else:
        print(f"The model {model_name} does not support feature importance.")

def plot_confusion_matrix(model, X_test, y_test, class_names=None):
    set_style() 
    model_name = model.__class__.__name__

    if isinstance(X_test, (list, tuple)):
        y_pred = model.predict(*X_test)
    else:
        y_pred = model.predict(X_test)
        
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else True,
                yticklabels=class_names if class_names else True)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix\n(Model: {model_name})')

    save_path = os.path.join(config.MODELS_PLOTS_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(save_path)
    plt.close()

def plot_n_trees_vs_f1(results_df, model_key):
    set_style()
    results_df = results_df.sort_values("n_estimators")
    
    plt.figure(figsize=(10, 6))
    y_col = "f1" if "f1" in results_df.columns else "mean_test_score"
    plt.plot(results_df["n_estimators"], results_df[y_col], marker='o', linestyle='-', color='b')
    
    plt.title(f"Tuning Results: n_estimators vs F1\n(Model: {model_key})")
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("Mean CV F1 Score")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(config.MODELS_PLOTS_DIR, f'tuning_n_trees_{model_key}.png')
    plt.savefig(save_path)
    plt.close()
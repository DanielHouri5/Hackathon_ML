import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import src.config as config

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

def plot_model_comparison(all_experiments, split='val'):
    """גרף השוואת ביצועי מודלים עם מספר מטריקות במקביל"""
    set_style()
    
    # 1. איסוף כל המטריקות מכל המודלים
    data_list = []
    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1']
    
    for model_name, results in all_experiments.items():
        # שליפת המטריקות עבור הספליט הנבחר (בדרך כלל val)
        model_metrics = results['metrics'][split]
        for m_name in metrics_to_show:
            if m_name in model_metrics:
                data_list.append({
                    'Model': model_name,
                    'Metric': m_name.capitalize(),
                    'Score': model_metrics[m_name]
                })
    
    # 2. יצירת DataFrame מתאים לגרף מקובץ
    df_plot = pd.DataFrame(data_list)
    
    # 3. יצירת הגרף
    plt.figure(figsize=(12, 7))
    # השימוש ב-hue='Metric' יוצר את הקבוצות כמו בתמונה
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_plot, palette='tab10')
    
    # 4. עיצוב ותוספות
    plt.title(f'Model Comparison - Detailed Performance ({split.upper()} Set)', fontsize=15)
    plt.ylim(0, 1.1) # כדי שיהיה מקום למקרא (Legend)
    plt.ylabel('Score')
    plt.xlabel('Models')
    plt.legend(title='Metrics', loc='upper right')
    
    # הוספת ערכים מספריים מעל העמודות (אופציונלי אך מומלץ)
    for p in ax.patches:
        if p.get_height() > 0: # מונע כתיבה על עמודות ריקות
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=9)

    plt.tight_layout()
    
    # שמירה אוטומטית (כמו שסיכמנו שיהיה מושלם)
    plt.savefig(config.MODELS_PLOTS_DIR, f'model_comparison_detailed.png')
    plt.show()

def plot_roc_curve(all_experiments, X_val_le, X_val_oh, y_val, tree_models):
    """עקומת ROC לכל המודלים שהורצו"""
    set_style()
    plt.figure()
    
    for name, exp in all_experiments.items():
        model = exp['model_object']
        # בחירת ה-X המתאים לפי סוג המודל
        X_target = X_val_le if name in tree_models else X_val_oh
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_target)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(config.MODELS_PLOTS_DIR, 'roc_curve_all_models.png')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15):
    """גרף חשיבות פיצ'רים הכולל את שם המודל בכותרת"""
    set_style()
    importance = None
    
    # שליפת שם המודל בצורה דינמית
    model_name = model.__class__.__name__
    
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_.flatten())

    if importance is not None:
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        # הוספת hue='Feature' כדי למנוע את ה-Warning ו-legend=False
        sns.barplot(x='Importance', y='Feature', data=fi_df, hue='Feature', palette='magma', legend=False)
        
        # כותרת דינמית עם שם המודל
        plt.title(f'Top {top_n} Feature Importances\n(Model: {model_name})', fontsize=14)
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # שמירה אוטומטית
        plt.savefig(config.MODELS_PLOTS_DIR, f'feature_importance_{model_name}.png')
        plt.show()
    else:
        print(f"The model {model_name} does not support feature importance.")

def plot_confusion_matrix(model, X_test, y_test, class_names=None):
    """מטריצת בלבול מעוצבת"""
    model_name = model.__class__.__name__

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else True,
                yticklabels=class_names if class_names else True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix\n(Model: {model_name})')
    plt.savefig(config.MODELS_PLOTS_DIR, f'confusion_matrix_{model_name}.png')
    plt.show()

def plot_correlation_heatmap(df):
    """מפת חום של קורלציות בעיצוב ריבועי מלא לפי דוגמת המשתמש"""
    set_style()
    
    # בחירת עמודות נומריות בלבד (כמו בתמונה)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(10, 8))
    
    # יצירת המפה
    sns.heatmap(corr, 
                annot=True,          # הצגת ערכים
                fmt=".2g",           # פורמט גנרי (מציג מספרים קטנים כמו בתמונה)
                cmap='coolwarm',     # סקאלת כחול-אדום
                cbar=False,          # הסרת סרגל הצבעים בצד (כמו בתמונה)
                square=True,         # שמירה על ריבועים מושלמים
                annot_kws={"size": 10, "color": "black"}) # טקסט שחור וברור
    
    plt.title('Correlation Heatmap of Numerical Features', fontsize=14, pad=20)
    plt.xticks(rotation=0) # שמות הפיצ'רים בציר X ישרים כמו בתמונה
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # שמירה
    plt.savefig(os.path.join(config.PREPROCESS_PLOTS_DIR, 'correlation_heatmap.png'))
    plt.show()

def plot_n_trees_vs_f1(model_key, X_train, y_train, X_val, y_val, n_trees_list=[10, 50, 100, 200, 500]):
    """גרף השפעת כמות העצים על ביצועי המודל"""
    from src.models.models import build_random_forest, build_xgboost # ייבוא מקומי למניעת circular import
    
    train_scores = []
    val_scores = []
    
    for n in n_trees_list:
        if 'forest' in model_key:
            model = build_random_forest(n_estimators=n, n_jobs=-1)
        elif 'xgboost' in model_key:
            model = build_xgboost(n_estimators=n)
        else:
            print("Only supported for forest/xgboost in this utility.")
            return

        model.fit(X_train, y_train)
        
        from sklearn.metrics import f1_score
        train_scores.append(f1_score(y_train, model.predict(X_train), average='weighted'))
        val_scores.append(f1_score(y_val, model.predict(X_val), average='weighted'))

    plt.figure()
    plt.plot(n_trees_list, train_scores, label='Train F1', marker='o')
    plt.plot(n_trees_list, val_scores, label='Val F1', marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('F1 Score')
    plt.title(f'Number of Trees vs Performance ({model_key})')
    plt.legend()
    plt.savefig(config.MODELS_PLOTS_DIR, f'n_trees_vs_f1_{model_key}.png')
    plt.show()
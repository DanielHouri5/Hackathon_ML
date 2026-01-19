import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

def cross_validation_stability(model, X, y, task="classification", metric="accuracy", n_splits=5):
    """
    כללי להאקתון: מבצע CV ומדפיס דוח ביצועים מעוצב ונוח לקריאה.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    
    y = pd.Series(y) if not isinstance(y, pd.Series) else y

    scores_map = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'rmse': []}

    if task == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
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
            scores_map['rmse'].append(mean_squared_error(y_test, y_pred, squared=False))

    # --- יצירת דוח מעוצב ---
    print("\n" + "="*40)
    print(f"📊 CROSS-VALIDATION REPORT ({task.upper()})")
    print("="*40)

    if task == "classification":
        # סידור המדדים בטבלה קטנה וברורה
        report = {
            "Accuracy":  f"{np.mean(scores_map['accuracy']):.4f} (±{np.std(scores_map['accuracy']):.4f})",
            "Precision": f"{np.mean(scores_map['precision']):.4f}",
            "Recall":    f"{np.mean(scores_map['recall']):.4f}",
            "F1-Score":  f"{np.mean(scores_map['f1']):.4f}",
            "Target Class": f"'{pos_label}' (minority)"
        }
        for k, v in report.items():
            print(f"{k:<15} : {v}")
    else:
        print(f"RMSE (Mean)     : {np.mean(scores_map['rmse']):.4f}")
        print(f"RMSE (Std)      : {np.std(scores_map['rmse']):.4f}")

    print("="*40 + "\n")

    # החזרת המילון המקורי לשימוש בקוד (אם צריך)
    return {k: (np.mean(v) if v else None) for k, v in scores_map.items()}
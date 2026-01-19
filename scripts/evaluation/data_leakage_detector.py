import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def detect_data_leakage(X, y, task="classification"):
    """
    מחשבת MI ומציגה את כל המשתנים המשפיעים (MI > 0).
    משתנים עם 0 נדחפים לסיכום בסוף - אידיאלי לניקוי דאטה בהאקתון.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    results = []
    for col in X.columns:
        try:
            if task == "classification":
                mi_val = mutual_info_classif(X[[col]], y, discrete_features='auto', random_state=42)
            else:
                mi_val = mutual_info_regression(X[[col]], y, random_state=42)
            mi_val = mi_val[0] if len(mi_val) > 0 else 0
        except:
            mi_val = 0
        results.append({'feature': col, 'mutual_info': mi_val})

    df_results = pd.DataFrame(results).sort_values(by='mutual_info', ascending=False)
    
    # פיצול הנתונים: אלו עם השפעה ואלו בלי
    influential_features = df_results[df_results['mutual_info'] > 0]
    zero_features_count = len(df_results[df_results['mutual_info'] <= 0])

    # --- הדפסת דוח מעוצב ---
    print("\n" + "!"*50)
    print(f"🔍 DATA LEAKAGE & FEATURE RELEVANCE ({task.upper()})")
    print("!"*50)
    print(f"{'Feature Name':<35} | {'MI Score':<10}")
    print("-" * 50)

    for _, row in influential_features.iterrows():
        alert = "🚨 HIGH" if row['mutual_info'] > 0.5 else "✅"
        print(f"{row['feature'][:35]:<35} | {row['mutual_info']:.4f}  {alert}")

    if zero_features_count > 0:
        print("-" * 50)
        print(f"... and {zero_features_count} other features with 0.0000 MI score.")
    
    print("!"*50 + "\n")

    return df_results
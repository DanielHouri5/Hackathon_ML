import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def shap_tree_summary(model, X, max_features=10, title="SHAP Summary"):
    """
    Generates SHAP summary plot with proper title handling.
    """
    if isinstance(X, np.ndarray):
        # המרת מערך ל-DataFrame אם צריך
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    # יצירת ה-Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # טיפול במקרה של סיווג (Classification) שבו SHAP מחזיר רשימה לכל מחלקה
    # אנחנו לוקחים את האינדקס 1 (בדרך כלל המחלקה החיובית - הכנסה גבוהה)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # יצירת דמות (Figure) חדשה
    plt.figure(figsize=(12, 6))

    # שים לב: show=False מאפשר לנו להוסיף כותרת לפני שהגרף מוצג
    shap.summary_plot(
        shap_values_to_plot, 
        X, 
        max_display=max_features, 
        show=False
    )

    # הוספת הכותרת עכשיו כשהגרף עדיין "פתוח" בזיכרון
    plt.title(title)
    
    # הצגה סופית
    plt.tight_layout()
    plt.show()
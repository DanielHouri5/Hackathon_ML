# main.py
import pandas as pd
from src.models.ml_models import train_model , print_model_results
import src.pre_processing.pre_processing as pre

if __name__ == "__main__":
    
    X_train_le, X_val_le, X_test_le, y_train, y_val, y_test = pre.run_preprocessing_pipeline()["trees_data"]

    # ----------------------------
    # קריאה לפונקציה עבור מודל Logistic Regression
    # ----------------------------
    results = train_model(
        model_type="logistic_regression",
        X_train_le=X_train_le,
        X_val_le=X_val_le,
        X_test_le=X_test_le,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        task_type="classification",
        save_model_flag=True,
        model_dir="./outputs/trained_models/",
        max_iter=200,           # פרמטרים ספציפיים למודל
        solver='lbfgs'
    )
    print_model_results(results, top_n_features=10)
    
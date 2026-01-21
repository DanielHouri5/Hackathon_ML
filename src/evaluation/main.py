"""
main.py - Hackathon Execution Script
Testing the evaluation utilities on a pre-processed CSV file.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .evaluation import detect_data_leakage, shap_tree_summary, cross_validation_stability
from src.pre_processing.pre_processing import run_preprocessing_pipeline

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing_pipeline()['trees_data']

    task = "classification" 
    
    X_cv = pd.concat([X_train, X_val])
    y_cv = pd.concat([y_train, y_val])

    print("\nStep 1: Analyzing Feature Relevance...")
    detect_data_leakage(X_train, y_train, task=task)

    print("\nStep 2: Checking Model Stability...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cross_validation_stability(model, X_cv, y_cv, task=task)

    print("\nStep 3: Generating SHAP Explanations...")
    model.fit(X_train, y_train)

    shap_tree_summary(model, X_test, max_features=10, title="Impact on Income Prediction")

if __name__ == "__main__":
    main()
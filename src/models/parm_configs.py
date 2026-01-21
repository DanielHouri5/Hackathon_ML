import numpy as np

# ==================== Default Model Parameters ====================
DEFAULT_PARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "C": 1.0,
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "tree_method": 'hist'  
    },
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": 42,
        "verbosity": -1
    },
    "catboost": {
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 6,
        "random_state": 42,
        "verbose": 0
    }
}

# ==================== Hyperparameter Tuning Grids ====================
PARAM_GRIDS = {
    "logistic_regression": {
        'penalty': ['l2'],           # lbfgs תומך רק ב-l2
        'C': [0.1, 1.0, 10.0],       # פחות ערכים = ריצה מהירה יותר
        'solver': ['lbfgs'],         # מוודא שזה מסונכרן עם הבנאי
        'max_iter': [1000]
    },
    "decision_tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    },
    "random_forest": {
        "n_estimators": [100, 200], # לפי הגרף שלך, אין טעם ביותר מ-200
        "max_depth": [10, 15, 20],   # מגביל את ה-Overfitting
        "min_samples_leaf": [5, 10], # מונע מהמודל להיות ספציפי מדי
        "max_features": ["sqrt"],    # גורם לעצים להיות שונים זה מזה
        "bootstrap": [True]
    },
    "adaboost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0]
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 8],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 5, 10]
    },
    "xgboost": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    },
    "lightgbm": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "feature_fraction": [0.7, 0.8, 0.9]
    },
    "catboost": {
        "iterations": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7]
    }
}
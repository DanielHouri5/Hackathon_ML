"""
Hyperparameter Tuning - Fast Tuning Script
RandomizedSearchCV for efficient hyperparameter search
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

def get_param_grid_minimal(model_type: str, task_type: str = "classification") -> dict:
    """
    Minimal param grid for core ML models, suitable for a course or hackathon.
    Only the most relevant parameters for tuning manually.
    """
    
    grids = {
        'logreg': {
            'C': [0.1, 1.0, 10],
            'max_iter': [100, 200]
        },
        'dt': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        },
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        },
        'xgb': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        },
        'lgbm': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, -1],
            'learning_rate': [0.05, 0.1]
        },
        'cat': {
            'iterations': [100, 200],
            'depth': [4, 6],
            'learning_rate': [0.05, 0.1]
        }
    }

    grid = grids.get(model_type, {})

    # Add regression objective if needed
    if task_type == "regression":
        if model_type == "xgb":
            grid['objective'] = ['reg:squarederror']
        elif model_type == "lgbm":
            grid['objective'] = ['regression']
        elif model_type == "cat":
            grid['loss_function'] = ['RMSE']
        elif model_type == "logreg":
            grid = {}  # LogisticRegression לא רלוונטי לרגרסיה

    return grid


def tune_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    scoring: str = 'accuracy',
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Tune a specific model using RandomizedSearchCV
    
    Parameters:
    -----------
    model_type : str
        Model type: 'rf', 'xgb', 'lgbm', 'cat'
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_iter : int
        Number of iterations for search
    cv : int
        Number of CV folds
    scoring : str
        Scoring metric
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print messages
        
    Returns:
    --------
    Dict[str, Any] : Tuning results with best model and parameters
    """
    if verbose:
        print(f"🔍 Searching hyperparameters for {model_type.upper()}...")
        print(f"   Iterations: {n_iter}, CV: {cv}, Scoring: {scoring}\n")
    
    # Select the model
    if model_type == 'rf':
        base_model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'xgb':
        base_model = XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
    elif model_type == 'lgbm':
        base_model = LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
    elif model_type == 'cat':
        base_model = CatBoostClassifier(
            random_state=random_state,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False
        )
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    
    # Get param grid
    param_grid = get_param_grid(model_type)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    # Search
    random_search.fit(X_train, y_train)
    
    # Return results
    results = {
        'best_score': random_search.best_score_,
        'best_params': random_search.best_params_,
        'best_model': random_search.best_estimator_,
        'cv_results': random_search.cv_results_
    }
    
    if verbose:
        print(f"\n✅ Tuning completed!")
        print(f"🏆 Best Score: {random_search.best_score_:.4f}")
        print(f"📋 Best Parameters:")
        for param, value in random_search.best_params_.items():
            print(f"   ├─ {param}: {value}")
        print()
    
    return results


def tune_all_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Tune all boosting models
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_iter : int
        Number of iterations per model
    cv : int
        Number of CV folds
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print messages
        
    Returns:
    --------
    Dict[str, Dict[str, Any]] : Results for all models
    """
    if verbose:
        print("=" * 60)
        print("🔥 Tuning the Iron Trinity")
        print("=" * 60 + "\n")
    
    models = ['xgb', 'lgbm', 'cat']
    results = {}
    
    for model_type in models:
        results[model_type] = tune_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            verbose=verbose
        )
        
        if verbose:
            print("-" * 60 + "\n")
    
    if verbose:
        print("=" * 60)
        print("📊 Results Summary")
        print("=" * 60)
        
        for model_type in models:
            score = results[model_type]['best_score']
            print(f"{model_type.upper():8s} - Best Score: {score:.4f}")
        
        best_model = max(
            results.items(), 
            key=lambda x: x[1]['best_score']
        )
        print(f"\n🏆 Best model: {best_model[0].upper()}")
        print(f"   Score: {best_model[1]['best_score']:.4f}\n")
    
    return results


def export_tuning_results(
    results: Dict[str, Dict[str, Any]],
    filename: str = 'best_params.txt'
) -> None:
    """
    Export best parameters to file
    
    Parameters:
    -----------
    results : Dict[str, Dict[str, Any]]
        Results from tune_model or tune_all_boosting
    filename : str
        Filename to save to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Best Hyperparameters\n")
        f.write("=" * 60 + "\n\n")
        
        for model_type, res in results.items():
            score = res['best_score']
            params = res['best_params']
            
            f.write(f"{model_type.upper()}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write("-" * 40 + "\n")
            
            for param, value in params.items():
                f.write(f"{param}: {value}\n")
            
            f.write("\n")
    
    print(f"💾 Parameters saved to {filename}")

# Quick Tuning Templates - Fast tuning functions

def quick_tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Quick tuning for XGBoost
    """
    return tune_model('xgb', X_train, y_train, n_iter=n_iter, cv=cv, random_state=random_state)


def quick_tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Quick tuning for LightGBM
    """
    return tune_model('lgbm', X_train, y_train, n_iter=n_iter, cv=cv, random_state=random_state)


def quick_tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Quick tuning for CatBoost
    """
    return tune_model('cat', X_train, y_train, n_iter=n_iter, cv=cv, random_state=random_state)


if __name__ == "__main__":
    # Usage example
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Quick tuning for XGBoost
    print("Example: Quick XGBoost Tuning\n")
    results = quick_tune_xgboost(X_df, y_series, n_iter=20, cv=3)
    
    # Or tune all models
    print("\n" + "=" * 60)
    print("Example: Tuning All Models")
    print("=" * 60 + "\n")
    
    all_results = tune_all_boosting(
        X_df, y_series, 
        n_iter=20,  # Fewer iterations for example
        cv=3  # Fewer folds for example
    )
    
    # Save parameters
    export_tuning_results(all_results, 'best_params_example.txt')

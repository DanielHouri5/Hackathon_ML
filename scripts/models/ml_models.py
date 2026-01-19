"""
ML Models - Complete Machine Learning Pipeline
All-in-one module for training and ensemble creation
Includes: Baseline Models, Boosting Models, Ensemble Methods, Model Factory
Note: Hyperparameter tuning is in a separate file (hyperparameter_tuning.py)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BASELINE MODELS - Basic Classification Models
# ============================================================================

# Initialize baseline classification models with default parameters
def create_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create all baseline models with reasonable parameters
    
    Parameters:
    -----------
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, Any] : Dictionary of model instances
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


# Train a single model with cross-validation evaluation
def train_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a single model and evaluate with cross-validation
    
    Parameters:
    -----------
    model : Any
        The model instance to train
    model_name : str
        Name of the model for display
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv : int
        Number of cross-validation folds
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    Dict[str, Any] : Dictionary with model and scores
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate with Cross-Validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    if verbose:
        print(f"\n📊 {model_name}")
        print(f"   ├─ Mean CV Score: {mean_score:.4f}")
        print(f"   ├─ Std CV Score:  {std_score:.4f}")
        print(f"   └─ CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    return {
        'model': model,
        'cv_mean': mean_score,
        'cv_std': std_score,
        'cv_scores': cv_scores
    }


# Train all baseline models and return comprehensive results
def train_all_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train all baseline models
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    Dict[str, Dict[str, Any]] : Results for all models (mean CV score)
    """
    if verbose:
        print("🚀 Starting baseline model training...\n")
        print("=" * 60)
    
    # Create models
    models = create_baseline_models(random_state)
    
    # Train all models
    results = {}
    for name, model in models.items():
        results[name] = train_model(model, name, X_train, y_train, cv, verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ Training completed!\n")
        
        # Show best model
        best_model = max(results.items(), key=lambda x: x[1]['cv_mean'])
        print(f"🏆 Best model: {best_model[0]}")
        print(f"   Score: {best_model[1]['cv_mean']:.4f}\n")
    
    return results


# Get the best performing baseline model from results
def get_best_baseline_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Any]:
    """
    Get the best performing baseline model
    
    Parameters:
    -----------
    results : Dict[str, Dict[str, Any]]
        Results from train_all_baseline_models
        
    Returns:
    --------
    Tuple[str, Any] : Model name and model instance
    """
    best_name = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
    return best_name, results[best_name]['model']


# Make predictions using a specific baseline model
def predict_with_baseline_model(
    results: Dict[str, Dict[str, Any]],
    model_name: str,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions with a specific baseline model
    
    Parameters:
    -----------
    results : Dict[str, Dict[str, Any]]
        Results from train_all_baseline_models
    model_name : str
        Name of the model
    X : pd.DataFrame
        Data for prediction
        
    Returns:
    --------
    np.ndarray : Predictions
    """
    if model_name not in results:
        raise ValueError(f"Model {model_name} does not exist or was not trained")
    
    return results[model_name]['model'].predict(X)


# ============================================================================
# BOOSTING MODELS - The Iron Trinity (XGBoost, LightGBM, CatBoost)
# ============================================================================

# Initialize all boosting models (XGBoost, LightGBM, CatBoost) with optimized parameters
def create_boosting_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initialize the Iron Trinity with strong hyperparameters
    
    Parameters:
    -----------
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, Any] : Dictionary of boosting model instances
    """
    models = {}
    
    # XGBoost - fast and powerful
    models['XGBoost'] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=random_state,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='logloss'
    )
    
    # LightGBM - especially fast and memory efficient
    models['LightGBM'] = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        num_leaves=31
    )
    
    # CatBoost - excellent handling of categorical features
    models['CatBoost'] = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bylevel=0.8,
        min_data_in_leaf=20,
        l2_leaf_reg=3,
        random_state=random_state,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False
    )
    
    return models


# Train a single boosting model with cross-validation
def train_boosting_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: Optional[list] = None,
    cv: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a single boosting model and evaluate with cross-validation
    
    Parameters:
    -----------
    model : Any
        The model instance to train
    model_name : str
        Name of the model for display
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cat_features : list, optional
        List of categorical columns (important for CatBoost)
    cv : int
        Number of cross-validation folds
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    Dict[str, Any] : Dictionary with model and scores
    """
    if verbose:
        print(f"\n⚡ Training {model_name}...")
    
    # Special handling for CatBoost with categorical features
    if model_name == 'CatBoost' and cat_features:
        model.set_params(cat_features=cat_features)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate with Cross-Validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    if verbose:
        print(f"   ├─ Mean CV Score: {mean_score:.4f}")
        print(f"   ├─ Std CV Score:  {std_score:.4f}")
        print(f"   └─ CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    return {
        'model': model,
        'cv_mean': mean_score,
        'cv_std': std_score,
        'cv_scores': cv_scores
    }


# Train all boosting models (The Iron Trinity) and return results
def train_all_boosting_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: Optional[list] = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train all boosting models
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cat_features : list, optional
        List of categorical columns (important for CatBoost)
    cv : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    Dict[str, Dict[str, Any]] : Results for all models
    """
    if verbose:
        print("🔥 Starting Iron Trinity training...\n")
        print("=" * 60)
    
    # Create models
    models = create_boosting_models(random_state)
    
    # Train all models
    results = {}
    for name, model in models.items():
        results[name] = train_boosting_model(
            model, name, X_train, y_train, cat_features, cv, verbose
        )
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ Training completed!\n")
        
        # Show best model
        best_model = max(results.items(), key=lambda x: x[1]['cv_mean'])
        print(f"🏆 Best model: {best_model[0]}")
        print(f"   Score: {best_model[1]['cv_mean']:.4f}\n")
    
    return results


# Get the best performing boosting model from results
def get_best_boosting_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Any]:
    """
    Get the best performing boosting model
    
    Parameters:
    -----------
    results : Dict[str, Dict[str, Any]]
        Results from train_all_boosting_models
        
    Returns:
    --------
    Tuple[str, Any] : Model name and model instance
    """
    best_name = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
    return best_name, results[best_name]['model']


# Make predictions using a specific boosting model
def predict_with_boosting_model(
    results: Dict[str, Dict[str, Any]],
    model_name: str,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions with a specific boosting model
    
    Parameters:
    -----------
    results : Dict[str, Dict[str, Any]]
        Results from train_all_boosting_models
    model_name : str
        Name of the model
    X : pd.DataFrame
        Data for prediction
        
    Returns:
    --------
    np.ndarray : Predictions
    """
    if model_name not in results:
        raise ValueError(f"Model {model_name} does not exist or was not trained")
    
    return results[model_name]['model'].predict(X)


# Extract and rank feature importance from a boosting model
def get_feature_importance(
    results: Dict[str, Dict[str, Any]],
    model_name: str,
    feature_names: Optional[list] = None,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get feature importance for a specific boosting model
    
    Parameters:
    -----------
    results : Dict[str, Dict[str, Any]]
        Results from train_all_boosting_models
    model_name : str
        Name of the model
    feature_names : list, optional
        Feature names
    top_n : int
        How many features to display
        
    Returns:
    --------
    pd.DataFrame : Feature importance table
    """
    if model_name not in results:
        raise ValueError(f"Model {model_name} does not exist or was not trained")
    
    model = results[model_name]['model']
    importance = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance))]
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    df = df.sort_values('Importance', ascending=False).head(top_n)
    return df.reset_index(drop=True)


# ============================================================================
# ENSEMBLE MODELS - Stacking & Weighted Average
# ============================================================================

# Create a stacking ensemble classifier with base models and meta-learner
def create_stacking_ensemble(
    base_models: List[Tuple[str, Any]],
    meta_model: Optional[Any] = None,
    cv: int = 5,
    random_state: int = 42
) -> StackingClassifier:
    """
    Create a Stacking Ensemble with base models and meta-learner
    
    Parameters:
    -----------
    base_models : List[Tuple[str, Any]]
        List of base models [(name, model), ...]
    meta_model : Any, optional
        Meta-learner (default: Logistic Regression)
    cv : int
        Number of folds for training the meta-model
    random_state : int
        Random seed
        
    Returns:
    --------
    StackingClassifier : The stacking ensemble instance
    """
    # Default meta-model: Logistic Regression
    if meta_model is None:
        meta_model = LogisticRegression(
            random_state=random_state,
            max_iter=1000
        )
    
    # Create StackingClassifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv,
        n_jobs=-1
    )
    
    return stacking_clf


# Train a stacking ensemble and evaluate performance
def train_stacking_ensemble(
    base_models: List[Tuple[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_model: Optional[Any] = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a Stacking Ensemble
    
    Parameters:
    -----------
    base_models : List[Tuple[str, Any]]
        List of base models [(name, model), ...]
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    meta_model : Any, optional
        Meta-learner (default: Logistic Regression)
    cv : int
        Number of folds
    random_state : int
        Random seed
    verbose : bool
        Whether to print messages
        
    Returns:
    --------
    Dict[str, Any] : Dictionary containing the trained model and evaluation results
    """
    if verbose:
        print("🏗️  Training Stacking Ensemble...")
        print(f"   Base Models: {[name for name, _ in base_models]}")
        print(f"   Meta Model: {type(meta_model).__name__ if meta_model else 'LogisticRegression'}")
        print(f"   CV Folds: {cv}\n")
    
    # Create stacking ensemble
    stacking_clf = create_stacking_ensemble(base_models, meta_model, cv, random_state)
    
    # Train the ensemble
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate with cross-validation
    cv_scores = cross_val_score(
        stacking_clf,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    if verbose:
        print("✅ Training completed!")
        print(f"📊 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return {
        'model': stacking_clf,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }


# Make predictions using a trained stacking ensemble
def predict_with_stacking(
    stacking_result: Dict[str, Any],
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions with Stacking Ensemble
    
    Parameters:
    -----------
    stacking_result : Dict[str, Any]
        Result from train_stacking_ensemble
    X : pd.DataFrame
        Data for prediction
        
    Returns:
    --------
    np.ndarray : Predictions
    """
    return stacking_result['model'].predict(X)


# Create and train a Blending Ensemble (Holdout-based ensemble)
def train_blending_ensemble(
    base_models: List[Tuple[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_model: Optional[Any] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a Blending Ensemble (holdout-based approach)
    
    Blending vs Stacking:
    - Blending: Uses holdout validation set for meta-model training (faster, simpler)
    - Stacking: Uses cross-validation for meta-model training (more robust, slower)
    
    Parameters:
    -----------
    base_models : List[Tuple[str, Any]]
        List of base models [(name, model), ...]
    X_train : pd.DataFrame
        Training features for base models
    y_train : pd.Series
        Training labels for base models
    X_val : pd.DataFrame
        Validation features for meta-model training
    y_val : pd.Series
        Validation labels for meta-model training
    meta_model : Any, optional
        Meta-learner (default: Logistic Regression)
    random_state : int
        Random seed
    verbose : bool
        Whether to print messages
        
    Returns:
    --------
    Dict[str, Any] : Dictionary containing trained models and meta-features
    """
    if verbose:
        print("🔀 Training Blending Ensemble...")
        print(f"   Base Models: {[name for name, _ in base_models]}")
        print(f"   Meta Model: {type(meta_model).__name__ if meta_model else 'LogisticRegression'}")
        print(f"   Train size: {len(X_train)}, Validation size: {len(X_val)}\n")
    
    # Default meta-model: Logistic Regression
    if meta_model is None:
        meta_model = LogisticRegression(
            random_state=random_state,
            max_iter=1000
        )
    
    # Step 1: Train base models on training set
    trained_base_models = []
    meta_features_val = []
    
    for name, model in base_models:
        if verbose:
            print(f"   🔹 Training {name} on training set...")
        
        # Train on training set
        model.fit(X_train, y_train)
        trained_base_models.append((name, model))
        
        # Predict on validation set to create meta-features
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val)
        else:
            pred = model.predict(X_val).reshape(-1, 1)
        
        meta_features_val.append(pred)
    
    # Step 2: Create meta-features for validation set
    if hasattr(trained_base_models[0][1], 'predict_proba'):
        # For probability predictions, stack all predictions
        meta_X_val = np.hstack(meta_features_val)
    else:
        # For simple predictions, stack as columns
        meta_X_val = np.column_stack(meta_features_val)
    
    # Step 3: Train meta-model on validation meta-features
    if verbose:
        print(f"\n   🎯 Training meta-model on validation predictions...")
    
    meta_model.fit(meta_X_val, y_val)
    
    # Evaluate on validation set
    from sklearn.metrics import accuracy_score
    meta_pred = meta_model.predict(meta_X_val)
    val_accuracy = accuracy_score(y_val, meta_pred)
    
    if verbose:
        print(f"\n✅ Blending completed!")
        print(f"📊 Validation Accuracy: {val_accuracy:.4f}\n")
    
    return {
        'base_models': trained_base_models,
        'meta_model': meta_model,
        'val_accuracy': val_accuracy,
        'meta_features_shape': meta_X_val.shape
    }


# Make predictions with Blending Ensemble
def predict_with_blending(
    blending_result: Dict[str, Any],
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions with Blending Ensemble
    
    Parameters:
    -----------
    blending_result : Dict[str, Any]
        Result from train_blending_ensemble
    X : pd.DataFrame
        Data for prediction
        
    Returns:
    --------
    np.ndarray : Predictions
    """
    base_models = blending_result['base_models']
    meta_model = blending_result['meta_model']
    
    # Step 1: Get predictions from base models
    meta_features = []
    for name, model in base_models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X)
        else:
            pred = model.predict(X).reshape(-1, 1)
        meta_features.append(pred)
    
    # Step 2: Stack meta-features
    if hasattr(base_models[0][1], 'predict_proba'):
        meta_X = np.hstack(meta_features)
    else:
        meta_X = np.column_stack(meta_features)
    
    # Step 3: Predict with meta-model
    return meta_model.predict(meta_X)


# Create a weighted average ensemble with custom or equal weights
def create_weighted_average_ensemble(
    models: List[Tuple[str, Any]],
    weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Create a Weighted Average Ensemble - fast manual weighting
    Perfect when there's no time for full Stacking
    
    Parameters:
    -----------
    models : List[Tuple[str, Any]]
        List of models [(name, model), ...]
    weights : List[float], optional
        Weights for models (default: equal weights)
        
    Returns:
    --------
    Dict[str, Any] : Dictionary with models and weights
    """
    # If no weights provided, use equal weights
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]
    
    return {
        'models': models,
        'weights': weights,
        'is_fitted': False
    }


# Train all models in a weighted average ensemble
def train_weighted_average_ensemble(
    models: List[Tuple[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: Optional[List[float]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train all models in Weighted Average Ensemble
    
    Parameters:
    -----------
    models : List[Tuple[str, Any]]
        List of models [(name, model), ...]
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    weights : List[float], optional
        Weights for models (default: equal weights)
    verbose : bool
        Whether to print messages
        
    Returns:
    --------
    Dict[str, Any] : Dictionary with trained models and weights
    """
    ensemble = create_weighted_average_ensemble(models, weights)
    
    if verbose:
        print("⚖️  Training Weighted Average Ensemble...")
        print(f"   Models: {[name for name, _ in models]}")
        print(f"   Weights: {[f'{w:.3f}' for w in ensemble['weights']]}\n")
    
    for name, model in ensemble['models']:
        if verbose:
            print(f"   🔹 Training {name}...")
        model.fit(X_train, y_train)
    
    ensemble['is_fitted'] = True
    
    if verbose:
        print("\n✅ Training completed!")
    
    return ensemble


# Predict class probabilities using weighted average of multiple models
def predict_proba_weighted(
    ensemble: Dict[str, Any],
    X: pd.DataFrame
) -> np.ndarray:
    """
    Predict weighted probabilities
    
    Parameters:
    -----------
    ensemble : Dict[str, Any]
        Result from train_weighted_average_ensemble
    X : pd.DataFrame
        Data for prediction
        
    Returns:
    --------
    np.ndarray : Weighted probabilities
    """
    if not ensemble['is_fitted']:
        raise ValueError("Models not trained yet! Call train_weighted_average_ensemble first.")
    
    # Calculate weighted probabilities
    weighted_proba = None
    
    for (name, model), weight in zip(ensemble['models'], ensemble['weights']):
        proba = model.predict_proba(X)
        
        if weighted_proba is None:
            weighted_proba = weight * proba
        else:
            weighted_proba += weight * proba
    
    return weighted_proba


# Make class predictions using weighted average ensemble
def predict_weighted(
    ensemble: Dict[str, Any],
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions with Weighted Average
    
    Parameters:
    -----------
    ensemble : Dict[str, Any]
        Result from train_weighted_average_ensemble
    X : pd.DataFrame
        Data for prediction
        
    Returns:
    --------
    np.ndarray : Predictions
    """
    proba = predict_proba_weighted(ensemble, X)
    return np.argmax(proba, axis=1)


# Optimize ensemble weights based on cross-validation performance
def optimize_ensemble_weights(
    ensemble: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    verbose: bool = True
) -> List[float]:
    """
    Simple weight optimization based on CV performance
    
    Parameters:
    -----------
    ensemble : Dict[str, Any]
        Ensemble dictionary with models
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    List[float] : Updated weights
    """
    if verbose:
        print("🎯 Optimizing weights based on CV performance...\n")
    
    # Calculate CV performance for each model
    scores = []
    for name, model in ensemble['models']:
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        mean_score = cv_scores.mean()
        scores.append(mean_score)
        
        if verbose:
            print(f"   {name}: {mean_score:.4f}")
    
    # Convert scores to weights (higher score = higher weight)
    # Using softmax method
    scores_array = np.array(scores)
    exp_scores = np.exp(scores_array - np.max(scores_array))
    new_weights = exp_scores / exp_scores.sum()
    
    ensemble['weights'] = new_weights.tolist()
    
    if verbose:
        print(f"\n📊 Updated weights: {[f'{w:.3f}' for w in ensemble['weights']]}")
    
    return ensemble['weights']


# ============================================================================
# MODEL FACTORY - Central orchestration module for all ML operations
# ============================================================================

# Train all models (baseline and boosting) in one function call
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: Optional[list] = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train all models - Baselines + Boosting
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cat_features : list, optional
        Categorical columns
    cv : int
        Number of CV folds
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print messages
        
    Returns:
    --------
    Dict[str, Any] : Dictionary with all results and trained models
    """
    if verbose:
        print("🏭 ML Pipeline - Full Training\n")
        print("=" * 60)
    
    results = {}
    
    # Train Baselines
    if verbose:
        print("\n📍 Step 1: Baseline Models")
        print("-" * 60)
    
    baseline_results = train_all_baseline_models(
        X_train, y_train, cv=cv, random_state=random_state, verbose=verbose
    )
    results['baseline_results'] = baseline_results
    results['baseline_df'] = get_baseline_results_dataframe(baseline_results)
    
    # Train Boosting
    if verbose:
        print("\n" + "=" * 60)
        print("📍 Step 2: Boosting Models")
        print("-" * 60)
    
    boosting_results = train_all_boosting_models(
        X_train, y_train, 
        cat_features=cat_features,
        cv=cv,
        random_state=random_state,
        verbose=verbose
    )
    results['boosting_results'] = boosting_results
    results['boosting_df'] = get_boosting_results_dataframe(boosting_results)
    
    # Combine results
    all_df = pd.concat([
        results['baseline_df'],
        results['boosting_df']
    ], ignore_index=True)
    all_df = all_df.sort_values('Mean CV Score', ascending=False).reset_index(drop=True)
    results['all_df'] = all_df
    
    if verbose:
        print("\n" + "=" * 60)
        print("📊 All Models Summary")
        print("=" * 60)
        print(all_df.to_string(index=False))
        print()
    
    return results


# Create and train ensemble models (stacking and weighted average)
def create_ensemble_models(
    baseline_results: Dict[str, Dict[str, Any]],
    boosting_results: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
    optimize_weights: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Create ensemble models (Stacking and Weighted Average)
    
    Parameters:
    -----------
    baseline_results : Dict
        Results from train_all_baseline_models
    boosting_results : Dict
        Results from train_all_boosting_models
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv : int
        Number of CV folds
    random_state : int
        Random seed
    optimize_weights : bool
        Whether to optimize weights for weighted ensemble
    verbose : bool
        Whether to print messages
        
    Returns:
    --------
    Dict[str, Any] : Dictionary with ensemble models and results
    """
    ensemble_results = {}
    
    # Prepare base models from boosting
    base_models = [
        (name, result['model'])
        for name, result in boosting_results.items()
    ]
    
    if verbose:
        print("\n" + "=" * 60)
        print("📍 Creating Ensemble Models")
        print("=" * 60)
    
    # Create Stacking
    if verbose:
        print("\n🏗️  Creating Stacking Ensemble")
    
    stacking_result = train_stacking_ensemble(
        base_models, X_train, y_train,
        cv=cv, random_state=random_state, verbose=verbose
    )
    ensemble_results['stacking'] = stacking_result
    
    # Create Weighted Average
    if verbose:
        print("\n⚖️  Creating Weighted Average Ensemble")
    
    weighted_result = train_weighted_average_ensemble(
        base_models, X_train, y_train, verbose=verbose
    )
    
    if optimize_weights:
        optimize_ensemble_weights(weighted_result, X_train, y_train, verbose=verbose)
    
    ensemble_results['weighted'] = weighted_result
    
    return ensemble_results


# Find the best performing model across all trained models
def get_best_model(
    baseline_results: Dict[str, Dict[str, Any]],
    boosting_results: Dict[str, Dict[str, Any]],
    ensemble_results: Optional[Dict[str, Any]] = None
) -> Tuple[str, Any, float]:
    """
    Get the best performing model from all trained models
    
    Parameters:
    -----------
    baseline_results : Dict
        Results from baseline models
    boosting_results : Dict
        Results from boosting models
    ensemble_results : Dict, optional
        Results from ensemble models
        
    Returns:
    --------
    Tuple[str, Any, float] : Model name, model object, and score
    """
    all_scores = {}
    all_models = {}
    
    # Baseline scores
    for name, result in baseline_results.items():
        all_scores[name] = result['cv_mean']
        all_models[name] = result['model']
    
    # Boosting scores
    for name, result in boosting_results.items():
        all_scores[name] = result['cv_mean']
        all_models[name] = result['model']
    
    # Ensemble scores
    if ensemble_results:
        if 'stacking' in ensemble_results:
            all_scores['Stacking'] = ensemble_results['stacking']['cv_mean']
            all_models['Stacking'] = ensemble_results['stacking']['model']
    
    # Find best
    best_name = max(all_scores.items(), key=lambda x: x[1])[0]
    best_score = all_scores[best_name]
    best_model = all_models[best_name]
    
    return best_name, best_model, best_score


# Make predictions using the best model or a specified model
def predict_with_best_model(
    baseline_results: Dict[str, Dict[str, Any]],
    boosting_results: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    ensemble_results: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> np.ndarray:
    """
    Make predictions with best model or specified model
    
    Parameters:
    -----------
    baseline_results : Dict
        Results from baseline models
    boosting_results : Dict
        Results from boosting models
    X : pd.DataFrame
        Data for prediction
    ensemble_results : Dict, optional
        Results from ensemble models
    model_name : str, optional
        Specific model name (default: best model)
        
    Returns:
    --------
    np.ndarray : Predictions
    """
    if model_name is None:
        model_name, model, _ = get_best_model(
            baseline_results, boosting_results, ensemble_results
        )
    else:
        # Find specified model
        if model_name == 'Stacking' and ensemble_results:
            return predict_with_stacking(ensemble_results['stacking'], X)
        elif model_name == 'Weighted' and ensemble_results:
            return predict_weighted(ensemble_results['weighted'], X)
        elif model_name in baseline_results:
            model = baseline_results[model_name]['model']
        elif model_name in boosting_results:
            model = boosting_results[model_name]['model']
        else:
            raise ValueError(f"Model {model_name} not found")
    
    return model.predict(X)


# Save a comprehensive summary of all model results to a text file
def save_results_summary(
    baseline_results: Dict[str, Dict[str, Any]],
    boosting_results: Dict[str, Dict[str, Any]],
    ensemble_results: Optional[Dict[str, Any]] = None,
    filename: str = 'model_summary.txt'
) -> None:
    """
    Save summary of all results to file
    
    Parameters:
    -----------
    baseline_results : Dict
        Results from baseline models
    boosting_results : Dict
        Results from boosting models
    ensemble_results : Dict, optional
        Results from ensemble models
    filename : str
        Filename to save to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ML Pipeline - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Baseline Models:\n")
        f.write("-" * 40 + "\n")
        for name, result in baseline_results.items():
            score = result['cv_mean']
            f.write(f"{name:25s} {score:.4f}\n")
        f.write("\n")
        
        f.write("Boosting Models:\n")
        f.write("-" * 40 + "\n")
        for name, result in boosting_results.items():
            score = result['cv_mean']
            f.write(f"{name:25s} {score:.4f}\n")
        f.write("\n")
        
        if ensemble_results:
            f.write("Ensemble Models:\n")
            f.write("-" * 40 + "\n")
            if 'stacking' in ensemble_results:
                score = ensemble_results['stacking']['cv_mean']
                f.write(f"{'Stacking':25s} {score:.4f}\n")
            f.write("\n")
        
        # Best model
        best_name, _, best_score = get_best_model(
            baseline_results, boosting_results, ensemble_results
        )
        f.write("=" * 60 + "\n")
        f.write(f"🏆 Best Model: {best_name}\n")
        f.write(f"   Score: {best_score:.4f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"💾 Summary saved to {filename}")


# ============================================================================
# EVALUATION - Training Results Evaluation
# ============================================================================

# Evaluate and display training results for all models
def evaluate_training_results(
    baseline_results: Dict[str, Dict[str, Any]],
    boosting_results: Dict[str, Dict[str, Any]],
    ensemble_results: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Evaluate and display training results (CV scores) for all models
    
    Parameters:
    -----------
    baseline_results : Dict
        Results from train_all_baseline_models
    boosting_results : Dict
        Results from train_all_boosting_models
    ensemble_results : Dict, optional
        Results from ensemble models
        
    Returns:
    --------
    pd.DataFrame : Training evaluation results sorted by CV score
    """
    print("\n" + "=" * 70)
    print("📊 Training Results Evaluation (Cross-Validation Scores)")
    print("=" * 70 + "\n")
    
    data = []
    
    # Baseline models
    print("📍 Baseline Models:")
    print("-" * 70)
    for name, result in baseline_results.items():
        cv_mean = result['cv_mean']
        cv_std = result['cv_std']
        data.append({
            'Model': name,
            'CV Mean Score': cv_mean,
            'CV Std': cv_std,
            'Type': 'Baseline'
        })
        print(f"  {name:25s} → CV: {cv_mean:.4f} (±{cv_std:.4f})")
    
    # Boosting models
    print("\n📍 Boosting Models:")
    print("-" * 70)
    for name, result in boosting_results.items():
        cv_mean = result['cv_mean']
        cv_std = result['cv_std']
        data.append({
            'Model': name,
            'CV Mean Score': cv_mean,
            'CV Std': cv_std,
            'Type': 'Boosting'
        })
        print(f"  {name:25s} → CV: {cv_mean:.4f} (±{cv_std:.4f})")
    
    # Ensemble models
    if ensemble_results:
        print("\n📍 Ensemble Models:")
        print("-" * 70)
        if 'stacking' in ensemble_results:
            cv_mean = ensemble_results['stacking']['cv_mean']
            cv_std = ensemble_results['stacking']['cv_std']
            data.append({
                'Model': 'Stacking',
                'CV Mean Score': cv_mean,
                'CV Std': cv_std,
                'Type': 'Ensemble'
            })
            print(f"  {'Stacking':25s} → CV: {cv_mean:.4f} (±{cv_std:.4f})")
    
    # Create DataFrame and sort
    df = pd.DataFrame(data)
    df = df.sort_values('CV Mean Score', ascending=False).reset_index(drop=True)
    
    # Display summary table
    print("\n" + "=" * 70)
    print("📋 Summary Table (Sorted by CV Score):")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Best model
    best = df.iloc[0]
    print("\n" + "=" * 70)
    print(f"🏆 Best Model: {best['Model']}")
    print(f"   CV Score: {best['CV Mean Score']:.4f} (±{best['CV Std']:.4f})")
    print(f"   Type: {best['Type']}")
    print("=" * 70 + "\n")
    
    return df


# ============================================================================
# MAIN - Usage Examples
# ============================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("=" * 70)
    print("🏭 ML Pipeline - Complete Example")
    print("=" * 70 + "\n")
    
    # Load sample data
    data = load_breast_cancer()
    X_df = pd.DataFrame(data.data, columns=data.feature_names)
    y_series = pd.Series(data.target, name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    
    # Step 1: Train all models
    print("📍 Step 1: Training all models")
    all_results = train_all_models(X_train, y_train, cv=5)
    
    # Step 2: Create ensembles
    print("\n📍 Step 2: Creating ensemble models")
    ensemble_results = create_ensemble_models(
        all_results['baseline_results'],
        all_results['boosting_results'],
        X_train, y_train, cv=5
    )
    
    # Step 3: Get best model
    print("\n" + "=" * 70)
    print("📍 Step 3: Selecting best model")
    best_name, best_model, best_score = get_best_model(
        all_results['baseline_results'],
        all_results['boosting_results'],
        ensemble_results
    )
    print(f"🎯 Best Model: {best_name}")
    print(f"   CV Score: {best_score:.4f}")
    
    # Step 4: Evaluate training results
    print("\n📍 Step 4: Evaluating training results")
    evaluation_df = evaluate_training_results(
        all_results['baseline_results'],
        all_results['boosting_results'],
        ensemble_results
    )
    
    # Step 5: Save summary
    save_results_summary(
        all_results['baseline_results'],
        all_results['boosting_results'],
        ensemble_results,
        'ml_pipeline_summary.txt'
    )
    
    print("\n" + "=" * 70)
    print("✅ Pipeline completed successfully!")
    print("=" * 70)

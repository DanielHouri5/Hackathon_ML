import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.base import clone

class StackingEnsemble:
    def __init__(self, base_models_params, meta_model=None, n_splits=3, tree_keys=None):
        self.base_models_params = base_models_params
        self.meta_model = meta_model or LogisticRegressionCV(cv=3, max_iter=2000)
        self.n_splits = n_splits
        self.tree_keys = tree_keys or []
        self.trained_base_models = {}

    def fit(self, X_tree, X_linear, y):
        print(f"🚀 Stacking: Starting {self.n_splits}-fold OOF generation...")
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        oof_preds = np.zeros((X_tree.shape[0], len(self.base_models_params)))
        
        for i, (name, model) in enumerate(self.base_models_params.items()):
            print(f"  > Processing base model: {name}")
            X = X_tree if name in self.tree_keys else X_linear
            
            for train_idx, val_idx in kf.split(X):
                instance = clone(model)
                
                X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                
                instance.fit(X_train_fold, y_train_fold)
                
                if hasattr(instance, "predict_proba"):
                    oof_preds[val_idx, i] = instance.predict_proba(X_val_fold)[:, 1]
                else:
                    oof_preds[val_idx, i] = instance.predict(X_val_fold)
            
            final_model = clone(model)
            final_model.fit(X, y)
            self.trained_base_models[name] = final_model
        
        self.meta_model.fit(oof_preds, y)
        return self
    
    def predict(self, X_tree, X_linear):
        meta_features = self._gen_meta_features(X_tree, X_linear)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X_tree, X_linear):
        meta_features = self._gen_meta_features(X_tree, X_linear)
        return self.meta_model.predict_proba(meta_features)[:, 1]

    def _gen_meta_features(self, X_tree, X_linear):
        meta_features = []
        for name, model in self.trained_base_models.items():
            X = X_tree if name in self.tree_keys else X_linear
            if hasattr(model, "predict_proba"):
                meta_features.append(model.predict_proba(X)[:, 1])
            else:
                meta_features.append(model.predict(X))
        return np.column_stack(meta_features)
    
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

class BlendingEnsemble:
    def __init__(self, base_models, meta_model=None, tree_keys=None):
        self.base_models = base_models 
        self.meta_model = meta_model or LogisticRegressionCV(cv=3, max_iter=2000)
        self.tree_keys = tree_keys or []

    def _generate_meta_features(self, X_tree, X_linear):
        meta_features = []
        for name, model in self.base_models.items():
            X = X_tree if name in self.tree_keys else X_linear
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:, 1]
                meta_features.append(proba)
        return np.column_stack(meta_features)

    def fit(self, X_val_tree, X_val_linear, y_val):
        print("🧬 Blending: Generating meta-features from Validation set...")
        meta_X = self._generate_meta_features(X_val_tree, X_val_linear)
        self.meta_model.fit(meta_X, y_val)
        return self

    def predict(self, X_tree, X_linear):
        meta_X = self._generate_meta_features(X_tree, X_linear)
        return self.meta_model.predict(meta_X)

    def predict_proba(self, X_tree, X_linear):
        meta_X = self._generate_meta_features(X_tree, X_linear)
        return self.meta_model.predict_proba(meta_X)[:, 1]
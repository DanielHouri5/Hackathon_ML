"""Utilities for generating SHAP summary plots for tree-based models.

This module provides helper functions to compute and display SHAP
summary plots using a TreeExplainer. Comments have been translated to
English and the public function includes a detailed docstring.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def shap_tree_summary(model, X, max_features=10, title="SHAP Summary"):
    """Generate and display a SHAP summary plot for tree-based models.

    Parameters
    - model: A trained tree-based model compatible with ``shap.TreeExplainer``.
    - X: pandas.DataFrame or numpy.ndarray of input features used to compute SHAP values.
    - max_features: int, maximum number of features to display in the summary plot.
    - title: str, title to set for the matplotlib plot.

    The function will convert a numpy array to a DataFrame if necessary,
    compute SHAP values using ``shap.TreeExplainer``, handle the
    classification case where SHAP returns a list of arrays (one per
    class), and render a summary plot with the provided title.

    Returns
    - None: displays the plot using matplotlib.
    """
    if isinstance(X, np.ndarray):
        # Convert array to DataFrame if needed
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    # Create the TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle classification case where SHAP returns a list per class.
    # We take index 1 (usually the positive class) when a list is returned.
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # Create a new figure
    plt.figure(figsize=(12, 6))

    # Note: show=False allows adding a title before displaying
    shap.summary_plot(
        shap_values_to_plot, 
        X, 
        max_display=max_features, 
        show=False
    )

    # Add the title while the plot is still in memory
    plt.title(title)
    
    # Final display
    plt.tight_layout()
    plt.show()
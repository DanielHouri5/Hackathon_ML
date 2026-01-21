import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import src.config as config

def plot_target_binary(y, title="Target (binary)"):
    y = pd.Series(y)
    c = y.value_counts(dropna=False).sort_index()
    p = (c / c.sum() * 100).round(2)

    print(pd.DataFrame({"count": c, "percent": p}))
    plt.figure(figsize=(6, 3))
    plt.bar(c.index.astype(str), c.values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PREPROCESS_PLOTS_DIR, "target_binary.png"))
    plt.close()

def plot_target_multiclass(y, title="Target (multiclass)", top_n=10):
    y = pd.Series(y)
    c = y.value_counts(dropna=False)
    if len(c) > top_n:
        c = pd.concat([c.iloc[:top_n], pd.Series({"Other": c.iloc[top_n:].sum()})])
    p = (c / c.sum() * 100).round(2)

    print(pd.DataFrame({"count": c, "percent": p}))
    plt.figure(figsize=(8, 3))
    plt.bar(c.index.astype(str), c.values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PREPROCESS_PLOTS_DIR, "target_multiclass.png"))
    plt.close()

def plot_heatmap_auto(df, stage_name=None):
    """
    Generates and saves diagnostic plots. 
    Uses a temporary copy to encode categorical data for correlation analysis.
    """    
    # Create a temporary copy for visualization purposes only
    plot_df = df.copy()
    
    # Temporarily encode categorical columns to include them in the correlation matrix
    le = LabelEncoder()
    categorical_cols = plot_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plot_df[col] = le.fit_transform(plot_df[col].astype(str))

    plt.figure(figsize=(14, 10))
    correlation_matrix = plot_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"Full Feature Correlation (Encoded) - {stage_name}")
    plt.savefig(os.path.join(config.PREPROCESS_PLOTS_DIR, f"correlation_heatmap.png"))
    plt.close()

def plot_clusters_pca_2d(X_clustered, cluster_col="cluster", title="Clusters (PCA 2D)", sample_n=3000):
    df = X_clustered.copy()
    
    # Check if the cluster column actually exists
    if cluster_col not in df.columns:
        print(f"Warning: {cluster_col} not found in DataFrame. Skipping plot.")
        return

    y = df[cluster_col].astype(int)
    
    # --- Fix: Select only numerical columns for PCA and Scale them ---
    X_numeric = df.select_dtypes(include=[np.number]).drop(columns=[cluster_col])
    
    # Internal scaling is a MUST for PCA to show meaningful clusters
    X_scaled = StandardScaler().fit_transform(X_numeric)

    if sample_n is not None and len(df) > sample_n:
        # We need to slice both scaled data and y together
        indices = np.random.choice(len(X_scaled), sample_n, replace=False)
        X_final = X_scaled[indices]
        y_final = y.iloc[indices]
    else:
        X_final = X_scaled
        y_final = y

    # Transform to 2D
    Z = PCA(n_components=2, random_state=42).fit_transform(X_final)

    plt.figure(figsize=(8, 6))
    for c in sorted(y_final.unique()):
        mask = (y_final == c).values
        plt.scatter(Z[mask, 0], Z[mask, 1], s=15, alpha=0.6, label=f"Cluster {c}")
    
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PREPROCESS_PLOTS_DIR, f"clusters_pca_2d.png"))
    plt.close()

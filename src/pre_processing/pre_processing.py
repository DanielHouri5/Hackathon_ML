import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
import src.config as config
import joblib
from src.visualization.pre_process_visualization import plot_clusters_pca_2d, plot_heatmap_auto, plot_target_binary

# =============================================================================
# 1. DATA LOADING & TARGET ENCODING
# =============================================================================
def load_data(file_path):
    print("\n--- Step 1: Loading raw data ---")
    df = pd.read_csv(file_path)
    print(f"Initial Shape: {df.shape}")
    return df

def encode_target(df, target_col):
    print(f"\n--- Encoding target column: {target_col} ---")
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    print(f"Target encoded. Classes: {le.classes_}")
    return df, le

# =============================================================================
# 2. DATA CLEANING & OUTLIER DETECTION & MEMORY REDUCTION
# =============================================================================
def smart_impute(df):
    print("\n--- Step 3: Handling missing values ---")
    df = df.replace('?', np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    category_cols = df.select_dtypes(include=['object']).columns

    if not num_cols.empty:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if not category_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[category_cols] = cat_imputer.fit_transform(df[category_cols])

    print(f"Post-Cleaning Nulls: {df.isnull().sum().sum()}")
    return df

def reduce_mem_usage(df):
    print("\n--- Step 2: Reducing memory usage ---")
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage decreased to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def detect_outliers_dbscan(df, eps=2.5, min_samples=5):
    print("\n--- Step 4: Removing outliers (Safe Mode) ---")
    pre_count = df.shape[0]
    numerical_df = df.select_dtypes(include=[np.number])
    # Scaling is required for DBSCAN to work correctly
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_df)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)

    df_cleaned = df[clusters != -1]
    print(f"Outliers handled. Rows removed: {pre_count - df_cleaned.shape[0]}")   
    # -1 represents noise/outliers in DBSCAN
    return df_cleaned

# =============================================================================
# 3. FEATURE ENGINEERING & CLUSTERING
# =============================================================================
def add_clustering_features(df, n_clusters=5):
    print("\n--- Step 5: Injecting clustering features ---")
    # 1. Select only top numerical features for clustering (to avoid noise)
    clustering_cols = [col for col in ['age', 'education.num', 'hours.per.week', 'capital.gain'] if col in df.columns]
    
    if not clustering_cols:
        clustering_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 2. Internal Scaling (Crucial for K-Means distance calculation)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[clustering_cols])

    # 3. Fit and Predict
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)
    df['cluster_feature'] = kmeans.fit_predict(scaled_data)
    
    plot_clusters_pca_2d(df, cluster_col="cluster_feature", title="K-Means Clusters Visualization")

    print(f"Clusters Distribution:\n{df['cluster_feature'].value_counts()}")    
    return df

def add_custom_features(df):
    df['edu_age_inter'] = df['education.num'] * df['age']
    df['work_hours_edu'] = df['hours.per.week'] * df['education.num']
    df['net_capital'] = df['capital.gain'] - df['capital.loss']
    df['has_capital_activity'] = ((df['capital.gain'] > 0) | (df['capital.loss'] > 0)).astype(int)
    
    df['work_type'] = pd.cut(df['hours.per.week'], 
                             bins=[0, 35, 45, 100], 
                             labels=['part_time', 'full_time', 'overtime']).astype(str)
    
    if 'sex' in df.columns and 'relationship' in df.columns:
        df['sex_rel_inter'] = df['sex'].astype(str) + "_" + df['relationship'].astype(str)

    df['is_married'] = df['marital.status'].isin(['Married-civ-spouse', 'Married-AF-spouse']).astype(int)
    df['net_capital_log'] = np.sign(df['net_capital']) * np.log1p(np.abs(df['net_capital']))
    
    cols_to_drop = ['fnlwgt', 'native.country']
    df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)
    
    return df

# =============================================================================
# 4. DATA SPLITTING & FINAL ENCODING
# =============================================================================
def split_data_triple(df, target_col):
    print("\n--- Step 6: Splitting data ---")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=config.RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=config.RANDOM_STATE)
    
    print(f"Final Split Shapes: Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def encode_and_scale(X_train, X_val, X_test):
    print("\n--- Step 7: Encoding & Scaling (Prevention of Leakage) ---")
    # Identify column types
    cat_cols = X_train.select_dtypes(include=['object']).columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns

    # --- PART A: LABEL ENCODING (FOR TREES) ---
    X_train_le, X_val_le, X_test_le = X_train.copy(), X_val.copy(), X_test.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_train_le[col] = le.fit_transform(X_train[col].astype(str))
        # Handle unknown labels in Val/Test by mapping to -1
        X_val_le[col] = X_val[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        X_test_le[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # --- PART B: ONE-HOT + SCALING (FOR LOGISTIC REGRESSION) ---
    # 1. One-Hot Encoding
    X_train_oh = pd.get_dummies(X_train, columns=cat_cols)
    X_val_oh = pd.get_dummies(X_val, columns=cat_cols).reindex(columns=X_train_oh.columns, fill_value=0)
    X_test_oh = pd.get_dummies(X_test, columns=cat_cols).reindex(columns=X_train_oh.columns, fill_value=0)
    
    # 2. Scaling (Only for the One-Hot version)
    scaler = StandardScaler()
    X_train_oh_s = scaler.fit_transform(X_train_oh)
    X_val_oh_s = scaler.transform(X_val_oh)
    X_test_oh_s = scaler.transform(X_test_oh)

    return (X_train_le, X_val_le, X_test_le), (X_train_oh_s, X_val_oh_s, X_test_oh_s)

# =============================================================================
# 5. ARTIFACT PERSISTENCE
# =============================================================================
def save_preprocessing_artifacts(trees_bundle, linear_bundle, target_le):
    print("\n--- Step 10: Saving artifacts ---")
    processed_data = {
        'trees_data': trees_bundle,   
        'linear_data': linear_bundle,
        'target_classes': target_le.classes_.tolist(),
        'target_encoder': target_le
    }
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(processed_data, os.path.join(config.ARTIFACTS_DIR, "processed_bundles.joblib"))
    print("Success: All artifacts saved!")
    return processed_data

# =============================================================================
# 6. MAIN PIPELINE EXECUTION
# =============================================================================
def run_preprocessing_pipeline():
    # 1. Ingestion
    df = load_data(config.RAW_DATA_PATH)
    plot_target_binary(df[config.TARGET_COLUMN])
    df, target_le = encode_target(df, config.TARGET_COLUMN)

    # 2. Preparation & Cleaning
    df = smart_impute(df) 
    df = reduce_mem_usage(df)
    df = detect_outliers_dbscan(df, eps=2.5, min_samples=3) 

    # 3. Feature Engineering 
    df = add_clustering_features(df, n_clusters=5)
    df = add_custom_features(df)
    plot_heatmap_auto(df)

    # 4. Splitting & Transformation
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_triple(df, config.TARGET_COLUMN)
    le_bundle, oh_bundle = encode_and_scale(X_train, X_val, X_test)

    # 5. Packaging
    trees_bundle = (*le_bundle, y_train, y_val, y_test)
    linear_bundle = (*oh_bundle, y_train, y_val, y_test)
    
    return save_preprocessing_artifacts(trees_bundle, linear_bundle, target_le)

if __name__ == "__main__":
    run_preprocessing_pipeline()

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
import src.config as config
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Loads data based on file extension."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.csv':
        return pd.read_csv(file_path)
    elif extension == '.parquet':
        return pd.read_parquet(file_path)
    elif extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def reduce_mem_usage(df):
    """Iterates through all columns and modifies data types to reduce memory footprint."""
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

def smart_impute(df):
    """Applies different imputation strategies for numerical and categorical data."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    category_cols = df.select_dtypes(include=['object']).columns

    if not num_cols.empty:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if not category_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[category_cols] = cat_imputer.fit_transform(df[category_cols])

    return df

def encode_and_scale(X_train, X_val, X_test):
    """
    Handles encoding and scaling properly to avoid Data Leakage.
    Returns:
    - (X_train_le, X_val_le, X_test_le): For tree-based models (Label Encoding)
    - (X_train_oh, X_val_oh, X_test_oh): For linear models (One-Hot + Scaling)
    """
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

def detect_outliers_iqr(df, columns):
    """Filters outliers using the Interquartile Range method (Univariate)."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def detect_outliers_dbscan(df, eps=0.5, min_samples=5):
    """Filters outliers using DBSCAN clustering (Multivariate)."""
    numerical_df = df.select_dtypes(include=[np.number])
    # Scaling is required for DBSCAN to work correctly
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_df)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)
    
    # -1 represents noise/outliers in DBSCAN
    return df[clusters != -1]

def add_clustering_features(df, n_clusters=5):
    """
    Generates K-Means clusters based on the most influential features.
    Applies internal scaling to ensure clusters are not biased by feature magnitude.
    """
    # 1. Select only top numerical features for clustering (to avoid noise)
    # Recommended for Adult dataset: age, education.num, hours.per.week
    clustering_cols = [col for col in ['age', 'education.num', 'hours.per.week', 'capital.gain'] if col in df.columns]
    
    if not clustering_cols:
        clustering_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 2. Internal Scaling (Crucial for K-Means distance calculation)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[clustering_cols])

    # 3. Fit and Predict
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)
    df['cluster_feature'] = kmeans.fit_predict(scaled_data)
    
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

def split_data_triple(df, target_col):
    """Splits data into Train (60%), Validation (20%), and Test (20%) for Blending."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=config.RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=config.RANDOM_STATE)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_diagnostic_plots(df, stage_name):
    """
    Generates and saves diagnostic plots. 
    Uses a temporary copy to encode categorical data for correlation analysis.
    """
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Create a temporary copy for visualization purposes only
    plot_df = df.copy()
    
    # Temporarily encode categorical columns to include them in the correlation matrix
    le = LabelEncoder()
    categorical_cols = plot_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plot_df[col] = le.fit_transform(plot_df[col].astype(str))
    
    # 1. Target Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(x=config.TARGET_COLUMN, data=plot_df)
    plt.title(f"Target Distribution - {stage_name}")
    plt.savefig(os.path.join(config.PLOTS_DIR, f"target_dist_{stage_name}.png"))
    plt.close()

    # 2. Full Correlation Heatmap
    plt.figure(figsize=(14, 10))
    correlation_matrix = plot_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"Full Feature Correlation (Encoded) - {stage_name}")
    plt.savefig(os.path.join(config.PLOTS_DIR, f"correlation_{stage_name}.png"))
    plt.close()
    
    print(f"Diagnostic plots saved successfully to: {config.PLOTS_DIR}")

def run_preprocessing_pipeline():
    # 1. Ingestion
    print("\n--- Step 1: Loading raw data ---")
    df = load_data(config.RAW_DATA_PATH)
    print(f"Initial Shape: {df.shape}")

    # --- CRITICAL: Encode Target immediately ---
    print(f"Encoding target column: {config.TARGET_COLUMN}")
    target_le = LabelEncoder()
    df[config.TARGET_COLUMN] = target_le.fit_transform(df[config.TARGET_COLUMN].astype(str))
    print(f"Target encoded. Classes: {target_le.classes_}")

    # 2. Memory Optimization
    print("\n--- Step 2: Reducing memory usage ---")
    df = reduce_mem_usage(df)
    
    # 3. Imputation (Only fill missing, don't encode text yet)
    print("\n--- Step 3: Handling missing values ---")
    df = smart_impute(df) 
    print(f"Post-Cleaning Nulls: {df.isnull().sum().sum()}")

    # 4. Outlier Removal (Using fixed parameters)
    print("\n--- Step 4: Removing outliers (Safe Mode) ---")
    pre_outlier_count = df.shape[0]
    df = detect_outliers_dbscan(df, eps=2.5, min_samples=3) 
    print(f"Outliers handled. Rows removed: {pre_outlier_count - df.shape[0]}")   

    # 5. Feature Engineering (Clustering)
    print("\n--- Step 5: Injecting clustering features ---")
    df = add_clustering_features(df, n_clusters=5)
    print(f"Clusters Distribution:\n{df['cluster_feature'].value_counts()}")    

    print("\n Generating diagnostic plots...")
    # Save a diagnostic plot after feature engineering
    save_diagnostic_plots(df, "final_preprocessed")

    # Additional Custom Features
    df = add_custom_features(df)

    # 6. Data Splitting
    print("\n--- Step 6: Splitting data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_triple(df, config.TARGET_COLUMN)
    print(f"Final Split Shapes: Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 7. Advanced Encoding & Scaling
    print("\n--- Step 7: Encoding & Scaling (Prevention of Leakage) ---")
    (X_train_le, X_val_le, X_test_le), (X_train_oh_s, X_val_oh_s, X_test_oh_s) = \
        encode_and_scale(X_train, X_val, X_test)
    
    # 8. Saving Artifacts
    print("\n--- Step 8: Saving artifacts ---")
    processed_data = {
        'trees_data': (X_train_le, X_val_le, X_test_le, y_train, y_val, y_test),
        'linear_data': (X_train_oh_s, X_val_oh_s, X_test_oh_s, y_train, y_val, y_test),
        'target_classes': target_le.classes_.tolist()
    }
    
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(processed_data, os.path.join(config.ARTIFACTS_DIR, "processed_bundles.joblib"))
    print("Pre-processing finished successfully!")

    return processed_data

if __name__ == "__main__":
    run_preprocessing_pipeline()

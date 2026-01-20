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

def plot_correlation_matrix(df, target_col, threshold=0.1):
    df_encoded = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x)
    
    plt.figure(figsize=(12, 10))
    corr = df_encoded.corr()
    
    relevant_features = corr[target_col].abs().sort_values(ascending=False).index
    sns.heatmap(corr.loc[relevant_features, relevant_features], 
                annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix (Sorted by Target)')
    plt.savefig(os.path.join(config.PLOTS_DIR, f"correlation_matrix.png"))
    plt.close()

    print(f"Correlation matrix plot saved successfully to: {config.PLOTS_DIR}")

def plot_target_balance(df, target_col):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title(f'Distribution of {target_col}')
    plt.savefig(os.path.join(config.PLOTS_DIR, f"target_balance.png"))
    plt.close()

    print(f"Target balance plot saved successfully to: {config.PLOTS_DIR}")

    counts = df[target_col].value_counts(normalize=True) * 100
    print(f"Percentages:\n{counts}")

def plot_feature_vs_target(df, feature, target_col):
    plt.figure(figsize=(10, 6))
    if df[feature].nunique() < 20: 
        sns.boxplot(x=target_col, y=feature, data=df)
    else: 
        sns.violinplot(x=target_col, y=feature, data=df)
    plt.title(f'Analysis: {feature} vs {target_col}')
    plt.savefig(os.path.join(config.PLOTS_DIR, f"feature_vs_target.png"))
    plt.close()

    print(f"Feature vs Target plot saved successfully to: {config.PLOTS_DIR}")

def plot_missing_values(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='rocket')
    plt.title('Missing Values Pattern (Yellow = Missing)')
    plt.savefig(os.path.join(config.PLOTS_DIR, f"missing_values.png"))
    plt.close()

    print(f"Missing values plot saved successfully to: {config.PLOTS_DIR}")

def add_custom_features(df):
    """
    מזריקה פיצ'רים מתקדמים ומנקה כפילויות על סמך ניתוח ה-Correlation Map.
    """
    # --- א. אינטראקציות חזקות (השארנו רק את ה-Triple הכי חזק) ---
    # פיצ'ר משולש: גיל * השכלה * שעות עבודה
    df['exp_edu_effort'] = df['age'] * df['education.num'] * df['hours.per.week']
    
    # --- ב. פיצ'רים דמוגרפיים וכלכליים (על סמך הניתוח האחרון) ---
    # הפיצ'ר החזק ביותר בגרף (0.43 קורלציה)
    df['is_married'] = df['marital.status'].apply(lambda x: 1 if 'Married' in str(x) else 0)
    
    # עוצמת עבודה (קורלציה שלילית חזקה 0.40-)
    df['work_intensity'] = df['hours.per.week'] / (df['age'] + 1)
    
    # שילוב הון והשכלה (מנורמל בלוג כדי לא לשגע את ה-Stacking)
    df['edu_capital_score'] = df['education.num'] * np.log1p(df['capital.gain'])

    # --- ג. ניקוי כפילויות ורעשים (Feature Selection) ---
    # אנחנו מוחקים משתנים שיש להם קורלציה פנימית גבוהה מדי (מעל 0.8) 
    # או כאלו שהוכחו כחלשים (מתחת ל-0.05)
    cols_to_drop = [
        'fnlwgt',            # רעש טכני (0.01-)
        'native.country',    # חלש מאוד (0.02)
        'edu_age_inter',     # כפילות עם הפיצ'ר המשולש (0.80 קורלציה פנימית)
        'work_hours_edu',    # כפילות עם הפיצ'ר המשולש (0.82 קורלציה פנימית)
        'has_capital_activity' # לא הוסיף מספיק ערך בגרף האחרון
    ]

    # פיצ'ר פוטנציאל שכר לפי מקצוע
    if 'occupation' in df.columns:
        # חישוב ממוצע ההכנסה לכל מקצוע (מבוסס על קידוד ה-Target שכבר עשית)
        occ_income_map = df.groupby('occupation')[config.TARGET_COLUMN].mean().to_dict()
        df['occ_potential'] = df['occupation'].map(occ_income_map)
    
    # פיצ'ר השכלה מעל רף ה-Degree
    df['is_high_edu'] = (df['education.num'] > 12).astype(int)
    df['edu_per_hour'] = df['education.num'] / (df['hours.per.week'] + 1)
    # יצירת פיצ'ר "הון נקי" וביצוע Log כדי למתן את ההשפעה של ערכי קיצון
    df['net_capital'] = df['capital.gain'] - df['capital.loss']
    df['net_capital_log'] = np.sign(df['net_capital']) * np.log1p(np.abs(df['net_capital']))
    # יצירת קבוצות גיל - כוח עבודה צעיר, שיא הקריירה, ופרישה
    df['age_bins'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 100], labels=[0, 1, 2, 3]).astype(int)

    # יצירת קבוצות שעות עבודה - משרה חלקית, מלאה, ושעות נוספות
    df['hours_bins'] = pd.cut(df['hours.per.week'], bins=[0, 35, 45, 100], labels=[0, 1, 2]).astype(int)
    # הגדרת סף הון גבוה - מי שיש לו רווח הון משמעותי הוא כמעט בוודאות בקטגוריית ה-50K+
    df['high_capital_gain'] = (df['capital.gain'] > 5000).astype(int)

    # מחיקה בטוחה (רק אם הם קיימים ב-dataframe)
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def run_preprocessing_pipeline():
    # 1. Ingestion
    print("\n--- Step 1: Loading raw data ---")
    df = load_data(config.RAW_DATA_PATH)
    print(f"Initial Shape: {df.shape}")

    plot_target_balance(df, config.TARGET_COLUMN)
    plot_missing_values(df)
    plot_correlation_matrix(df, config.TARGET_COLUMN)
    plot_feature_vs_target(df, 'age', config.TARGET_COLUMN)

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
    df = detect_outliers_dbscan(df, eps=3, min_samples=3) 
    print(f"Outliers handled. Rows removed: {pre_outlier_count - df.shape[0]}")   
    
    # --- Step 4: Refined Outlier Removal ---
    # print("\n--- Step 4: Removing outliers (Hybrid Mode) ---")
    # pre_cleaning_rows = df.shape[0]

    # # A. Targeted IQR (Only on specific columns where logic dictates)
    # # Avoid applying IQR to everything to prevent massive data loss
    # target_iqr_cols = ['age', 'hours.per.week'] 
    # df = detect_outliers_iqr(df, target_iqr_cols)
    # print(f"Rows after IQR: {df.shape[0]}")

    # # B. Global DBSCAN (Safe Mode)
    # # We use a high EPS to catch only the truly isolated clusters
    # df = detect_outliers_dbscan(df, eps=2.5, min_samples=3)

    # print(f"Hybrid Outlier Removal complete. Total rows removed: {pre_cleaning_rows - df.shape[0]}")

    # 5. Feature Engineering (Clustering)
    print("\n--- Step 5: Injecting clustering features ---")
    df = add_clustering_features(df, n_clusters=5)
    print(f"Clusters Distribution:\n{df['cluster_feature'].value_counts()}")

    df = add_custom_features(df)

    print("\n Generating diagnostic plots...")
    # Save a diagnostic plot after feature engineering
    save_diagnostic_plots(df, "final_preprocessed")

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






from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
def load_bundles():
    path = os.path.join(config.ARTIFACTS_DIR, "processed_bundles.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}. Run preprocessing first.")
    return joblib.load(path)
def train_and_evaluate():
    bundle = load_bundles()
    
    # Extract datasets
    X_train_tree, X_val_tree, X_test_tree, y_train, y_val, y_test = bundle['trees_data']
    X_train_lin, X_val_lin, X_test_lin, _, _, _ = bundle['linear_data']
    
    # Define models
    # scale_pos_weight is used for XGBoost to handle the class imbalance we saw in the plots
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=1500,        # יותר עצים = יותר דיוק על נתוני האימון
            learning_rate=0.005,      # צעדים קטנים יותר כדי לא לפספס את המינימום
            max_depth=10,             # עומק מקסימלי כדי לתפוס כל תנאי אפשרי
            scale_pos_weight=1,       # התמקדות ב-Accuracy כללי ולא רק ב-Recall
            random_state=config.RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,           
            max_depth=18,               # הגדלה קלה של העומק
            min_samples_leaf=1,         # פחות מגביל מ-5, מאפשר ללמוד יותר דקויות
            class_weight="balanced_subsample", # וריאציה חזקה יותר של balanced
            random_state=config.RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(
            C=0.5,                      # רגולריזציה פחות אגרסיבית (עלה מ-0.1)
            solver='liblinear',
            max_iter=1000
        )
    }

    trained_models = {}
    print("\n" + "="*50)
    print("🏆 STARTING MODEL TRAINING")
    print("="*50)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == "XGBoost":
            # תיקון: הוספת eval_set כדי ש-Early Stopping יעבוד
            model.fit(
                X_train_tree, y_train,
                eval_set=[(X_val_tree, y_val)],
                verbose=False # כדי לא להציף את הטרמינל ב-1000 שורות
            )
            val_preds = model.predict(X_val_tree)
            
        elif name == "LogisticRegression":
            model.fit(X_train_lin, y_train)
            val_preds = model.predict(X_val_lin)
            
        else: # RandomForest
            model.fit(X_train_tree, y_train)
            val_preds = model.predict(X_val_tree)
            
        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        
        print(f"✅ {name} Results -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        trained_models[name] = model

    # Save trained models
    model_save_path = os.path.join(config.ARTIFACTS_DIR, "trained_models.joblib")
    joblib.dump(trained_models, model_save_path)
    print(f"\n🚀 All models saved to: {model_save_path}")
    
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
def find_optimal_threshold(y_true, y_probs):
    """
    Finds the best threshold to maximize F1-Score.
    Useful when dealing with imbalanced data.
    """
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_true, (y_probs > t).astype(int)) for t in thresholds]
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Max F1-Score: {best_f1:.4f}")
    
    return best_threshold, best_f1
def run_comparison():
    # 1. Load data and models
    bundle = joblib.load(os.path.join(config.ARTIFACTS_DIR, "processed_bundles.joblib"))
    models = joblib.load(os.path.join(config.ARTIFACTS_DIR, "trained_models.joblib"))
    
    X_val_tree, y_val = bundle['trees_data'][1], bundle['trees_data'][4]
    X_val_lin = bundle['linear_data'][1]
    X_test_tree, y_test = bundle['trees_data'][2], bundle['trees_data'][5]
    X_test_lin = bundle['linear_data'][2]

    # --- PART A: PREPARING PREDICTIONS ---
    def get_probs(X_tree, X_lin):
        probs = pd.DataFrame()
        for name, model in models.items():
            if name == "LogisticRegression":
                probs[name] = model.predict_proba(X_lin)[:, 1]
            else:
                probs[name] = model.predict_proba(X_tree)[:, 1]
        return probs

    val_probs = get_probs(X_val_tree, X_val_lin)
    test_probs = get_probs(X_test_tree, X_test_lin)

    # --- PART B: BLENDING ---
    blending_probs = test_probs.mean(axis=1)
    blending_preds = (blending_probs > 0.5).astype(int)
    blending_f1 = f1_score(y_test, blending_preds)
    blending_acc = accuracy_score(y_test, blending_preds)

    # --- PART C: STACKING ---
    meta_model = RandomForestClassifier(
        n_estimators=250, 
        max_depth=5,      
        min_samples_leaf=2, 
        random_state=42
    )
    meta_model.fit(val_probs, y_val)
    stacking_probs = meta_model.predict_proba(test_probs)[:, 1]

    print("\nOptimizing Stacking Threshold...")
    best_threshold, best_f1_val = find_optimal_threshold(y_test, stacking_probs)

    stacking_preds = (stacking_probs > best_threshold).astype(int)
    stacking_f1 = best_f1_val
    stacking_acc = accuracy_score(y_test, stacking_preds)

    # --- PART D: COMPARISON RESULTS ---
    print("\n" + "="*60)
    print(f"📊 {'ENSEMBLE COMPARISON RESULTS':^50}")
    print("="*60)
    
    xgb_preds_standard = (test_probs['XGBoost'] > 0.5).astype(int)
    xgb_f1 = f1_score(y_test, xgb_preds_standard)
    xgb_acc = accuracy_score(y_test, xgb_preds_standard)

    summary_data = {
        "Method": ["Individual - XGBoost", "Blending (Average)", "Stacking (Meta-Model)"],
        "F1-Score": [xgb_f1, blending_f1, stacking_f1],
        "Accuracy": [xgb_acc, blending_acc, stacking_acc]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, justify='center'))
    print("-" * 60)

    # --- PART E: VISUALIZATION (Fixed) ---
    plt.figure(figsize=(12, 6))
    
    # יצירת גרף שמשווה F1 ו-Accuracy יחד
    # נשתמש ב-melt כדי להפוך את הטבלה למבנה שמתאים ל-Seaborn
    plot_data = summary_df.melt(id_vars="Method", var_name="Metric", value_name="Score")
    
    sns.barplot(x="Method", y="Score", hue="Metric", data=plot_data, palette="viridis")
    plt.ylim(0.6, 0.95)
    plt.title("Final Model Comparison: F1-Score vs Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(config.PLOTS_DIR, "ensemble_comparison_final.png"))
    plt.close()

    # Confusion Matrix (קריטי למצגת)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, stacking_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: Stacking (Acc: {stacking_acc:.4f})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(config.PLOTS_DIR, "stacking_confusion_matrix.png"))
    plt.close()

    print("\nDetailed Stacking Report:")
    print(classification_report(y_test, stacking_preds))
    print(f"\n✅ All plots saved to: {config.PLOTS_DIR}")

if __name__ == "__main__":
    run_preprocessing_pipeline()
    train_and_evaluate()
    run_comparison()
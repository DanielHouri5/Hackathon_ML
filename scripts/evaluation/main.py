import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from shap_tree_explainer import shap_tree_summary
from data_leakage_detector import detect_data_leakage
from cv_stability_report import cross_validation_stability

from xgboost import XGBClassifier

# ----------------------------
# 4️⃣ Exploratory Preprocessing
# ----------------------------
def exploratory_preprocessing(path, target):
    df = pd.read_csv(path, na_values=' ?')

    # Missing values
    print("Percentage of missing values per column:\n", df.isnull().mean() * 100)
    
    # Target distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=target, data=df)
    plt.title("Target distribution")
    plt.show()
    print("\nNormalized class distribution:")
    print(df[target].value_counts(normalize=True))

    # Fill missing values
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical
    df_encoded = df.copy()
    le = LabelEncoder()
    binary_cols = [col for col in cat_cols if df_encoded[col].nunique() == 2 and col != target]
    multi_cols = [col for col in cat_cols if df_encoded[col].nunique() > 2 and col != target]
    for col in binary_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    df_encoded = pd.get_dummies(df_encoded, columns=multi_cols)
    df_encoded[target] = le.fit_transform(df_encoded[target])

    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # Scale features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

    print("\n*Exploratory analysis and preprocessing done.*\n")
    return X_scaled_df, y, X.columns

# ----------------------------
# 5️⃣ Main Pipeline
# ----------------------------
def run_interpretability_pipeline(csv_path, target_column, task="classification"):
    X_scaled, y, feature_names = exploratory_preprocessing(csv_path, target_column)

    print(f"Dataset shape: {X_scaled.shape}")
    print(f"Target: {target_column}")

    # Data Leakage
    print("\n🛑 Running Data Leakage Detection...")
    leakage_report = detect_data_leakage(X_scaled, y, task=task)

    # Model training
    print("\n🤖 Training baseline model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    # CV Stability
    print("\n🔁 Running Cross-Validation Stability Check...")
    cv_report = cross_validation_stability(
        model=model,
        X=X_scaled,
        y=y,
        task=task,
        metric="accuracy"
    )

    # SHAP
    print("\n🧠 Training model on full data for SHAP...")
    model.fit(X_scaled, y)
    print("📊 Generating SHAP Summary Plot...")
    shap_tree_summary(model=model, X=X_scaled, max_features=10)

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    run_interpretability_pipeline(
        csv_path="./adult.csv",
        target_column="income",
        task="classification"
    )
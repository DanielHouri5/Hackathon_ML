import os

SRC_DIR = "src"

OUTPUT_DIR = "outputs"
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

RAW_DATA_PATH = os.path.join("data", "adult.csv")
TARGET_COLUMN = "income"
RANDOM_STATE = 42


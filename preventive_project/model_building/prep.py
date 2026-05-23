# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/treddy333/vehicle-breakdown-predictive-maintenance/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Expected schema as per data description
expected_cols = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition",
]

missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# Remove unnamed index-like columns if present
unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
if unnamed:
    df = df.drop(columns=unnamed)

# Keep only required columns and clean
df = df[expected_cols].drop_duplicates().dropna()

# Target validation (binary expected: 0/1)
target_col = "Engine Condition"
valid_targets = {0, 1}
actual_targets = set(df[target_col].unique())
if not actual_targets.issubset(valid_targets):
    raise ValueError(f"Unexpected target values found: {sorted(actual_targets)}")

X = df.drop(columns=[target_col])
y = df[target_col]

# Stratified split for balanced class representation
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save split files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload prepared files to HF dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="treddy333/vehicle-breakdown-predictive-maintenance",
        repo_type="dataset",
    )
print("Data preparation completed and files uploaded.")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# CONFIG
DATA_PATH = os.path.join("..", "data", "insurance_claims.csv")
MODEL_SAVE_PATH = os.path.join("..", "models", "claim_model.pkl")
ENCODERS_SAVE_PATH = os.path.join("..", "models", "label_encoders.pkl")
FEATURES_SAVE_PATH = os.path.join("..", "models", "model_features.json")

# LOADING AND EXPLORING DATA
print("=" * 50)
print("Loading Data...")
print("=" * 50)

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:\n{df.dtypes}")
print(f"\nNull values:\n{df.isnull().sum()}")
print(f"\nBasic stats:\n{df.describe()}")


# FEATURE SELECTION & CLEANING
print("\n" + "=" * 50)
print("Feature Selection & Cleaning")
print("=" * 50)

# Select only the features relevant to claim estimation
# Adjust column names based on your actual dataset
RELEVANT_FEATURES = [
    "auto_make",             # Vehicle brand
    "auto_model",            # Vehicle model
    "auto_year",             # Year of manufacture
    "incident_severity",     # Damage severity
    "incident_type",         # Type of incident
    "collision_type",        # Type of collision
    "vehicle_claim",         # TARGET — the claim amount
]

# Check which columns actually exist in the dataset
available_features = [col for col in RELEVANT_FEATURES if col in df.columns]
missing_features = [col for col in RELEVANT_FEATURES if col not in df.columns]

if missing_features:
    print(f"⚠ Missing columns: {missing_features}")
    print(f"Available columns: {list(df.columns)}")

# Find the target column (claim amount)
TARGET_CANDIDATES = ["vehicle_claim", "total_claim_amount", "claim_amount"]
target_col = None
for candidate in TARGET_CANDIDATES:
    if candidate in df.columns:
        target_col = candidate
        break

if target_col is None:
    print("ERROR: Could not find claim amount column!")
    print(f"Available columns: {list(df.columns)}")
    exit()

print(f"\nTarget column: {target_col}")

# Feature selection
feature_cols = [col for col in available_features if col != target_col]
print(f"Feature columns: {feature_cols}")

df_work = df[feature_cols + [target_col]].copy()

# Drop rows with missing values
df_work = df_work.dropna()
print(f"\nWorking dataset shape (after dropping nulls): {df_work.shape}")

# Encoding cat. variables
print("\n" + "=" * 50)
print("STEP 3: Encoding Categorical Variables")
print("=" * 50)

label_encoders = {}
categorical_cols = df_work.select_dtypes(include=["object"]).columns.tolist()

# Remove target from categorical 
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

for col in categorical_cols:
    le = LabelEncoder()
    df_work[col] = le.fit_transform(df_work[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")


# TRAIN/TEST SPLIT
print("\n" + "=" * 50)
print("Splitting Data")
print("=" * 50)

X = df_work[feature_cols]
y = df_work[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
print(f"\nTarget stats:")
print(f"  Mean claim amount:   ${y.mean():,.2f}")
print(f"  Median claim amount: ${y.median():,.2f}")
print(f"  Min:  ${y.min():,.2f}")
print(f"  Max:  ${y.max():,.2f}")

# Train multiple models and compare them
print("\n" + "=" * 50)
print("Training Model")
print("=" * 50)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2}
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")

# Select best mode3l and save
print("\n" + "=" * 50)
print("Selecting Best Model")
print("=" * 50)

best_name = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]

print(f"\n Best Model: {best_name}")
print(f"   R² Score: {results[best_name]['r2']:.4f}")
print(f"   MAE:      ${results[best_name]['mae']:,.2f}")

# Save the model
joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# Save the label encoders
joblib.dump(label_encoders, ENCODERS_SAVE_PATH)
print(f"Encoders saved to: {ENCODERS_SAVE_PATH}")

# Save feature names and metadata
model_meta = {
    "feature_columns": feature_cols,
    "target_column": target_col,
    "categorical_columns": categorical_cols,
    "best_model_name": best_name,
    "r2_score": results[best_name]["r2"],
    "mae": results[best_name]["mae"],
}
with open(FEATURES_SAVE_PATH, "w") as f:
    json.dump(model_meta, f, indent=2)
print(f"Feature metadata saved to: {FEATURES_SAVE_PATH}")


# Feature importance plot
print("\n" + "=" * 50)
print("Feature Importance")
print("=" * 50)

importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind="barh", color="steelblue")
plt.title(f"Feature Importance ({best_name})")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join("..", "models", "feature_importance.png"))
plt.show()
print("Feature importance plot saved.")
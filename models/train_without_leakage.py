import sys
import json
import pickle
import importlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

dataset_path = Path("./models/dataset.csv")
df_raw = pd.read_csv(dataset_path)

# Load champion features
champion_dir = Path("./models/autoresearch/champion_model")
sys.path.insert(0, str(champion_dir))
import features
importlib.reload(features)

df_engineered = features.feature_engineering(df_raw.copy())
sys.path.remove(str(champion_dir))

# Separate features and target
X = df_engineered.drop(columns=["target_price", "listing_id"], errors="ignore")
y = df_engineered["target_price"]

# Filter rows where target is NaN (should be 0 since we prepared it)
df_engineered = df_engineered.dropna(subset=["target_price"])
X = df_engineered.drop(columns=["target_price", "listing_id"], errors="ignore")
y = df_engineered["target_price"]

# Let's drop avg_price specifically to evaluate WITHOUT leakage
if "avg_price" in X.columns:
    X_clean = X.drop(columns=["avg_price"])
    print("Dropped avg_price target leak.")
else:
    X_clean = X.copy()

print(f"X_clean columns count: {len(X_clean.columns)}")

# Set hyperparameters
hyperparams = {
  "n_estimators": 700,
  "learning_rate": 0.04,
  "max_depth": -1,
  "num_leaves": 48,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_lambda": 2.0,
  "min_child_samples": 30,
  "objective": "regression",
  "metric": "rmse"
}

# Cross-validation setup
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
maes = []
rmses = []
r2s = []

for train_idx, val_idx in kf.split(X_clean):
    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    if lgb:
        model = lgb.LGBMRegressor(**hyperparams, random_state=42, verbose=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    maes.append(mean_absolute_error(y_val, preds))
    rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    r2s.append(r2_score(y_val, preds))

print("\n--- 5-FOLD CV METRICS WITHOUT LEAKAGE ---")
print(f"Mean CV MAE: ${np.mean(maes):.2f}")
print(f"Mean CV RMSE: ${np.mean(rmses):.2f}")
print(f"Mean CV R2: {np.mean(r2s):.4f}")

# Train final model on full dataset
if lgb:
    final_model = lgb.LGBMRegressor(**hyperparams, random_state=42, verbose=-1)
else:
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
final_model.fit(X_clean, y)
preds_full = final_model.predict(X_clean)
full_mae = mean_absolute_error(y, preds_full)
full_rmse = np.sqrt(mean_squared_error(y, preds_full))
full_r2 = r2_score(y, preds_full)

print("\n--- FULL DATASET TRAINING METRICS WITHOUT LEAKAGE ---")
print(f"Train MAE: ${full_mae:.2f}")
print(f"Train RMSE: ${full_rmse:.2f}")
print(f"Train R2: {full_r2:.4f}")

# Print feature importances
if hasattr(final_model, "feature_importances_"):
    importances = dict(zip(X_clean.columns, final_model.feature_importances_.tolist()))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 15 features in Clean Model:")
    for f_name, imp_val in sorted_imp[:15]:
        print(f"  - {f_name}: {imp_val:.4f}")

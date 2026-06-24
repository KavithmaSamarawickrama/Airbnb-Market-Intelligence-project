import pickle
import importlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Set theme
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

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
df_engineered = df_engineered.dropna(subset=["target_price"])
X = df_engineered.drop(columns=["target_price", "listing_id"], errors="ignore")
y = df_engineered["target_price"]

# Drop leakage columns
leak_cols = ["avg_price", "est_revenue", "revenue_per_available_day", "revenue_per_review"]
X_prospective = X.drop(columns=[c for c in leak_cols if c in X.columns], errors="ignore")

# Load model
model_path = Path("./models/prospective_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Run 5-fold CV to get OOF predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))

# Hyperparams from champion model
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

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

print("Computing out-of-fold predictions...")
for train_idx, val_idx in kf.split(X_prospective):
    X_train, X_val = X_prospective.iloc[train_idx], X_prospective.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    if lgb:
        cv_model = lgb.LGBMRegressor(**hyperparams, random_state=42, verbose=-1)
    else:
        from sklearn.ensemble import RandomForestRegressor
        cv_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
    cv_model.fit(X_train, y_train)
    oof_preds[val_idx] = cv_model.predict(X_val)

print("Generating plots...")

# Remove extreme outliers for plotting readability (e.g. keep actual price < $1000)
mask = y < 1000
y_plot = y[mask]
oof_plot = oof_preds[mask]

# 1. Predicted vs Actual
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_plot, y=oof_plot, alpha=0.3, color="teal")
max_val = max(y_plot.max(), oof_plot.max())
plt.plot([0, max_val], [0, max_val], color="red", linestyle="--")
plt.xlabel("Actual Price ($)", fontsize=12)
plt.ylabel("Predicted Price ($)", fontsize=12)
plt.title("Predicted vs. Actual Listing Price (Out-of-Fold Cross-Validation)", fontsize=14, pad=15)
plt.tight_layout()
pva_path = Path("./models/predicted_vs_actual_clean.png")
plt.savefig(pva_path, dpi=150)
plt.close()
print(f"[OK] Saved predicted vs actual plot to {pva_path}")

# 2. Residuals Plot
plt.figure(figsize=(10, 5))
residuals = y_plot - oof_plot
sns.scatterplot(x=oof_plot, y=residuals, alpha=0.3, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price ($)", fontsize=12)
plt.ylabel("Residual (Actual - Predicted) ($)", fontsize=12)
plt.title("Residuals vs. Predicted Values (Out-of-Fold Cross-Validation)", fontsize=14, pad=15)
plt.tight_layout()
res_path = Path("./models/residuals_plot_clean.png")
plt.savefig(res_path, dpi=150)
plt.close()
print(f"[OK] Saved residuals plot to {res_path}")

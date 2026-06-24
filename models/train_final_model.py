import os
import sys
import json
import pickle
import importlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

def train_final_model(dataset_path: Path, output_dir: Path):
    """
    Train the final model on the full dataset using the champion spec
    developed during the autoresearch loop.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    df_raw = pd.read_csv(dataset_path)
    
    # Path to champion model specs
    champion_dir = Path("./models/autoresearch/champion_model")
    
    has_champion = (champion_dir / "features.py").exists() and (champion_dir / "model_spec.json").exists()
    
    if has_champion:
        print("Found champion model from autoresearch history. Loading specifications...")
        # Import feature engineering dynamically
        sys.path.insert(0, str(champion_dir))
        if 'features' in sys.modules:
            sys.modules.pop('features')
        import features
        importlib.reload(features)
        
        df_engineered = features.feature_engineering(df_raw.copy())
        sys.path.remove(str(champion_dir))
        
        with open(champion_dir / "model_spec.json") as f:
            spec = json.load(f)
        model_type = spec["model_type"]
        hyperparams = spec["hyperparameters"]
        
        with open(champion_dir / "metrics.json") as f:
            champion_metrics = json.load(f)
            
        print(f"Champion model type: {model_type}")
        print(f"Champion parameters: {json.dumps(hyperparams, indent=2)}")
        print(f"Autoresearch cross-validation MAE: {champion_metrics['mae']:.2f}")
    else:
        print("No champion model found. Training a high-performance default baseline model...")
        model_type = "LightGBM" if lgb else ("XGBoost" if xgb else "RandomForest")
        hyperparams = {"n_estimators": 100, "learning_rate": 0.05} if model_type != "RandomForest" else {"n_estimators": 100}
        
        # Simple default baseline feature engineering
        def feature_engineering(df):
            # Fill NaNs
            df['bedrooms'] = df['bedrooms'].fillna(1)
            df['beds'] = df['beds'].fillna(1)
            df['host_listings_count'] = df['host_listings_count'].fillna(1)
            df['is_superhost'] = df['is_superhost'].fillna(False).astype(int)
            df['availability_rate'] = df['availability_rate'].fillna(0.5)
            
            # One-hot encode room_type
            df = pd.get_dummies(df, columns=['room_type'], drop_first=True)
            
            # Drop strings & IDs
            cols_to_drop = ['name', 'neighbourhood', 'neighbourhood_group', 'property_type', 'host_since', 'listing_id']
            df_numeric = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
            return df_numeric

        df_engineered = feature_engineering(df_raw.copy())
        
        # Save baseline features.py for consistency
        champion_dir.mkdir(parents=True, exist_ok=True)
        baseline_code = """def feature_engineering(df):
    import pandas as pd
    # Fill NaNs
    df['bedrooms'] = df['bedrooms'].fillna(1)
    df['beds'] = df['beds'].fillna(1)
    df['host_listings_count'] = df['host_listings_count'].fillna(1)
    df['is_superhost'] = df['is_superhost'].fillna(False).astype(int)
    df['availability_rate'] = df['availability_rate'].fillna(0.5)
    
    # One-hot encode room_type
    df = pd.get_dummies(df, columns=['room_type'], drop_first=True)
    
    # Drop strings & IDs
    cols_to_drop = ['name', 'neighbourhood', 'neighbourhood_group', 'property_type', 'host_since', 'listing_id']
    df_numeric = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df_numeric
"""
        with open(champion_dir / "features.py", 'w') as f:
            f.write(baseline_code)
            
        print(f"Default model type: {model_type}")
        champion_metrics = None

    # 1. Clean the engineered dataframe
    # Drop rows where the target is NaN (we cannot train on these)
    df_engineered = df_engineered.dropna(subset=["target_price"])
    
    # Separate features and target
    X = df_engineered.drop(columns=["target_price", "listing_id"], errors="ignore")
    y = df_engineered["target_price"]
    
    # 2. Robust Null Handling for features
    # Fill remaining NaNs in features with median
    if X.isnull().values.any():
        print("Warning: NaNs found in features. Filling with median...")
        X = X.fillna(X.median())

    # 3. Final safety check before fitting
    if np.isnan(y).any():
        raise ValueError("Target variable 'target_price' still contains NaNs after dropping!")
        
    print(f"Dataset shape: {X.shape}")
    
    # Instantiate and fit
    print("Fitting final model on full dataset...")
    if model_type == "LightGBM" and lgb:
        model = lgb.LGBMRegressor(**hyperparams, random_state=42, verbose=-1)
    elif model_type == "XGBoost" and xgb:
        model = xgb.XGBRegressor(**hyperparams, random_state=42, verbosity=0)
    else:
        rf_params = {k: int(v) for k, v in hyperparams.items() if k in ["n_estimators", "max_depth"]}
        model = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
        
    model.fit(X, y)
    
    # Save model binary
    model_bin_path = output_dir / "final_model.pkl"
    with open(model_bin_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] Serialized model saved to {model_bin_path}")
    
    # Compute full dataset metrics
    preds = model.predict(X)
    train_mae = mean_absolute_error(y, preds)
    train_rmse = np.sqrt(mean_squared_error(y, preds))
    train_r2 = r2_score(y, preds)
    
    # Save summary report
    summary_path = output_dir / "model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== AIRBNB PRICE PREDICTION CHAMPION MODEL SUMMARY ===\n")
        f.write(f"Trained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}\n")
        f.write(f"Features used ({len(X.columns)} total):\n")
        for col in X.columns:
            f.write(f"  - {col}\n")
        f.write("\n=== PERFORMANCE METRICS ON FULL DATASET ===\n")
        f.write(f"Mean Absolute Error (MAE): ${train_mae:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): ${train_rmse:.2f}\n")
        f.write(f"R-squared Score (R2): {train_r2:.4f}\n")
        if champion_metrics:
            f.write(f"\nAutoresearch Cross-Validation MAE: ${champion_metrics['mae']:.2f}\n")
            f.write(f"Autoresearch Cross-Validation R2: {champion_metrics['r2']:.4f}\n")
            
    print(f"[OK] Summary documentation generated at {summary_path}")

if __name__ == "__main__":
    ds_path = Path("./models/dataset.csv")
    out_dir = Path("./models")
    train_final_model(ds_path, out_dir)

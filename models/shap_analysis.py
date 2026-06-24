import os
import sys
import pickle
import importlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None

def run_shap_analysis(dataset_path: Path, model_path: Path, output_dir: Path):
    """
    Perform SHAP analysis on the trained final model.
    Saves SHAP summary plot explaining feature importance and impact directions.
    """
    if not shap:
        print("Error: SHAP library is not installed.")
        print("Please install it in your virtual environment by running: pip install shap")
        sys.exit(1)
        
    print(f"Loading final model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_final_model.py first.")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print(f"Loading dataset from {dataset_path}...")
    df_raw = pd.read_csv(dataset_path)
    
    # Import champion features
    champion_dir = Path("./models/autoresearch/champion_model")
    if not (champion_dir / "features.py").exists():
        raise FileNotFoundError("Champion features.py not found. Please run train_final_model.py first to establish model features.")
        
    sys.path.insert(0, str(champion_dir))
    if 'features' in sys.modules:
        sys.modules.pop('features')
    import features
    importlib.reload(features)
    
    df_engineered = features.feature_engineering(df_raw.copy())
    sys.path.remove(str(champion_dir))
    
    # Separate features
    X = df_engineered.drop(columns=["target_price", "listing_id"], errors="ignore")
    
    if X.isnull().values.any():
         X = X.fillna(X.median())
         
    print("Calculating SHAP values...")
    # Initialize TreeExplainer (works for LightGBM, XGBoost, RandomForest)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"TreeExplainer failed or not supported for this model type: {e}")
        print("Falling back to generic KernelExplainer (this might take longer)...")
        # Subsample for speed if kernel explainer is used
        X_sub = shap.sample(X, 100)
        explainer = shap.KernelExplainer(model.predict, X_sub)
        shap_values = explainer.shap_values(X_sub)
        X = X_sub

    # Save summary plot
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    
    # Support both multi-class and single-output structures
    if isinstance(shap_values, list):
        # For some configurations of ensemble tree models, shap returns a list of arrays
        shap.summary_plot(shap_values[0] if len(shap_values) > 0 else shap_values, X, show=False)
    else:
        shap.summary_plot(shap_values, X, show=False)
        
    plt.title("SHAP Feature Impact Summary (Price Prediction)", fontsize=14, pad=15)
    plt.tight_layout()
    plot_path = output_dir / "shap_summary_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"[OK] SHAP explainability plot saved successfully to {plot_path}")

if __name__ == "__main__":
    ds_path = Path("./models/dataset.csv")
    mod_path = Path("./models/final_model.pkl")
    out_dir = Path("./models")
    run_shap_analysis(ds_path, mod_path, out_dir)

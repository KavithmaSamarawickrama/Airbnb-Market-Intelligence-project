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

def run_shap_analysis():
    if not shap:
        print("Error: SHAP library is not installed.")
        sys.exit(1)
        
    model_path = Path("./models/prospective_model.pkl")
    dataset_path = Path("./models/dataset.csv")
    output_path = Path("./models/shap_summary_plot_clean.png")
    
    print(f"Loading prospective model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print(f"Loading dataset from {dataset_path}...")
    df_raw = pd.read_csv(dataset_path)
    
    # Import champion features
    champion_dir = Path("./models/autoresearch/champion_model")
    sys.path.insert(0, str(champion_dir))
    import features
    importlib.reload(features)
    
    df_engineered = features.feature_engineering(df_raw.copy())
    sys.path.remove(str(champion_dir))
    
    # Separate features and target
    X = df_engineered.drop(columns=["target_price", "listing_id"], errors="ignore")
    df_engineered = df_engineered.dropna(subset=["target_price"])
    X = df_engineered.drop(columns=["target_price", "listing_id"], errors="ignore")
    
    # Drop all price-derived or historical booking-revenue features
    leak_cols = ["avg_price", "est_revenue", "revenue_per_available_day", "revenue_per_review"]
    X_prospective = X.drop(columns=[c for c in leak_cols if c in X.columns], errors="ignore")
    
    print("Calculating SHAP values...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_prospective)
    except Exception as e:
        print(f"TreeExplainer failed: {e}. Falling back to KernelExplainer...")
        X_sub = shap.sample(X_prospective, 100)
        explainer = shap.KernelExplainer(model.predict, X_sub)
        shap_values = explainer.shap_values(X_sub)
        X_prospective = X_sub

    # Save summary plot
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0] if len(shap_values) > 0 else shap_values, X_prospective, show=False)
    else:
        shap.summary_plot(shap_values, X_prospective, show=False)
        
    plt.title("SHAP Feature Impact (Prospective Price Prediction Model)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"[OK] Clean SHAP plot saved successfully to {output_path}")

if __name__ == "__main__":
    run_shap_analysis()

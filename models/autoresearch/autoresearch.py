import os
import sys
import json
import time
import shutil
import argparse
import importlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Try imports, fallback gracefully if not installed
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# LLM Libraries
try:
    import openai
    from openai import AzureOpenAI
except ImportError:
    openai = None
    AzureOpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Configure plot styles
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]


SYSTEM_PROMPT = """You are a world-class Machine Learning Research Agent implementing Andrej Karpathy's autoresearch methodology.
Your objective is to find the best possible regression model to predict Airbnb prices (`target_price`) using a tabular dataset.

You will be given:
1. The schema and description of the base features.
2. A history of all previous runs, including the code proposed, model selected, and the validation metrics achieved (MAE, RMSE, MAPE, R2).
3. The details of the current best performing model ("the champion").

Your task is to propose the next experiment in the research loop. An experiment consists of:
1. A reasoning explaining your hypothesis (what features or parameters you are testing and why).
2. The model family to use ("LightGBM", "XGBoost", "RandomForest", or "Linear").
3. A dictionary of hyperparameters for the chosen model.
4. Python code for a function `feature_engineering(df)` that takes a pandas DataFrame as input and returns a modified DataFrame.

Guidelines for Feature Engineering Code:
- You must define EXACTLY a function: `def feature_engineering(df):`
- Inside this function, perform feature creation, handling of null values, extraction of text features from the `name` column, encoding of categorical variables (`room_type`, `property_type`, `neighbourhood`), etc.
- Do NOT perform scaling inside this function. Keep the numerical features in their natural units.
- Do NOT cause data leakage! If you compute target encoding, you must use out-of-fold averages or other safe techniques. Or focus on non-leakage features like keyword counts from the listing name, log transforms of skewed numerical values, geographic distance calculations (from centroid lat/lon or city center), or categorical frequency mapping.
- The returned DataFrame must contain only numeric columns, plus the index (no strings or datetimes, except 'target_price' which is handled automatically).
- Keep the code clean, robust, and handle potential NaN values safely (e.g., using `.fillna()`).

You must respond with a single, valid JSON object with the following keys. Do NOT wrap it in markdown code blocks:
{
  "reasoning": "Detailed explanation of your hypothesis and strategy.",
  "model_type": "LightGBM",  // Choose from: "LightGBM", "XGBoost", "RandomForest", "Linear"
  "hyperparameters": {
     // Model parameters, e.g., for LightGBM:
     "n_estimators": 200,
     "learning_rate": 0.05,
     "max_depth": 6,
     "num_leaves": 31
  },
  "feature_engineering_code": "def feature_engineering(df):\n    import pandas as pd\n    import numpy as np\n    # code here\n    # drop columns that are not numeric or target\n    cols_to_drop = ['name', 'neighbourhood', 'neighbourhood_group', 'room_type', 'property_type', 'host_since']\n    df_numeric = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')\n    return df_numeric"
}
"""

BASE_FEATURE_INFO = """
Columns in the input dataset:
- `listing_id`: unique identifier for the listing (drop for training)
- `name`: free text title of the Airbnb listing (useful for extracting keywords like 'luxury', 'cozy', 'apartment')
- `room_type`: categorical (e.g. Entire home/apt, Private room, Shared room, Hotel room)
- `property_type`: detailed property category (e.g. Entire rental unit, Private room in residential home)
- `accommodates`: maximum capacity (int)
- `bedrooms`: count of bedrooms (float)
- `beds`: count of beds (float)
- `latitude` / `longitude`: geocoordinates of the listing
- `is_superhost`: host status (boolean)
- `host_listings_count`: total listings managed by host (int)
- `host_since`: date host registered (datetime)
- `host_tenure_days`: calculated tenure in days relative to 2026-06-23 (float)
- `neighbourhood` / `neighbourhood_group`: geographic neighborhood names (categorical)
- `availability_rate`: percentage of days available in calendar year (0.0 to 1.0)
- `est_revenue`: total estimated revenue from calendar bookings (float)
- `total_reviews`: total guest reviews count (int)
- `target_price`: target variable, median listing price across active days (float)
"""


class AutoresearchLoop:
    def __init__(self, provider: str, iterations: int):
        self.provider = provider
        self.iterations = iterations
        self.history_file = Path("./models/autoresearch/history.json")
        self.history = []
        self._load_history()
        self.dataset_path = Path("./models/dataset.csv")
        
        # Load dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Please run models/prepare_data.py first.")
        self.df_raw = pd.read_csv(self.dataset_path)
        print(f"Loaded training dataset with {len(self.df_raw)} records.")

        # Initialize LLM client
        self._init_llm_client()

    def _load_history(self):
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    self.history = json.load(f)
                print(f"Loaded history with {len(self.history)} previous experiments.")
            except Exception as e:
                print(f"Warning: Failed to load history: {e}. Starting fresh.")
                self.history = []
        else:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history = []

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _init_llm_client(self):
        if self.provider == "azure":
            if not AzureOpenAI:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if not all([api_key, endpoint, deployment]):
                raise ValueError("Missing Azure OpenAI environment variables (.env). Required: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")
            
            self.azure_client = AzureOpenAI(
                api_key=api_key,
                api_version="2023-05-15",
                azure_endpoint=endpoint
            )
            self.azure_deployment = deployment
            print(f"Azure OpenAI client initialized using deployment: {deployment}")
        elif self.provider == "gemini":
            if not genai:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Missing GEMINI_API_KEY environment variable (.env).")
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            print("Gemini model client initialized (gemini-1.5-pro)")
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _query_llm(self, prompt: str) -> str:
        if self.provider == "azure":
            response = self.azure_client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        elif self.provider == "gemini":
            full_prompt = f"{SYSTEM_PROMPT}\n\nUSER PROMPT:\n{prompt}"
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config={"temperature": 0.2}
            )
            return response.text.strip()

    def _build_model(self, model_type: str, hyperparams: dict):
        """Construct model with grace fallbacks if libraries are missing."""
        if model_type == "LightGBM":
            if lgb:
                return lgb.LGBMRegressor(**hyperparams, random_state=42, verbose=-1)
            else:
                print("Warning: LightGBM not installed. Falling back to RandomForest.")
                model_type = "RandomForest"
                
        if model_type == "XGBoost":
            if xgb:
                return xgb.XGBRegressor(**hyperparams, random_state=42, verbosity=0)
            else:
                print("Warning: XGBoost not installed. Falling back to RandomForest.")
                model_type = "RandomForest"

        if model_type == "RandomForest":
            # Map params if LLM supplied boosted tree params
            rf_params = {}
            if "n_estimators" in hyperparams:
                rf_params["n_estimators"] = int(hyperparams["n_estimators"])
            if "max_depth" in hyperparams:
                rf_params["max_depth"] = int(hyperparams["max_depth"]) if hyperparams["max_depth"] else None
            return RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)

        if model_type == "Linear":
            return Ridge(alpha=hyperparams.get("alpha", 1.0))
            
        # Global fallback
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, model_type: str, hyperparams: dict):
        """Run 5-Fold Cross Validation and calculate metrics."""
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        maes = []
        rmses = []
        mapes = []
        r2s = []
        
        # OOF predictions
        oof_preds = np.zeros(len(y))
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self._build_model(model_type, hyperparams)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            oof_preds[val_idx] = preds
            
            maes.append(mean_absolute_error(y_val, preds))
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            mapes.append(np.mean(np.abs((y_val - preds) / y_val)) * 100)
            r2s.append(r2_score(y_val, preds))
            
        # Fit final model on all data to extract feature importances
        final_model = self._build_model(model_type, hyperparams)
        final_model.fit(X, y)
        
        importances = None
        if hasattr(final_model, "feature_importances_"):
            importances = dict(zip(X.columns, final_model.feature_importances_.tolist()))
        elif hasattr(final_model, "coef_"):
            importances = dict(zip(X.columns, final_model.coef_.tolist()))
            
        metrics = {
            "mae": float(np.mean(maes)),
            "rmse": float(np.mean(rmses)),
            "mape": float(np.mean(mapes)),
            "r2": float(np.mean(r2s))
        }
        
        return metrics, oof_preds, importances, final_model

    def _generate_plots(self, y: pd.Series, oof_preds: np.ndarray, importances: dict, output_dir: Path):
        """Generate residuals and feature importance charts."""
        # 1. Residuals Plot
        plt.figure(figsize=(10, 5))
        residuals = y - oof_preds
        sns.scatterplot(x=oof_preds, y=residuals, alpha=0.3, color="purple")
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted Price")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Residuals vs. Predicted Values")
        plt.tight_layout()
        plt.savefig(output_dir / "residuals_plot.png", dpi=150)
        plt.close()
        
        # 2. Predicted vs Actual
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y, y=oof_preds, alpha=0.3, color="teal")
        max_val = max(y.max(), oof_preds.max())
        plt.plot([0, max_val], [0, max_val], color="red", linestyle="--")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Predicted vs. Actual Listing Price")
        plt.tight_layout()
        plt.savefig(output_dir / "predicted_vs_actual.png", dpi=150)
        plt.close()

        # 3. Feature Importance Plot
        if importances:
            imp_df = pd.DataFrame(list(importances.items()), columns=["feature", "importance"])
            imp_df["importance_abs"] = imp_df["importance"].abs()
            top_imp = imp_df.sort_values(by="importance_abs", ascending=False).head(15)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top_imp, x="importance", y="feature", palette="viridis")
            plt.title("Top 15 Feature Importances/Coefficients")
            plt.xlabel("Importance Score")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(output_dir / "feature_importance.png", dpi=150)
            plt.close()

    def run_loop(self):
        print(f"Starting autoresearch loop with {self.iterations} iterations.")
        
        for i in range(self.iterations):
            run_id = len(self.history) + 1
            print(f"\n--- Run {run_id} / Iteration {i+1} ---")
            
            # Find current champion (lowest MAE)
            champion = None
            if self.history:
                champion = min(self.history, key=lambda x: x["metrics"]["mae"])
                print(f"Current Champion: Run {champion['run_id']} (MAE: {champion['metrics']['mae']:.2f}, R2: {champion['metrics']['r2']:.4f})")
            else:
                print("No runs recorded yet. Executing baseline initial run.")
                
            # Build prompt history context
            history_context = []
            for r in self.history[-5:]: # Feed back the last 5 runs to manage prompt token limits
                history_context.append({
                    "run_id": r["run_id"],
                    "model_type": r["model_type"],
                    "metrics": r["metrics"],
                    "reasoning": r["reasoning"]
                })
                
            prompt = f"""
            Available Base Dataset Information:
            {BASE_FEATURE_INFO}
            
            Experiment History (most recent runs):
            {json.dumps(history_context, indent=2)}
            
            Current Champion Run:
            {json.dumps(champion, indent=2) if champion else "None (No runs yet)"}
            
            Based on the history and the base features, generate the next experiment. 
            Propose feature engineering code and model configuration. Try to improve the MAE and R2.
            If this is the first run, implement a simple numerical baseline.
            If there are previous runs, inspect what worked and propose improvements (e.g. creating ratio terms, extracting text indicators from name, etc.).
            
            Ensure your response is valid JSON matching the exact schema specified in the instructions. Return only the JSON object.
            """
            
            print("Querying LLM for next experiment proposal...")
            llm_text = self._query_llm(prompt)
            
            # Parse LLM response
            try:
                # Remove json block wrapping if LLM returned it
                clean_text = llm_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()
                
                proposal = json.loads(clean_text)
            except Exception as e:
                print(f"Error parsing LLM response as JSON: {e}")
                print(f"Raw Response:\n{llm_text}")
                print("Retrying with a simpler query...")
                continue
                
            print(f"Hypothesis: {proposal.get('reasoning', 'No reasoning provided')}")
            print(f"Selected Model: {proposal.get('model_type')}")
            
            # Create Run Directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(f"./models/autoresearch/run_{timestamp}_{run_id}")
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save feature engineering code to dynamic file
            code_file = run_dir / "features.py"
            with open(code_file, 'w') as f:
                f.write(proposal.get("feature_engineering_code", ""))
                
            # Dynamic imports
            try:
                # Add run_dir to path and import
                sys.path.insert(0, str(run_dir))
                if 'features' in sys.modules:
                    sys.modules.pop('features')
                    
                import features
                importlib.reload(features)
                
                # Execute feature engineering
                print("Running feature engineering code on dataset...")
                # Work on a copy of raw df
                df_engineered = features.feature_engineering(self.df_raw.copy())
                
                # Remove sys path
                sys.path.remove(str(run_dir))
            except Exception as e:
                print(f"Failed to execute proposed feature engineering code: {e}")
                shutil.rmtree(run_dir)
                continue
                
            # Validate generated DataFrame
            if "target_price" not in df_engineered.columns:
                print("Error: The engineered DataFrame must contain 'target_price' column.")
                shutil.rmtree(run_dir)
                continue
                
            # Separate features and target
            X = df_engineered.drop(columns=["target_price", "listing_id"], errors="ignore")
            y = df_engineered["target_price"]
            
            # Ensure all columns are numeric
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                print(f"Error: Engineered DataFrame contains non-numeric feature columns: {non_numeric_cols}")
                shutil.rmtree(run_dir)
                continue
                
            # Fill remaining NaNs if any
            if X.isnull().values.any():
                print("Warning: Input features contain NaN values. Filling with column medians.")
                X = X.fillna(X.median())
                
            # Run Evaluation
            print(f"Training and validating model on {len(X)} records with {len(X.columns)} features...")
            model_type = proposal.get("model_type", "RandomForest")
            hyperparams = proposal.get("hyperparameters", {})
            
            try:
                metrics, oof_preds, importances, final_model = self._evaluate_model(X, y, model_type, hyperparams)
            except Exception as e:
                print(f"Error training/evaluating model: {e}")
                shutil.rmtree(run_dir)
                continue
                
            print(f"Metrics Achieved: MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%, R2: {metrics['r2']:.4f}")
            
            # Save Run Artifacts
            with open(run_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
                
            with open(run_dir / "reasoning.txt", 'w') as f:
                f.write(proposal.get("reasoning", ""))
                
            self._generate_plots(y, oof_preds, importances, run_dir)
            
            # Record in global log
            history_record = {
                "run_id": run_id,
                "timestamp": timestamp,
                "model_type": model_type,
                "hyperparameters": hyperparams,
                "metrics": metrics,
                "reasoning": proposal.get("reasoning", ""),
                "features_used": X.columns.tolist()
            }
            self.history.append(history_record)
            self._save_history()
            
            # Mark champion update
            if champion and metrics["mae"] < champion["metrics"]["mae"]:
                print(f"🎉 New Champion! Beat previous run {champion['run_id']} MAE by {champion['metrics']['mae'] - metrics['mae']:.2f}")
                # Save as final_model reference
                champion_dir = Path("./models/autoresearch/champion_model")
                champion_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy current run files to champion
                shutil.copy2(code_file, champion_dir / "features.py")
                with open(champion_dir / "metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
                with open(champion_dir / "model_spec.json", 'w') as f:
                    json.dump({
                        "model_type": model_type,
                        "hyperparameters": hyperparams
                    }, f, indent=2)
                    
        print("\nAutoresearch loop finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Karpathy-style Autoresearch loop for Airbnb price prediction.")
    parser.add_argument("--provider", type=str, default="azure", choices=["azure", "gemini"],
                        help="LLM Provider to orchestrate the loop ('azure' or 'gemini').")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of search iterations to perform.")
    args = parser.parse_args()

    # Load environment
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=dotenv_path)

    loop = AutoresearchLoop(provider=args.provider, iterations=args.iterations)
    loop.run_loop()

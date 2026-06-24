# Guide: Executing the ML Autoresearch Loop

This directory contains the pipeline to automate machine learning model research for Airbnb listing price prediction. Following Andrej Karpathy's autoresearch methodology, the agent runs in a continuous loop: querying an LLM (Azure OpenAI GPT-4o or Google Gemini) to generate feature engineering code and hyperparameters, executing the training job dynamically using 5-fold cross-validation, plotting metrics, logging outputs, and feeding results back to the LLM to guide subsequent iterations.

---

## 📋 Setup Instructions

### 1. Install Dependencies
Activate your virtual environment and install the required modeling and LLM orchestration packages:
```powershell
# Activate venv
venv\Scripts\activate

# Install requirements
pip install scikit-learn lightgbm xgboost shap openai google-generativeai matplotlib seaborn
```

### 2. Configure Environment Variables
Open your `.env` file in the project root and provide the API credentials.

For **Azure OpenAI (GPT-4o)**, add:
```
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-gpt-4o-deployment-name
```

For **Google Gemini**, add:
```
GEMINI_API_KEY=your-gemini-api-key
```

---

## 🚀 Execution Workflow

### Step 1: Prepare the Dataset
Query the local conformed Gold layer tables in DuckDB to generate the listing-level tabular dataset:
```powershell
python models/prepare_data.py
```
This generates `models/dataset.csv` with ~7,800 records and features.

### Step 2: Run the Autoresearch Loop
You can let the agent run in a loop for hours to explore hundreds of combinations. Run the command specifying the LLM provider and number of search iterations:

Using **Azure OpenAI (GPT-4o)**:
```powershell
python models/autoresearch/autoresearch.py --provider azure --iterations 50
```

Using **Google Gemini**:
```powershell
python models/autoresearch/autoresearch.py --provider gemini --iterations 50
```

#### What the Agent Does in Each Run:
1. Reads previous run history from `models/autoresearch/history.json`.
2. Asks the LLM to formulate a new hypothesis based on what has succeeded or failed.
3. Receives a proposed python function `feature_engineering(df)` and model parameters.
4. Dynamically imports and runs the feature code inside a unique folder `models/autoresearch/run_[timestamp]_[run_id]/`.
5. Evaluates model performance using 5-Fold Cross-Validation.
6. Computes MAE, RMSE, MAPE, and R2 metrics.
7. Renders predicted-vs-actual, residual distribution, and feature importance charts.
8. If the run achieves the lowest MAE, it copies the code and specs to `models/autoresearch/champion_model/`.

### Step 3: Train the Final Champion Model
Once you stop the loop or it finishes, retrieve the best feature code and hyperparameters from the champion folder and fit it on the full dataset to serialize the model binary:
```powershell
python models/train_final_model.py
```
This writes:
- `models/final_model.pkl`: Serialized Python model binary.
- `models/model_summary.txt`: A detailed summary of the training run, listing engineered columns, parameters, and final scores.

### Step 4: Run SHAP Explainability Analysis
To understand feature impacts and interpret model predictions:
```powershell
python models/shap_analysis.py
```
This generates `models/shap_summary_plot.png`, which shows the contribution (magnitude and direction) of each feature to listing prices.

---

## 📊 Directory & Log Structure

```
models/
├── dataset.csv                 # Aggregate tabular data
├── final_model.pkl             # Serialized champion model
├── model_summary.txt           # Champion metrics & parameters summary
├── shap_summary_plot.png       # SHAP importance/direction plot
├── prepare_data.py             # Prepares dataset from DuckDB
├── train_final_model.py        # Serializes the final model
├── shap_analysis.py            # Generates SHAP explanations
└── autoresearch/
    ├── README.md               # This guide
    ├── autoresearch.py         # Main agent loop script
    ├── history.json            # Global run history log
    ├── champion_model/         # Folder pointing to the best run
    │   ├── features.py         # Best feature engineering code
    │   ├── model_spec.json     # Best hyperparameters & model type
    │   └── metrics.json        # Best cross-validation metrics
    └── run_[timestamp]_[id]/   # Workspace created for a single run
        ├── features.py         # Code executed for this experiment
        ├── metrics.json        # Evaluation scores
        ├── reasoning.txt       # LLM research hypothesis
        ├── residuals_plot.png  # Error analysis plot
        ├── predicted_vs_actual.png
        └── feature_importance.png # Importance or coefficient plot
```

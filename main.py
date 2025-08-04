# main.py

from src.preprocessing import preprocess_data
from src.feature_engineering import generate_features
from src.modeling import train_models
from src.explainability import shap_analysis

import os

def main():
    print("🚀 Starting EV Fault Detection pipeline...")

    # Create folders if they do not exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Step 1: Data cleaning
    print("📦 Step 1: Preprocessing...")
    preprocess_data(
        input_csv="data/NEV_fault_dataset new (3).csv",
        output_csv="data/preprocessed.csv"
    )

    # Step 2: Feature Engineering
    print("🧠 Step 2: Feature Engineering...")
    generate_features(
        input_csv="data/preprocessed.csv",
        output_csv="data/features_enriched.csv"
    )

    # Step 3: Model training and performance comparison
    print("🤖 Step 3: Training models...")
    train_models(
        input_csv="data/features_enriched.csv"
    )

    # Step 4: Explain predictions with SHAP
    print("🔍 Step 4: SHAP Explainability...")
    shap_analysis(
        model_path="models/original_XGBoost.joblib",
        data_path="data/features_enriched.csv",
        output_prefix="xgb"
    )

    print("✅ Full pipeline complete! Results saved in 'models' and 'plots' folders.")

if __name__ == "__main__":
    main()

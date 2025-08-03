import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def shap_analysis(model_path, data_path, output_prefix):
    # Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    if 'Fault_Level' in df.columns:
        y = df['Fault_Level']
        X = df.drop(columns=['Fault_Level'])
    else:
        raise ValueError("Missing 'Fault_Level' column.")

    # Scale adjustment
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # General summary graph
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"plots/{output_prefix}_shap_summary.png")
    plt.close()

    # One example forecast graph
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(f"plots/{output_prefix}_shap_waterfall_0.png")
    plt.close()

    print(f"✅ SHAP explainability completed. Results saved to plots/")

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    shap_analysis(
        model_path="models/original_XGBoost.joblib",
        data_path="data/features_enriched.csv",
        output_prefix="xgb"
    )

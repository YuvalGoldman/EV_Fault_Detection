import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(path):
    df = pd.read_csv(path)
    if 'Fault_Level' in df.columns:
        y = df['Fault_Level']
        X = df.drop('Fault_Level', axis=1)
    else:
        raise ValueError("Missing 'Fault_Level' column.")
    return X, y

def train_models(X_train, X_test, y_train, y_test, prefix, results):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[f"{prefix}_{name}"] = {
            "accuracy": acc,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"]
        }

        joblib.dump(model, f"models/{prefix}_{name}.joblib")

        # Feature Importance (XGBoost only)
        if name == "XGBoost":
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            fi_df.to_csv(f"plots/{prefix}_xgb_importance.csv", index=False)

            # Plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=fi_df.head(15), palette="viridis")
            plt.title(f"{prefix} - Top 15 Feature Importances (XGBoost)")
            plt.tight_layout()
            plt.savefig(f"plots/{prefix}_xgb_importance.png")
            plt.close()

def model_pipeline():
    X, y = load_data("data/features_enriched.csv")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

    results = {}
    train_models(pd.DataFrame(X_train, columns=X.columns), pd.DataFrame(X_test, columns=X.columns), y_train, y_test, "original", results)

    # PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.3, stratify=y, random_state=42)

    train_models(pd.DataFrame(X_train_pca), pd.DataFrame(X_test_pca), y_train, y_test, "pca", results)

    # Model comparison
    results_df = pd.DataFrame(results).T
    results_df.to_csv("plots/model_comparison.csv")

    # Comparison chart
    plt.figure(figsize=(10, 5))
    results_df[["accuracy", "precision", "recall", "f1"]].plot(kind="bar")
    plt.title("🔍 השוואת ביצועים בין מודלים")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png")
    plt.close()

    print("✅ Modeling completed. Results saved to 'plots/' and models/")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    model_pipeline()

import pandas as pd
import numpy as np
import os


def generate_features(input_path, output_path):
    df = pd.read_csv(input_path)

    new_features = {}

    # 1. The differences between Main and Sub
    for col in df.columns:
        if col.endswith("_main"):
            base = col.replace("_main", "")
            sub_col = f"{base}_sub"
            if sub_col in df.columns:
                new_name = f"{base}_diff"
                df[new_name] = df[col] - df[sub_col]
                new_features[new_name] = "∆ Main-Sub"

    # 2. Change between rows (temp, voltage, current only)
    for col in df.columns:
        if any(sensor in col for sensor in ["Temperature", "Voltage", "Current"]) and col.endswith("_main"):
            df[f"{col}_delta"] = df[col].diff()

    # 3. Average between Main and Sub
    for col in df.columns:
        if col.endswith("_main"):
            base = col.replace("_main", "")
            sub_col = f"{base}_sub"
            if sub_col in df.columns:
                avg_name = f"{base}_avg"
                df[avg_name] = (df[col] + df[sub_col]) / 2

    # Removing first lines with NaN (from diff())
    df.dropna(inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Feature engineering complete. Saved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/processed_data.csv"
    output_file = "data/features_enriched.csv"
    generate_features(input_file, output_file)

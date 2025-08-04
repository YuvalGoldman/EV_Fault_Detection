import pandas as pd
import numpy as np
import os


def load_and_process_data(input_path, output_path):
    # Load the data
    df_raw = pd.read_csv(input_path, header=None)
    
    # Improvised column names
    num_features = df_raw.shape[1]
    column_names = [f"Feature_{i}" for i in range(num_features)]
    df_raw.columns = column_names

    # Split into a primary and secondary component in each column
    new_columns = {}
    for col in df_raw.columns:
        df_raw[[f"{col}_main", f"{col}_sub"]] = df_raw[col].astype(str).str.split(".", expand=True).iloc[:, :2]
        new_columns[f"{col}_main"] = f"{col}_main"
        new_columns[f"{col}_sub"] = f"{col}_sub"
        df_raw.drop(columns=[col], inplace=True)

    df_split = df_raw.copy()

    # Convert everything to float, errors -> NaN
    df_split = df_split.apply(pd.to_numeric, errors='coerce')

    # Convert everything to float, errors -> NaN
    for col in df_split.columns:
        for i in range(1, len(df_split)-1):
            if pd.isna(df_split.loc[i, col]):
                before = df_split.loc[i - 1, col]
                after = df_split.loc[i + 1, col]
                if not pd.isna(before) and not pd.isna(after):
                    df_split.loc[i, col] = (before + after) / 2

    # Save processed file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_split.to_csv(output_path, index=False)
    print(f"✅ Data preprocessing complete. Saved to: {output_path}")


# The file can be run as a standalone
if __name__ == "__main__":
    input_csv = "data/NEV_fault_dataset.csv"
    output_csv = "data/processed_data.csv"
    load_and_process_data(input_csv, output_csv)

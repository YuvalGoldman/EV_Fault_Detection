import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
df = pd.read_csv("data/processed_data.csv")

# Descriptive statistics
stats = df.describe().T
stats["median"] = df.median()
stats.to_csv("plots/feature_statistics.csv")

# Histograms
os.makedirs("plots/histograms", exist_ok=True)
for col in df.columns:
    plt.figure(figsize=(5, 3))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Histogram - {col}")
    plt.tight_layout()
    plt.savefig(f"plots/histograms/{col}_hist.png")
    plt.close()

# Boxplot
os.makedirs("plots/boxplots", exist_ok=True)
for col in df.columns:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f"Boxplot - {col}")
    plt.tight_layout()
    plt.savefig(f"plots/boxplots/{col}_box.png")
    plt.close()

# Heatmap 
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("✅ EDA complete: Statistics and graphs saved in a folder plots/")

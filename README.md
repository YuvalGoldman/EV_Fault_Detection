# 🔧 EV Fault Detection Using Machine Learning

A full-stack Data Science project for classifying electric vehicle (EV) fault levels using multivariate sensor readings. The pipeline covers end-to-end processing: from raw data parsing to model training, explainable AI, and live predictions via a Streamlit dashboard.

---

## 📊 Project Overview

This project predicts the **fault level (0–3)** of an EV system based on structured sensor inputs represented as compound vectors.

### ✅ Key Capabilities
- 🧹 **Data Cleaning** – parsing custom vector format, handling missing values
- 🧠 **Feature Engineering** – generating statistical features like deltas, averages, and differences
- 🔬 **Dimensionality Reduction** – using PCA & SVD for analysis and model comparison
- 🤖 **Model Training** – classification using Random Forest and XGBoost
- 🔍 **Explainability** – interpreting model behavior using SHAP
- 📈 **Evaluation** – comparison across models and feature sets
- 🧩 **Streamlit App** – interactive dashboard for live predictions (optional)

---

## 🧪 Model Performance Summary

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Random Forest      | 0.91     | 0.91      | 0.91   | 0.91     |
| XGBoost            | 0.93     | 0.93      | 0.93   | 0.93     |
| PCA + XGBoost      | 0.89     | 0.89      | 0.89   | 0.89     |

📊 **Visual Comparison**: See `plots/model_comparison.png` for a graphical view of performance.

---

## 🔍 SHAP Explainability

![SHAP Summary Plot](plots/xgb_shap_summary.png)

Key insights from SHAP analysis:
- `Voltage_diff`, `Current_avg`, and `Temperature_delta` are among the most impactful features.
- Helps interpret model predictions for each fault level class.

---

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Data Processing**: `pandas`, `numpy`
- **Modeling**: `scikit-learn`, `XGBoost`
- **Visualization**: `matplotlib`, `seaborn`
- **Explainability**: `SHAP`
- **Deployment**: `Streamlit` (optional dashboard)

---

## ⚙️ Installation & Setup

# 1. Clone the repository
git clone https://github.com/yourusername/EV_Fault_Detection.git
cd EV_Fault_Detection

# 2. Install required packages
pip install -r requirements.txt

# 3. Place the raw data file in the 'data' folder:
#    data/NEV_fault_dataset new (3).csv

# 4. Run the full pipeline
python main.py

---

## 📎 Streamlit Dashboard


streamlit run app.py

## 📁 Repository Structure

EV_Fault_Detection/
│
├── data/                  # Input & output CSVs
├── models/                # Trained model files
├── plots/                 # Evaluation & SHAP graphs
├── src/                   # Core pipeline modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── explainability.py
├── app.py                 # Streamlit dashboard (optional)
├── main.py                # Pipeline orchestrator
└── requirements.txt       # Project dependencies

## 👨‍💻 Author
Yuval Goldman
Data Scientist | Biologist | Agronomist
🔗 LinkedIn • GitHub: @YuvalGoldman

## 📌 License
This project is for educational and professional portfolio purposes. Please contact the author for reuse permissions.

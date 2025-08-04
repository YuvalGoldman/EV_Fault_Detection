# 🔧 EV Fault Detection Using Machine Learning

A full Data Science project for detecting fault levels in electric vehicles using multivariate sensor data, machine learning, and explainable AI (XAI).

## 📊 Project Overview
This pipeline classifies the **Fault Level** of an electric vehicle based on vector-like sensor readings separated by periods.

The project includes:
- Data cleaning and extrapolation
- Feature engineering (diffs, deltas, averages)
- PCA & SVD for dimensionality analysis
- Classification models: Random Forest, XGBoost
- SHAP explainability
- Streamlit dashboard for live predictions

---

## 🧪 Models & Results

| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| Random Forest    | 0.91     | 0.91      | 0.91   | 0.91     |
| XGBoost          | 0.93     | 0.93      | 0.93   | 0.93     |
| PCA + XGBoost    | 0.89     | 0.89      | 0.89   | 0.89     |

> Full visual comparison in [`plots/model_comparison.png`](plots/model_comparison.png)

---

## 🧠 SHAP Explainability

![SHAP summary](plots/xgb_shap_summary.png)

> SHAP reveals that **Voltage_diff**, **Current_avg**, and **Temperature_delta** are highly influential.

---

## 📎 Streamlit Dashboard

```bash
streamlit run app.py

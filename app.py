import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Load model
MODEL_PATH = "models/original_XGBoost.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="EV Fault Detector", layout="wide")
st.title("🚗 EV Fault Detection System")

uploaded_file = st.file_uploader("העלה קובץ CSV של רכבים לבדיקה", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 הצצה בנתונים:")
    st.dataframe(df.head())

    if 'Fault_Level' in df.columns:
        df = df.drop(columns=['Fault_Level'])

    # Prediciton
    preds = model.predict(df)
    df_results = df.copy()
    df_results["Predicted_Fault"] = preds

    st.subheader("🔍 תוצאות חיזוי")
    st.dataframe(df_results)

    # Feature Importance
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": df.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))

    # SHAP
    st.subheader("🔎 ניתוח תחזית ב-SHAP")
    explainer = shap.Explainer(model, df)
    shap_values = explainer(df)

    st.markdown("**גרף סיכום SHAP**")
    fig1 = shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)

    st.markdown("**Waterfall לנתון ראשון**")
    fig2 = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)

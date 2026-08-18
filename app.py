import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="M.Tech ML Workspace", layout="wide")
st.title("📊 ML Model Evaluation Dashboard")
st.write("Upload your experiment test data to view your evaluation metrics.")

st.sidebar.header("Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV Data", type=["csv"])

model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl"
}

selected_model_name = st.sidebar.selectbox("Choose Classification Model", list(model_options.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    # Strip any potential hidden quotes from headers
    test_df.columns = test_df.columns.str.replace('"', '').str.replace("'", "").str.strip()
    
    target_col = 'y' if 'y' in test_df.columns else test_df.columns[-1]
    
    st.success(f"Successfully loaded dataset with {test_df.shape[0]} rows.")
    
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(str).str.replace('"', '').str.replace("'", "").str.strip().str.lower()
    y_test = y_test.map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    
    # Paths mappings
    model_path = os.path.join("models", model_options[selected_model_name])
    scaler_path = os.path.join("models", "scaler.pkl")
    cols_path = os.path.join("models", "feature_columns.pkl")
    
    if os.path.exists(model_path) and os.path.exists(cols_path):
        model = joblib.load(model_path)
        expected_columns = joblib.load(cols_path)
        
        # Match training encoding state perfectly
        X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True, dtype=int)
        
        # Reindex to ensure structural column equivalence with training state
        for col in expected_columns:
            if col not in X_test_encoded.columns:
                X_test_encoded[col] = 0
        X_test_encoded = X_test_encoded[expected_columns].astype(float)
        
        # Apply transformation scaling if required
        if selected_model_name in ["Logistic Regression", "kNN", "Naive Bayes"] and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_eval = pd.DataFrame(scaler.transform(X_test_encoded), columns=expected_columns)
        else:
            X_eval = X_test_encoded
            
        # Run inference
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Metrics Calculations
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Display Core Metrics Layout
        st.subheader(f"Metrics Output: {selected_model_name}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        st.write("---")
        layout_col1, layout_col2 = st.columns(2)
        
        with layout_col1:
            st.subheader("📋 Classification Text Report")
            report_str = classification_report(y_test, y_pred, zero_division=0)
            st.text(report_str)
            
        with layout_col2:
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual Negative (0)", "Actual Positive (1)"], 
                                 columns=["Predicted Negative (0)", "Predicted Positive (1)"])
            st.dataframe(cm_df, use_container_width=True)
            
    else:
        st.error("Model artifacts not found. Please execute train.py first to generate all models.")
else:
    st.info("Please upload your generated 'test_data.csv' file via the sidebar to initiate model metrics computation.")

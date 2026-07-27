# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)



st.set_page_config(
    page_title="ML Classification App",
    layout="wide"
)


st.title(
    "Breast Cancer Classification using Machine Learning"
)


st.write(
"""
This application compares multiple classification
models trained on the Breast Cancer Wisconsin dataset.
"""
)



# -------------------------
# Model Loading
# -------------------------


models = {

"Logistic Regression":
"models/logistic_regression.pkl",

"Decision Tree":
"models/decision_tree.pkl",

"KNN":
"models/knn.pkl",

"Naive Bayes":
"models/naive_bayes.pkl",

"Random Forest":
"models/random_forest.pkl"

}



selected_model = st.selectbox(
    "Select Classification Model",
    list(models.keys())
)



model = joblib.load(
    models[selected_model]
)



# -------------------------
# Upload Dataset
# -------------------------


uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)



if uploaded_file:


    df = pd.read_csv(
        uploaded_file
    )


    st.subheader(
        "Uploaded Dataset Preview"
    )

    st.dataframe(
        df.head()
    )


    if "target" not in df.columns:

        st.error(
            "CSV must contain target column"
        )

    else:


        X = df.drop(
            "target",
            axis=1
        )


        y = df["target"]



        predictions = model.predict(
            X
        )


        # Probability

        if hasattr(model,"predict_proba"):

            probabilities = (
                model.predict_proba(X)[:,1]
            )

            auc = roc_auc_score(
                y,
                probabilities
            )

        else:

            auc = 0



        metrics = {

        "Accuracy":
        accuracy_score(
            y,
            predictions
        ),

        "AUC":
        auc,

        "Precision":
        precision_score(
            y,
            predictions
        ),

        "Recall":
        recall_score(
            y,
            predictions
        ),

        "F1 Score":
        f1_score(
            y,
            predictions
        ),

        "MCC":
        matthews_corrcoef(
            y,
            predictions
        )

        }



        st.subheader(
            "Evaluation Metrics"
        )


        metric_df = pd.DataFrame(
            metrics.items(),
            columns=[
                "Metric",
                "Value"
            ]
        )


        st.table(
            metric_df
        )



        # Confusion Matrix

        st.subheader(
            "Confusion Matrix"
        )


        cm = confusion_matrix(
            y,
            predictions
        )


        fig, ax = plt.subplots()


        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )


        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )


        st.pyplot(
            fig
        )



        st.subheader(
            "Classification Report"
        )


        st.text(
            classification_report(
                y,
                predictions
            )
        )

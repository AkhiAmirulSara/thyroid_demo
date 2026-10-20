import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "response_free_clinical_rf_pipeline.joblib"
META_PATH = BASE / "streamlit_metadata.json"

st.set_page_config(page_title="Thyroid Recurrence Research Prototype", layout="centered")
st.title("Thyroid Recurrence — Research Prototype")
st.warning("Research use only. This tool is not clinically validated and must not guide diagnosis or treatment.")
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
model = joblib.load(MODEL_PATH)
st.caption("This prototype excludes post-treatment Response and derived Stage. No clinical cutoff is applied.")

with st.form("patient_inputs"):
    row = {}
    row["Age"] = st.number_input("Age (years)", min_value=0, max_value=120,
                                 value=int((meta["age_min"] + meta["age_max"]) / 2), step=1)
    for col in meta["categorical_columns"]:
        options = meta["categorical_options"].get(col, [])
        if not options:
            st.error(f"No known training values found for {col}.")
            st.stop()
        row[col] = st.selectbox(col, options=options)
    submitted = st.form_submit_button("Compute research model score")

if submitted:
    X_input = pd.DataFrame([row], columns=meta["feature_columns"])
    score = float(model.predict_proba(X_input)[0, 1])
    st.metric("Model-estimated recurrence probability", f"{score:.1%}")
    st.info("This is an unvalidated model score, not a clinical risk category. Interpret only in a research setting with clinician oversight.")
    try:
        import shap
        prep = model.named_steps["prep"]
        estimator = model.named_steps["model"]
        transformed = prep.transform(X_input)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
        feature_names = list(prep.get_feature_names_out())
        contributions = pd.DataFrame({"Encoded feature": feature_names, "SHAP contribution": values[0]})
        contributions["absolute"] = contributions["SHAP contribution"].abs()
        contributions = contributions.sort_values("absolute", ascending=False).head(5).drop(columns="absolute")
        st.subheader("Local SHAP (model behavior, not causality)")
        st.dataframe(contributions, use_container_width=True)
    except Exception as exc:
        st.info(f"Local SHAP display unavailable: {type(exc).__name__}: {exc}")
    st.write("No diagnosis, treatment, or follow-up recommendation is generated.")

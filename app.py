from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


# Make local imports work even without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parent
try:
    import insurance_project  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insurance_project.features import add_age_and_tenure_groups
from insurance_project.preprocessing import encode_categorical_features, scale_numerical_features


st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")


MODEL_FILENAME = "best_xgboost_v1.pkl"
SCALER_CANDIDATES = ["scaler_v1.pkl", "scaler.pkl"]
TRAINING_DATA_FILENAME = "Automobile_insurance_fraud_cleaned.csv"


@st.cache_resource
def load_model():
    candidates = [
        PROJECT_ROOT / "models" / MODEL_FILENAME,
        PROJECT_ROOT / "src" / "models" / MODEL_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"Model not found. Looked in: {', '.join(str(p) for p in candidates)}")


@st.cache_data
def load_training_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / TRAINING_DATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    return pd.read_csv(path)


def _default_row_from_training(training_df: pd.DataFrame) -> dict:
    df = training_df.copy()
    if "fraud_reported" in df.columns:
        df = df.drop(columns=["fraud_reported"])

    defaults: dict[str, object] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            defaults[col] = 0
        elif pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median())
        else:
            defaults[col] = series.mode().iloc[0]
    return defaults


@st.cache_resource
def load_or_fit_scaler():
    # Prefer a pre-saved scaler. If missing, fit it from training data as a fallback.
    for fname in SCALER_CANDIDATES:
        path = PROJECT_ROOT / "models" / fname
        if path.exists():
            return joblib.load(path)

    train_df = load_training_data()
    train_df = add_age_and_tenure_groups(train_df)
    train_df = encode_categorical_features(train_df)
    _, scaler = scale_numerical_features(train_df, is_train=True)

    # Best-effort persist so subsequent runs are faster.
    try:
        out_path = PROJECT_ROOT / "models" / SCALER_CANDIDATES[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, out_path)
    except Exception:
        pass

    return scaler


def user_input_features(training_df: pd.DataFrame) -> pd.DataFrame:
    defaults = _default_row_from_training(training_df)
    columns = [c for c in training_df.columns if c != "fraud_reported"]

    st.sidebar.header("📋 Enter insurance policy information")
    show_all = st.sidebar.checkbox("Show all fields", value=False)

    # Minimal inputs (others default to training medians/modes).
    incident_severity = st.sidebar.selectbox(
        "Incident severity",
        ("Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"),
        index=("Trivial Damage", "Minor Damage", "Major Damage", "Total Loss").index(
            defaults.get("incident_severity", "Minor Damage")
            if defaults.get("incident_severity", "Minor Damage") in ("Trivial Damage", "Minor Damage", "Major Damage", "Total Loss")
            else "Minor Damage"
        ),
    )
    age = st.sidebar.number_input("Age", min_value=16, max_value=120, value=int(float(defaults.get("age", 35))))
    months_as_customer = st.sidebar.number_input(
        "Months as customer", min_value=0, max_value=600, value=int(float(defaults.get("months_as_customer", 60)))
    )
    total_claim_amount = st.sidebar.number_input(
        "Total claim amount ($)", min_value=0.0, value=float(defaults.get("total_claim_amount", 50000.0)), step=100.0
    )
    incident_hour = st.sidebar.slider(
        "Incident hour of the day", min_value=0, max_value=23, value=int(float(defaults.get("incident_hour_of_the_day", 12)))
    )
    days_to_incident = st.sidebar.number_input(
        "Days to incident", min_value=0, value=int(float(defaults.get("days_to_incident", 30)))
    )

    overrides: dict[str, object] = {
        "incident_severity": incident_severity,
        "age": age,
        "months_as_customer": months_as_customer,
        "total_claim_amount": total_claim_amount,
        "incident_hour_of_the_day": incident_hour,
        "days_to_incident": days_to_incident,
    }

    if show_all:
        st.sidebar.subheader("Optional fields")

        def options_for(col: str) -> list[str]:
            if col not in training_df.columns:
                return []
            vals = training_df[col].dropna().astype(str).unique().tolist()
            vals.sort()
            return vals

        select_cols = [
            "policy_state",
            "policy_csl",
            "insured_sex",
            "insured_education_level",
            "insured_occupation",
            "insured_hobbies",
            "insured_relationship",
            "incident_type",
            "collision_type",
            "authorities_contacted",
            "incident_state",
            "incident_city",
            "property_damage",
            "police_report_available",
            "auto_make",
            "auto_model",
        ]
        num_cols = [
            "policy_deductable",
            "policy_annual_premium",
            "umbrella_limit",
            "capital-gains",
            "capital-loss",
            "number_of_vehicles_involved",
            "bodily_injuries",
            "witnesses",
            "injury_claim",
            "property_claim",
            "vehicle_claim",
            "auto_year",
        ]

        for col in select_cols:
            opts = options_for(col)
            if not opts:
                continue
            default_val = str(defaults.get(col, opts[0]))
            if default_val not in opts:
                default_val = opts[0]
            overrides[col] = st.sidebar.selectbox(col, opts, index=opts.index(default_val))

        for col in num_cols:
            if col not in training_df.columns:
                continue
            overrides[col] = st.sidebar.number_input(col, value=float(defaults.get(col, 0.0)), step=1.0)

    row = defaults
    row.update(overrides)
    return pd.DataFrame([row], columns=columns)


def preprocess_for_model(df_raw: pd.DataFrame, model, scaler) -> pd.DataFrame:
    df = add_age_and_tenure_groups(df_raw)
    df = encode_categorical_features(df)
    df = scale_numerical_features(df, is_train=False, scaler=scaler)

    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        raise ValueError("Model is missing `feature_names_in_`; cannot align features.")

    X = df.reindex(columns=expected, fill_value=0)
    return X.astype(float)


def main():
    st.title("Insurance Fraud Detection System")

    model = load_model()
    training_df = load_training_data()
    scaler = load_or_fit_scaler()

    df_input_raw = user_input_features(training_df)
    st.subheader("Input (raw schema)")
    st.dataframe(df_input_raw, use_container_width=True)

    threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)

    if st.button("Check profile"):
        try:
            X = preprocess_for_model(df_input_raw, model=model, scaler=scaler)
            proba = float(model.predict_proba(X)[0, 1])
            pred = int(proba >= threshold)
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            with st.expander("Debug"):
                st.write("Expected feature count:", len(getattr(model, "feature_names_in_", [])))
                st.write("Raw input columns:", list(df_input_raw.columns))
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Profile Status")
            if pred == 1:
                st.error("WARNING: SIGNS OF FRAUD ARE VISIBLE")
            else:
                st.success("PROFILE IS SAFE")
        with col2:
            st.subheader("Risk index")
            st.metric(label="Risk Score", value=f"{proba*100:.2f}%")
            st.progress(min(max(proba, 0.0), 1.0))

        with st.expander("Processed features (debug)"):
            st.write("X shape:", X.shape)
            st.dataframe(X.iloc[:1, :50], use_container_width=True)


if __name__ == "__main__":
    main()

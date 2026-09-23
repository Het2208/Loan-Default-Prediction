"""Streamlit inference UI for the bundled model artifact.

The model file is intentionally treated as the sole source of truth: this app
never trains, fits, or transforms data outside a fitted pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
MODEL_FILENAMES = ("balanced_random_forest.pkl",)
# A valid, encoded example record. These values are embedded so the deployed
# app does not need access to a CSV file; users can replace every value.
DEFAULT_INPUTS: dict[str, float] = {
    "Age": 56,
    "Income": 85994,
    "LoanAmount": 50587,
    "CreditScore": 520,
    "MonthsEmployed": 80,
    "NumCreditLines": 4,
    "InterestRate": 15.23,
    "LoanTerm": 36,
    "DTIRatio": 0.44,
    "Education": 0,
    "EmploymentType": 0,
    "MaritalStatus": 0,
    "HasMortgage": 1,
    "HasDependents": 1,
    "LoanPurpose": 4,
    "HasCoSigner": 1,
}


@dataclass(frozen=True)
class ModelDetails:
    """Metadata which can safely be discovered from a fitted artifact."""

    model_type: str
    is_pipeline: bool
    feature_names: tuple[str, ...] | None
    feature_count: int | None
    classes: tuple[Any, ...] | None
    supports_probability: bool
    is_fitted: bool


def find_model_file() -> Path:
    """Locate only an approved project-local model artifact."""
    candidates = [PROJECT_ROOT / "model" / name for name in MODEL_FILENAMES]
    candidates.extend(PROJECT_ROOT / name for name in MODEL_FILENAMES)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No approved model artifact was found in the project.")


@st.cache_resource(show_spinner="Loading prediction model…")
def load_model() -> Any:
    """Load the fixed local model once per Streamlit process."""
    model_path = find_model_file()
    try:
        # The supplied artifact is Joblib-serialized. joblib.load also safely
        # handles its associated array blocks, which plain pickle cannot read.
        return joblib.load(model_path)
    except Exception as exc:
        LOGGER.exception("Unable to load the configured model artifact")
        raise RuntimeError("The prediction model could not be loaded.") from exc


def inspect_model(model: Any) -> ModelDetails:
    """Read model metadata without fitting, modifying, or calling prediction."""
    feature_names_raw = getattr(model, "feature_names_in_", None)
    feature_names = (
        tuple(str(name) for name in feature_names_raw)
        if feature_names_raw is not None
        else None
    )
    feature_count = getattr(model, "n_features_in_", None)
    classes_raw = getattr(model, "classes_", None)
    classes = tuple(classes_raw.tolist() if hasattr(classes_raw, "tolist") else classes_raw) if classes_raw is not None else None
    is_pipeline = hasattr(model, "named_steps")
    # Fitted sklearn classifiers expose classes_; this avoids a predict() call
    # on an unfitted artifact and does not mutate the object.
    is_fitted = classes is not None and feature_count is not None
    return ModelDetails(
        model_type=f"{type(model).__module__}.{type(model).__name__}",
        is_pipeline=is_pipeline,
        feature_names=feature_names,
        feature_count=feature_count,
        classes=classes,
        supports_probability=callable(getattr(model, "predict_proba", None)),
        is_fitted=is_fitted,
    )


def probability_for_class(
    probabilities: Any, classes: tuple[Any, ...], target_class: int
) -> float:
    """Return a probability by its saved class label, never by column position."""
    try:
        return float(probabilities[list(classes).index(target_class)])
    except ValueError as exc:
        raise RuntimeError(f"The model does not contain class {target_class}.") from exc


def render_sidebar(details: ModelDetails) -> None:
    with st.sidebar:
        st.header("Model information")
        st.write(f"**Model type:** `{details.model_type}`")
        st.write(f"**Pipeline:** {'Yes' if details.is_pipeline else 'No'}")
        st.write(
            "**Expected features:** "
            + (str(details.feature_count) if details.feature_count is not None else "Unavailable")
        )
        st.write(f"**Class labels:** {list(details.classes) if details.classes else 'Unavailable'}")
        st.write(f"**Probability API:** {'Available' if details.supports_probability else 'Unavailable'}")
        st.divider()
        st.caption("Deployment: Streamlit-ready, using the project-local model artifact only.")


def prediction_form(model: Any, details: ModelDetails) -> None:
    if not details.is_fitted or details.feature_count is None:
        st.error(
            "This model artifact is not fitted and does not expose an input schema. "
            "Predictions cannot be made safely from this file."
        )
        st.info(
            "Please supply the fitted model or the original fitted preprocessing pipeline. "
            "The application will not invent insurance fields or fit the model."
        )
        return

    st.subheader("Applicant information")
    st.caption("A sample encoded application is prefilled below. You can update every value.")
    with st.form("claim_prediction"):
        values: dict[str, float] = {}
        if details.feature_names is None:
            st.warning("Feature names were not saved. Generic numeric fields are shown in model order.")
            field_names = [f"Feature {index}" for index in range(1, details.feature_count + 1)]
        else:
            field_names = list(details.feature_names)
        columns = st.columns(2)
        for index, name in enumerate(field_names):
            with columns[index % 2]:
                values[name] = st.number_input(
                    name,
                    value=float(DEFAULT_INPUTS.get(name, 0.0)),
                    key=f"input_{index}",
                )
        submitted = st.form_submit_button("Predict Default Risk", type="primary")

    if not submitted:
        return

    try:
        # A DataFrame preserves names and insertion order for estimators that
        # received a named training schema. The model is never fitted here.
        input_frame = pd.DataFrame([values])
        prediction = model.predict(input_frame)[0]
        st.subheader("Prediction result")
        # The requested UI maps the model's binary classes as 0 = valid and
        # 1 = fraudulent. Probabilities are located by class label, not order.
        if prediction == 1:
            st.warning("⚠️ Fraudulent Claim Detected")
        elif prediction == 0:
            st.success("✅ Claim Appears Valid")
        else:
            st.info("Prediction received from the model.")

        if details.supports_probability and details.classes is not None:
            probabilities = model.predict_proba(input_frame)[0]
            fraud_probability = probability_for_class(probabilities, details.classes, 1)
            valid_probability = probability_for_class(probabilities, details.classes, 0)
            fraud_column, valid_column = st.columns(2)
            fraud_column.metric("Fraud probability", f"{fraud_probability:.2%}")
            valid_column.metric("Valid probability", f"{valid_probability:.2%}")
    except Exception:
        LOGGER.exception("Prediction failed")
        st.error("Unable to make a prediction with the supplied applicant information.")


def main() -> None:
    st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")
    st.title("Loan Default Prediction")
    st.write("Enter the applicant information below to predict the model's default-risk class.")

    try:
        model = load_model()
        details = inspect_model(model)
    except FileNotFoundError:
        st.error("Unable to load the prediction model. Add the approved PKL artifact to the project.")
        return
    except RuntimeError:
        st.error("Unable to load the prediction model because its Python package versions are incompatible.")
        st.code("python -m pip install --upgrade --force-reinstall -r requirements.txt", language="powershell")
        st.info(
            "Then start Streamlit with the same Python environment: "
            "`python -m streamlit run app.py`. This model requires "
            "scikit-learn 1.7.2 and imbalanced-learn 0.14.0."
        )
        return

    render_sidebar(details)
    prediction_form(model, details)


if __name__ == "__main__":
    main()

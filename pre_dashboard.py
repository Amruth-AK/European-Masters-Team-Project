# pre_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing_suggestions import (
    suggest_missing_value_handling,
    suggest_duplicate_handling,
    suggest_outlier_handling,
    suggest_numerical_scaling,
    suggest_categorical_encoding,
    suggest_identifier_removal,
    suggest_correlation_based_features,
    suggest_feature_combination,
    suggest_fastica_features,
)

from preprocessing_registry import FUNC_MAP
from preprocessing_pipeline import fit_preprocessing_pipeline


def apply_suggestion(df, suggestion, analysis_results=None):
    """
    Legacy helper used in a few places (and convenient for debugging).
    For the real workflow we prefer fit_preprocessing_pipeline, but this
    remains available and uses the central FUNC_MAP.
    """
    func_name = suggestion["function_to_call"]
    kwargs = suggestion.get("kwargs", {}).copy()
    func = FUNC_MAP.get(func_name)

    if func_name == "create_features_from_correlation_analysis" and analysis_results is not None:
        kwargs.setdefault("analysis_results", analysis_results)

    if func_name == "clip_outliers_iqr" and analysis_results is not None:
        kwargs.setdefault("analysis_results", analysis_results)

    if func:
        try:
            return func(df, **kwargs)
        except Exception as e:
            st.error(f"⚠️ Error applying {func_name} to {suggestion.get('feature', 'N/A')}: {e}")
    return df


def run_preprocessing_dashboard(analysis_results: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit preprocessing dashboard with Apply/Ignore All buttons."""
    st.title("🛠 Preprocessing Suggestions")

    # Initialize session state
    if "pre_df" not in st.session_state or st.session_state.pre_df is None:
        st.session_state.pre_df = df.copy()
    if "pre_status" not in st.session_state:
        st.session_state.pre_status = None
    if "transformation_pipeline" not in st.session_state:
        st.session_state.transformation_pipeline = None
    if "fitted_pipeline" not in st.session_state:
        st.session_state.fitted_pipeline = None

    current_df = st.session_state.pre_df
    target_column = st.session_state.target_column

    # Collect all suggestions
    steps = [
        ("Identifier Removal", suggest_identifier_removal(analysis_results)),
        ("Missing Values", suggest_missing_value_handling(analysis_results, target_column)),
        ("Duplicate Rows", suggest_duplicate_handling(analysis_results)),
        ("Outliers", suggest_outlier_handling(analysis_results, target_column)),
        ("Numerical Scaling", suggest_numerical_scaling(analysis_results, target_column)),
        ("Correlation-based Features", suggest_correlation_based_features(analysis_results, target_column)),
        ("Feature Combination", suggest_feature_combination(analysis_results, target_column)),
        ("Categorical Encoding", suggest_categorical_encoding(analysis_results, target_column)),
        ("FastICA Feature Extraction", suggest_fastica_features(analysis_results, target_column)),
    ]
    valid_steps = [(name, sug) for name, sug in steps if sug]

    if not valid_steps:
        st.success("No preprocessing suggestions found — your dataset looks clean!")
        st.dataframe(current_df.head())
        # Even if no steps, mark as applied so modeling can proceed
        st.session_state.pre_status = "applied"
        st.session_state.transformation_pipeline = []
        st.session_state.fitted_pipeline = []
        return current_df

    # Show suggestions grouped by type
    for step_name, suggestions in valid_steps:
        with st.expander(f"🔹 {step_name}", expanded=True):
            for sug in suggestions:
                st.markdown(f"**Attribute:** `{sug['feature']}`")
                st.info(f"**Issue:** {sug['issue']}")
                st.warning(f"**Suggestion:** {sug['suggestion']}")
            st.divider()

    # Buttons
    if st.session_state.pre_status is None:
        col1, col2 = st.columns([1, 1])
        apply_all = col1.button("Apply All Suggestions", key="apply_all_btn")
        ignore_all = col2.button("Ignore All Suggestions", key="ignore_all_btn")

        if apply_all or ignore_all:
            # ✅ For presentation only: do NOT actually transform the data.
            # Just mark preprocessing as "done" and keep the original df.
            st.session_state.transformation_pipeline = []
            st.session_state.fitted_pipeline = []
            st.session_state.pre_df = df.copy()
            st.session_state.pre_status = "ignored"  # or "applied", doesn't matter now
            st.rerun()


    if st.session_state.pre_status == "applied":
        st.success("All preprocessing steps applied successfully!")
    elif st.session_state.pre_status == "ignored":
        st.info("All preprocessing suggestions were ignored.")

    st.subheader("📊 Final Preprocessed Dataset Preview")
    st.dataframe(st.session_state.pre_df.head())

    return st.session_state.pre_df

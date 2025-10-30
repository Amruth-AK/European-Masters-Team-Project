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
    suggest_datetime_features
)
from preprocessing_function import (
    delete_missing_columns,
    delete_missing_rows,
    impute_mean,
    impute_median,
    impute_mode,
    delete_duplicates,
    standard_scaler,
    minmax_scaler,
    clip_outliers_iqr,
    remove_outliers_iqr,
    binary_encode,
    ordinal_encode,
    one_hot_encode,
    label_encode,
    frequency_encode,
    remove_identifier_columns,
    calculate_datetime_diff,
    extract_datetime_features
)

# Map string function names to actual functions
FUNC_MAP = {
    'delete_missing_columns': delete_missing_columns,
    'delete_missing_rows': delete_missing_rows,
    'impute_mean': impute_mean,
    'impute_median': impute_median,
    'impute_mode': impute_mode,
    'delete_duplicates': delete_duplicates,
    'standard_scaler': standard_scaler,
    'minmax_scaler': minmax_scaler,
    'clip_outliers_iqr': clip_outliers_iqr,
    'remove_outliers_iqr': remove_outliers_iqr,
    'binary_encode': binary_encode,
    'ordinal_encode': ordinal_encode,
    'one_hot_encode': one_hot_encode,
    'label_encode': label_encode,
    'frequency_encode': frequency_encode,
    'remove_identifier_columns' : remove_identifier_columns,
    'calculate_datetime_diff': calculate_datetime_diff,
    'extract_datetime_features': extract_datetime_features
}


def apply_suggestion(df, suggestion):
    """Apply a preprocessing function safely."""
    func_name = suggestion['function_to_call']
    kwargs = suggestion.get('kwargs', {})
    func = FUNC_MAP.get(func_name)
    if func:
        try:
            return func(df, **kwargs)
        except Exception as e:
            st.error(f"⚠️ Error applying {func_name} to {suggestion['feature']}: {e}")
    return df

def run_preprocessing_dashboard(analysis_results: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit preprocessing dashboard with Apply/Ignore All buttons."""
    st.title("🛠 Preprocessing Suggestions")

    # Initialize session state
    if 'pre_df' not in st.session_state or st.session_state.pre_df is None:
        st.session_state.pre_df = df.copy()
    if 'pre_status' not in st.session_state:
        st.session_state.pre_status = None  
    df = st.session_state.pre_df

    # Collect all suggestions
    steps = [
        ("Missing Values", suggest_missing_value_handling(analysis_results)),
        ("Duplicate Rows", suggest_duplicate_handling(analysis_results)),
        ("Outliers", suggest_outlier_handling(analysis_results)),
        ("Numerical Scaling", suggest_numerical_scaling(analysis_results)),
        ("Categorical Encoding", suggest_categorical_encoding(analysis_results)),
        ("Datetime Feature Engineering", suggest_datetime_features(analysis_results)),
        ("Identifier Removal", suggest_identifier_removal(analysis_results))
    ]
    valid_steps = [(name, sug) for name, sug in steps if sug]

    if not valid_steps:
        st.success("No preprocessing suggestions found — your dataset looks clean!")
        st.dataframe(df.head(), width='stretch')
        return df

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
        apply_all = col1.button("Apply All Suggestions", key="apply_all_btn", width='stretch')
        ignore_all = col2.button("Ignore All Suggestions", key="ignore_all_btn", width='stretch')

        if apply_all:
            for _, suggestions in valid_steps:
                for sug in suggestions:
                    df = apply_suggestion(df, sug)
            st.session_state.pre_df = df
            st.session_state.pre_status = "applied"
            st.rerun()  

        elif ignore_all:
            st.session_state.pre_status = "ignored"
            st.rerun()

    if st.session_state.pre_status == "applied":
        st.success("All preprocessing steps applied successfully!")
    elif st.session_state.pre_status == "ignored":
        st.info("All preprocessing suggestions were ignored.")

    st.subheader("📊 Final Preprocessed Dataset Preview")
    st.dataframe(df.head(), width='stretch')

    return df



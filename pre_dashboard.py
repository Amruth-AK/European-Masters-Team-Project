import streamlit as st
import pandas as pd

# -------------------------------
# Import preprocessing functions
# -------------------------------
from Irrelevant_features import (
    suggest_irrelevant_feature_removal,
    drop_constant_features,
    drop_low_variance_features,
    drop_correlated_features,
    reduce_categorical_cardinality
)

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

# -------------------------------
# Map string function names to actual functions
# -------------------------------
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
    'remove_identifier_columns': remove_identifier_columns,
    'calculate_datetime_diff': calculate_datetime_diff,
    'extract_datetime_features': extract_datetime_features,
    'drop_constant_features': drop_constant_features,
    'drop_low_variance_features': drop_low_variance_features,
    'drop_correlated_features': drop_correlated_features,
    'reduce_categorical_cardinality': reduce_categorical_cardinality
}


# -------------------------------
# Apply a preprocessing suggestion
# -------------------------------
def apply_suggestion(df: pd.DataFrame, suggestion: dict) -> pd.DataFrame:
    """Apply a preprocessing function safely to the dataframe."""
    func_name = suggestion['function_to_call']
    kwargs = suggestion.get('kwargs', {})
    func = FUNC_MAP.get(func_name)

    if func:
        try:
            df = func(df, **kwargs)
            st.success(f"✅ Applied: {suggestion['issue']}")
        except Exception as e:
            st.error(f"⚠️ Error applying {func_name} to {suggestion['feature']}: {e}")
    else:
        st.warning(f"⚠️ Function {func_name} not mapped for {suggestion['feature']}")
    return df


# -------------------------------
# Run the Streamlit preprocessing dashboard
# -------------------------------
def run_preprocessing_dashboard(analysis_results: dict, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """Streamlit preprocessing dashboard with suggestions from multiple modules."""

    st.title("🛠 Preprocessing Suggestions")

    # Initialize session state
    if 'pre_df' not in st.session_state or st.session_state.pre_df is None:
        st.session_state.pre_df = df.copy()
    if 'pre_status' not in st.session_state:
        st.session_state.pre_status = None
    st.session_state.target_column = target_column
    df = st.session_state.pre_df

    # -------------------------------
    # Collect suggestions from all steps
    # -------------------------------
    steps = [
        ("Missing Values", suggest_missing_value_handling(analysis_results, target_column)),
        ("Duplicate Rows", suggest_duplicate_handling(analysis_results)),
        ("Outliers", suggest_outlier_handling(analysis_results, target_column)),
        ("Numerical Scaling", suggest_numerical_scaling(analysis_results, target_column)),
        ("Categorical Encoding", suggest_categorical_encoding(analysis_results, target_column)),
        ("Datetime Feature Engineering", suggest_datetime_features(analysis_results, target_column)),
        ("Identifier Removal", suggest_identifier_removal(analysis_results)),
        ("Irrelevant Feature Removal", suggest_irrelevant_feature_removal(analysis_results, target_column))
    ]
    valid_steps = [(name, sug) for name, sug in steps if sug]

    if not valid_steps:
        st.success("✅ No preprocessing suggestions found — your dataset looks clean!")
        st.dataframe(df.head(), width='stretch')
        return df

    # -------------------------------
    # Show suggestions in expanders
    # -------------------------------
    for step_name, suggestions in valid_steps:
        with st.expander(f"🔹 {step_name}", expanded=True):
            for sug in suggestions:
                st.markdown(f"**Attribute:** `{sug['feature']}`")
                st.info(f"**Issue:** {sug['issue']}")
                st.warning(f"**Suggestion:** {sug['suggestion']}")
            st.divider()

    # -------------------------------
    # Buttons for Apply All / Ignore All
    # -------------------------------
    col1, col2 = st.columns([1, 1])
    apply_all = col1.button("Apply All Suggestions", key="apply_all_btn")
    ignore_all = col2.button("Ignore All Suggestions", key="ignore_all_btn")

    if apply_all:
        for _, suggestions in valid_steps:
            for sug in suggestions:
                df = apply_suggestion(df, sug)
        st.session_state.pre_df = df
        st.session_state.pre_status = "applied"

    if ignore_all:
        st.session_state.pre_status = "ignored"

    # -------------------------------
    # Show status after action
    # -------------------------------
    if st.session_state.pre_status == "applied":
        st.success("✅ All preprocessing steps applied successfully!")
    elif st.session_state.pre_status == "ignored":
        st.info("ℹ️ All preprocessing suggestions were ignored.")

    # -------------------------------
    # Final preview
    # -------------------------------
    st.subheader("📊 Preprocessed Dataset Preview")
    st.dataframe(st.session_state.pre_df.head(), width='stretch')

    return st.session_state.pre_df

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
    suggest_fastica_features
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
    extract_datetime_features,
    create_features_from_correlation_analysis,
    combine_categorical_features,
    apply_fastica
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
    'extract_datetime_features': extract_datetime_features,
    'create_features_from_correlation_analysis': create_features_from_correlation_analysis,
    'combine_categorical_features': combine_categorical_features,
    'apply_fastica': apply_fastica
}


def apply_suggestion(df, suggestion, analysis_results=None):
    """Apply a preprocessing function safely."""
    func_name = suggestion['function_to_call']
    kwargs = suggestion.get('kwargs', {}).copy()
    func = FUNC_MAP.get(func_name)

    # Inject analysis_results for correlation-based feature creation if not already present
    if func_name == 'create_features_from_correlation_analysis' and analysis_results is not None:
        kwargs.setdefault('analysis_results', analysis_results)
    
    # --- NEW: Ensure analysis_results is available for outlier clipping on new data ---
    if func_name == 'clip_outliers_iqr' and analysis_results is not None:
        kwargs.setdefault('analysis_results', analysis_results)

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
    if 'pre_df' not in st.session_state or st.session_state.pre_df is None:
        st.session_state.pre_df = df.copy()
    if 'pre_status' not in st.session_state:
        st.session_state.pre_status = None
    
    # Use the preprocessed df if it exists, otherwise use the original
    current_df = st.session_state.pre_df

    target_column = st.session_state.target_column

    # Collect all suggestions
    steps = [
        ("Missing Values", suggest_missing_value_handling(analysis_results, target_column)),
        ("Duplicate Rows", suggest_duplicate_handling(analysis_results)),
        ("Outliers", suggest_outlier_handling(analysis_results, target_column)),
        ("Numerical Scaling", suggest_numerical_scaling(analysis_results, target_column)),
        ("Correlation-based Features", suggest_correlation_based_features(analysis_results, target_column)),
        ("Feature Combination", suggest_feature_combination(analysis_results, target_column)),
        ("Categorical Encoding", suggest_categorical_encoding(analysis_results, target_column)),
        ("Identifier Removal", suggest_identifier_removal(analysis_results)),
        ("FastICA Feature Extraction", suggest_fastica_features(analysis_results, target_column))
    ]
    valid_steps = [(name, sug) for name, sug in steps if sug]

    if not valid_steps:
        st.success("No preprocessing suggestions found — your dataset looks clean!")
        st.dataframe(current_df.head())
        # --- NEW: Even if no steps, set status so modeling can proceed ---
        st.session_state.pre_status = "applied" 
        return current_df

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

        if apply_all:
            # --- MODIFIED: Capture the pipeline before applying it ---
            pipeline_to_save = []
            for _, suggestions in valid_steps:
                for sug in suggestions:
                    pipeline_to_save.append(sug)
            
            st.session_state.transformation_pipeline = pipeline_to_save
            
            # Now, apply the steps to the current dataframe
            processed_df = current_df.copy()
            with st.spinner("Applying preprocessing steps..."):
                for step in pipeline_to_save:
                    processed_df = apply_suggestion(processed_df, step, analysis_results=analysis_results)
            
            st.session_state.pre_df = processed_df
            st.session_state.pre_status = "applied"
            st.rerun()

        elif ignore_all:
            # --- NEW: If ignored, save an empty pipeline ---
            st.session_state.transformation_pipeline = []
            st.session_state.pre_status = "ignored"
            st.rerun()

    if st.session_state.pre_status == "applied":
        st.success("All preprocessing steps applied successfully!")
    elif st.session_state.pre_status == "ignored":
        st.info("All preprocessing suggestions were ignored.")

    st.subheader("📊 Final Preprocessed Dataset Preview")
    st.dataframe(st.session_state.pre_df.head())

    return st.session_state.pre_df
# pre_dashboard.py

import streamlit as st
import pandas as pd
from preprocessing_suggestions import (
    suggest_missing_value_handling,
    suggest_duplicate_handling,
    suggest_outlier_handling,
    suggest_numerical_scaling,
    suggest_categorical_encoding
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
    frequency_encode
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
    'frequency_encode': frequency_encode
}


def apply_suggestion(df, suggestion):
    """Apply a preprocessing function safely."""
    func_name = suggestion['function_to_call']
    kwargs = suggestion.get('kwargs', {})
    func = FUNC_MAP.get(func_name)
    if func:
        return func(df, **kwargs)
    return df


def run_preprocessing_dashboard(analysis_results: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit preprocessing dashboard that applies suggestions in order
    and returns the final preprocessed DataFrame.
    """
    st.title("🛠 Preprocessing Suggestions")

    # Copy df to avoid modifying the original
    df = df.copy()

    # Initialize session_state trackers for applied/skipped suggestions
    if 'pre_steps_done' not in st.session_state:
        st.session_state.pre_steps_done = []

    # Safe preprocessing order
    steps = [
        ("Missing Values", suggest_missing_value_handling(analysis_results)),
        ("Duplicate Rows", suggest_duplicate_handling(analysis_results)),
        ("Outliers", suggest_outlier_handling(analysis_results)),
        ("Numerical Scaling", suggest_numerical_scaling(analysis_results)),
        ("Categorical Encoding", suggest_categorical_encoding(analysis_results))
    ]

    for step_name, suggestions in steps:
        st.subheader(f"🔹 {step_name}")

        if not suggestions:
            st.info(f"No suggestions for {step_name}")
            continue

        for i, sug in enumerate(suggestions):
            key_base = f"{step_name}_{i}"
            already_done = key_base in st.session_state.pre_steps_done
            if already_done:
                st.info(f"✅ Already applied or skipped: {sug['feature']}")
                continue

            st.markdown(f"**{sug['feature']}**: {sug['issue']}")
            st.info(sug['suggestion'])

            col1, col2 = st.columns(2)
            if col1.button(f"Apply ({sug['feature']})", key=f"apply_{key_base}"):
                df = apply_suggestion(df, sug)
                st.session_state.pre_steps_done.append(key_base)
                st.success(f"Applied suggestion for {sug['feature']}")
            if col2.button(f"Skip ({sug['feature']})", key=f"skip_{key_base}"):
                st.session_state.pre_steps_done.append(key_base)
                st.info(f"Skipped suggestion for {sug['feature']}")

    st.subheader("✅ Final Preprocessed Dataset Preview")
    st.dataframe(df.head())

    # Option to reset preprocessing
    if st.button("🔄 Reset Preprocessing"):
        st.session_state.pre_steps_done = []
        st.success("Preprocessing reset. You can start applying suggestions again.")

    # Return the final dataset
    return df

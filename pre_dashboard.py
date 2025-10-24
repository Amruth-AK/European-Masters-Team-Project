# pre_dashboard.py (only show relevant suggestion sections)

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
    Streamlit preprocessing dashboard that only shows sections with suggestions.
    """

    st.title("🛠 Preprocessing Suggestions")
    st.markdown(
        "Review and apply only the relevant preprocessing suggestions detected in your dataset."
    )

    # Initialize session_state for dataset and applied steps
    if 'pre_df' not in st.session_state or st.session_state.pre_df is None:
        st.session_state.pre_df = df.copy()
    if 'pre_steps_done' not in st.session_state:
        st.session_state.pre_steps_done = []

    df = st.session_state.pre_df  # work with session-state dataset

    # Preprocessing steps (with suggestions)
    steps = [
        ("1️⃣ Missing Values", suggest_missing_value_handling(analysis_results)),
        ("2️⃣ Duplicate Rows", suggest_duplicate_handling(analysis_results)),
        ("3️⃣ Outliers", suggest_outlier_handling(analysis_results)),
        ("4️⃣ Numerical Scaling", suggest_numerical_scaling(analysis_results)),
        ("5️⃣ Categorical Encoding", suggest_categorical_encoding(analysis_results))
    ]

    # Filter out steps with no suggestions
    steps_with_suggestions = [(name, suggs) for name, suggs in steps if suggs]

    if not steps_with_suggestions:
        st.success("🎉 No preprocessing issues detected! Your dataset looks clean and ready.")
    else:
        for step_name, suggestions in steps_with_suggestions:
            with st.expander(step_name, expanded=True):
                for i, sug in enumerate(suggestions):
                    key_base = f"{step_name}_{i}"
                    if key_base in st.session_state.pre_steps_done:
                        st.success(f"✅ Already applied or skipped: {sug['feature']}")
                        continue

                    st.markdown(f"**📝 Feature:** `{sug['feature']}`")
                    st.info(f"**Issue:** {sug['issue']}")
                    st.warning(f"💡 Suggestion: {sug['suggestion']}")

                    col1, col2 = st.columns([1, 1])
                    if col1.button(f"✅ Apply", key=f"apply_{key_base}"):
                        df = apply_suggestion(df, sug)
                        st.session_state.pre_df = df
                        st.session_state.pre_steps_done.append(key_base)
                        st.success(f"Applied suggestion for {sug['feature']}")
                        st.rerun()

                    if col2.button(f"⏭ Skip", key=f"skip_{key_base}"):
                        st.session_state.pre_steps_done.append(key_base)
                        st.info(f"Skipped suggestion for {sug['feature']}")
                        st.rerun()

    # Final dataset preview
    st.subheader("📊 Final Preprocessed Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    return df

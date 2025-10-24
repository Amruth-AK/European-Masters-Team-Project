<<<<<<< Updated upstream
# pre_dashboard.py

=======
>>>>>>> Stashed changes
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
        try:
            return func(df, **kwargs)
        except Exception as e:
            st.error(f"⚠️ Error applying {func_name} to {suggestion['feature']}: {e}")
    return df


def run_preprocessing_dashboard(analysis_results: dict, df: pd.DataFrame) -> pd.DataFrame:
<<<<<<< Updated upstream
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
=======
    """Streamlit preprocessing dashboard with Apply/Ignore All buttons."""
    st.title("🛠 Preprocessing Suggestions")

    # Initialize session state
    if 'pre_df' not in st.session_state or st.session_state.pre_df is None:
        st.session_state.pre_df = df.copy()
    if 'pre_status' not in st.session_state:
        st.session_state.pre_status = None  
    df = st.session_state.pre_df

    # Collect all suggestions
>>>>>>> Stashed changes
    steps = [
        ("Missing Values", suggest_missing_value_handling(analysis_results)),
        ("Duplicate Rows", suggest_duplicate_handling(analysis_results)),
        ("Outliers", suggest_outlier_handling(analysis_results)),
        ("Numerical Scaling", suggest_numerical_scaling(analysis_results)),
        ("Categorical Encoding", suggest_categorical_encoding(analysis_results))
    ]
    valid_steps = [(name, sug) for name, sug in steps if sug]

<<<<<<< Updated upstream
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
=======
    if not valid_steps:
        st.success("No preprocessing suggestions found — your dataset looks clean!")
        st.dataframe(df.head(), width='stretch')
        return df

    st.subheader("📋 Suggestions Summary")
    for step_name, suggestions in valid_steps:
        with st.expander(f"🔹 {step_name}", expanded=True):
            for sug in suggestions:
                st.markdown(f"**Attribute:** `{sug['feature']}`")
                st.info(f"**Issue:** {sug['issue']}")
                st.warning(f"**Suggestion:** {sug['suggestion']}")
            st.divider()

    # --- Buttons ---
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
>>>>>>> Stashed changes

    # Option to reset preprocessing
    if st.button("🔄 Reset Preprocessing"):
        st.session_state.pre_steps_done = []
        st.success("Preprocessing reset. You can start applying suggestions again.")

    # Return the final dataset
    return df

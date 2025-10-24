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


def apply_suggestion(df, suggestion):
    """
    Calls the appropriate function from preprocessing_functions.py
    """
    func_name = suggestion['function_to_call']
    kwargs = suggestion.get('kwargs', {})

    # Map string function names to actual functions
    func_map = {
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

    func = func_map.get(func_name)
    if func:
        df = func(df, **kwargs)
    return df


def create_preprocessing_dashboard(analysis_results: dict):
    """
    Streamlit page to show preprocessing suggestions and allow auto-application.
    """
    st.title("🛠 Preprocessing Suggestions")

    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset and run analysis on the Home page first.")
        return

    df = st.session_state.df

    # --- 1. Missing Values ---
    st.subheader("1️⃣ Missing Values")
    missing_suggestions = suggest_missing_value_handling(analysis_results)
    if missing_suggestions:
        for i, sug in enumerate(missing_suggestions):
            st.markdown(f"**{sug['feature']}**: {sug['issue']}")
            st.info(sug['suggestion'])
            if st.button(f"✅ Apply Suggestion ({sug['feature']})", key=f"missing_{i}"):
                df = apply_suggestion(df, sug)
                st.success(f"Applied suggestion for {sug['feature']}")
    else:
        st.success("No missing value suggestions!")

    # --- 2. Duplicates ---
    st.subheader("2️⃣ Duplicate Rows")
    dup_suggestions = suggest_duplicate_handling(analysis_results)
    if dup_suggestions:
        for i, sug in enumerate(dup_suggestions):
            st.markdown(f"**{sug['feature']}**: {sug['issue']}")
            st.info(sug['suggestion'])
            if st.button(f"✅ Apply Suggestion ({sug['feature']})", key=f"dup_{i}"):
                df = apply_suggestion(df, sug)
                st.success(f"Applied suggestion for {sug['feature']}")
    else:
        st.success("No duplicate suggestions!")

    # --- 3. Outliers ---
    st.subheader("3️⃣ Outliers")
    outlier_suggestions = suggest_outlier_handling(analysis_results)
    if outlier_suggestions:
        for i, sug in enumerate(outlier_suggestions):
            st.markdown(f"**{sug['feature']}**: {sug['issue']}")
            st.info(sug['suggestion'])
            if st.button(f"✅ Apply Suggestion ({sug['feature']})", key=f"outlier_{i}"):
                df = apply_suggestion(df, sug)
                st.success(f"Applied suggestion for {sug['feature']}")
    else:
        st.success("No outlier suggestions!")

    # --- 4. Numerical Scaling ---
    st.subheader("4️⃣ Numerical Scaling")
    scale_suggestions = suggest_numerical_scaling(analysis_results)
    if scale_suggestions:
        for i, sug in enumerate(scale_suggestions):
            st.markdown(f"**{sug['feature']}**: {sug['issue']}")
            st.info(sug['suggestion'])
            if st.button(f"✅ Apply Suggestion ({sug['feature']})", key=f"scale_{i}"):
                df = apply_suggestion(df, sug)
                st.success(f"Applied suggestion for {sug['feature']}")
    else:
        st.success("No scaling suggestions!")

    # --- 5. Categorical Encoding ---
    st.subheader("5️⃣ Categorical Encoding")
    cat_suggestions = suggest_categorical_encoding(analysis_results)
    if cat_suggestions:
        for i, sug in enumerate(cat_suggestions):
            st.markdown(f"**{sug['feature']}**: {sug['issue']}")
            st.info(sug['suggestion'])
            if st.button(f"✅ Apply Suggestion ({sug['feature']})", key=f"cat_{i}"):
                df = apply_suggestion(df, sug)
                st.success(f"Applied suggestion for {sug['feature']}")
    else:
        st.success("No categorical encoding suggestions!")

    # Update session state with the new DataFrame
    st.session_state.df = df
    st.success("✅ Preprocessing suggestions applied successfully. You can continue analysis with the updated dataset.")

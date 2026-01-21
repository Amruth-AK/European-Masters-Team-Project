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
    suggest_datetime_features,
)

from preprocessing_registry import FUNC_MAP
from preprocessing_pipeline import fit_preprocessing_pipeline

# --- CALLBACK FUNCTION FOR "SELECT ALL" ---
def toggle_group(step_name, count):
    """
    Callback to select/deselect all checkboxes in a group.
    """
    master_key = f"select_all_{step_name}"
    master_value = st.session_state[master_key]
    
    for i in range(count):
        child_key = f"select_{step_name}_{i}"
        # Update the child checkbox state to match the master
        st.session_state[child_key] = master_value


def run_preprocessing_dashboard() -> pd.DataFrame:
    """
    Streamlit preprocessing dashboard using session state.
    """
    st.title("🛠 Preprocessing Suggestions")

    # --- Read from session state ---
    df = st.session_state.df
    analysis_results = st.session_state.analysis_results
    target_column = st.session_state.target_column

    if df is None or analysis_results is None:
        st.warning("⚠️ Please upload a dataset and run analysis first.")
        return

    # --- Initialize session state variables ---
    if "pre_df" not in st.session_state or st.session_state.pre_df is None:
        st.session_state.pre_df = df.copy()
    if "pre_status" not in st.session_state:
        st.session_state.pre_status = None
    if "transformation_pipeline" not in st.session_state:
        st.session_state.transformation_pipeline = None
    if "fitted_pipeline" not in st.session_state:
        st.session_state.fitted_pipeline = None

    current_df = st.session_state.pre_df

    # --- Collect preprocessing suggestions ---
    steps = [
        ("Identifier Removal", suggest_identifier_removal(analysis_results)),
        ("Missing Values", suggest_missing_value_handling(analysis_results, target_column)),
        ("Duplicate Rows", suggest_duplicate_handling(analysis_results)),
        ("Outliers", suggest_outlier_handling(analysis_results, target_column)),
        ("Datetime Features", suggest_datetime_features(analysis_results, target_column)),
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
        st.session_state.pre_status = "applied"
        return current_df

    # --- Display suggestions ---
    selected_suggestions_list = []

    for step_name, suggestions in valid_steps:
        # Use columns to put the expander and a count badge or similar if desired
        with st.expander(f"🔹 {step_name} ({len(suggestions)})", expanded=False):
            
            # --- "SELECT ALL" CHECKBOX ---
            # This uses the callback defined above
            st.checkbox(
                "Select All in this category",
                key=f"select_all_{step_name}",
                value=True,
                on_change=toggle_group,
                args=(step_name, len(suggestions))
            )
            st.markdown("---") # Visual separator

            for i, sug in enumerate(suggestions):
                key = f"select_{step_name}_{i}"
                
                # IMPORTANT: Initialize key in session state if not present
                # This ensures the callback works even if the widget hasn't rendered yet
                if key not in st.session_state:
                    st.session_state[key] = True

                col_chk, col_det = st.columns([0.05, 0.95])
                with col_chk:
                    is_selected = st.checkbox(
                        "Select",
                        key=key,
                        # No explicit 'value=' here because key in session_state takes precedence
                        label_visibility="collapsed"
                    )
                with col_det:
                    st.markdown(f"**{sug['suggestion']}**")
                    st.caption(f"Feature: `{sug['feature']}` | Issue: {sug['issue']}")
                
                if is_selected:
                    selected_suggestions_list.append(sug)

    st.divider()

    # --- Buttons to apply/ignore ---
    if st.session_state.pre_status is None:
        col1, col2 = st.columns([1, 1])
        n_selected = len(selected_suggestions_list)
        
        apply_selected = col1.button(
            f"Apply Selected Suggestions ({n_selected})",
            key="apply_selected_btn",
            type="primary",
            disabled=n_selected == 0
        )
        ignore_all = col2.button("Ignore All Suggestions", key="ignore_all_btn")

        if apply_selected:
            pipeline_to_save = []
            for sug in selected_suggestions_list:
                if sug.get("function_to_call") is not None:
                    # Pass dynamic params like FastICA ratio if they exist
                    if sug["function_to_call"] == "apply_fastica" and "fastica_replace_ratio" in st.session_state:
                        sug["kwargs"]["replace_ratio"] = st.session_state.fastica_replace_ratio
                    pipeline_to_save.append(sug)

            st.session_state.transformation_pipeline = pipeline_to_save

            with st.spinner(f"Applying {len(pipeline_to_save)} preprocessing steps..."):
                processed_df, fitted_steps = fit_preprocessing_pipeline(
                    current_df,
                    pipeline_to_save,
                    analysis_results=analysis_results,
                )

            st.session_state.pre_df = processed_df
            st.session_state.fitted_pipeline = fitted_steps
            st.session_state.pre_status = "applied"
            st.rerun()

        elif ignore_all:
            st.session_state.transformation_pipeline = []
            st.session_state.fitted_pipeline = []
            st.session_state.pre_status = "ignored"
            st.rerun()

    # --- Status messages ---
    if st.session_state.pre_status == "applied":
        st.success("All preprocessing steps applied successfully!")
    elif st.session_state.pre_status == "ignored":
        st.info("All preprocessing suggestions were ignored.")

    st.subheader("📊 Final Preprocessed Dataset Preview")
    st.dataframe(st.session_state.pre_df.head())

    return st.session_state.pre_df
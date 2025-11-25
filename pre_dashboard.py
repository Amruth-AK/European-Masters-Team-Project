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
    suggest_datetime_features,
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
        # Even if no steps, mark as applied so modeling can proceed
        st.session_state.pre_status = "applied"
        st.session_state.transformation_pipeline = []
        st.session_state.fitted_pipeline = []
        return current_df

    # Store selected suggestions
    selected_suggestions_list = []

    # Show suggestions grouped by type
    for step_name, suggestions in valid_steps:
        with st.expander(f"🔹 {step_name}", expanded=True):
            for i, sug in enumerate(suggestions):
                # Create a unique key for the checkbox
                key = f"select_{step_name}_{i}"
                
                # Checkbox for selection (Default: Checked)
                col_chk, col_det = st.columns([0.05, 0.95])
                with col_chk:
                    is_selected = st.checkbox(
                        "Select", 
                        value=True, 
                        key=key, 
                        label_visibility="collapsed"
                    )
                
                with col_det:
                    st.markdown(f"**{sug['suggestion']}**")
                    st.caption(f"Feature: `{sug['feature']}` | Issue: {sug['issue']}")

                if is_selected:
                    selected_suggestions_list.append(sug)

            st.divider()

        # Special UI for FastICA Tuning
        if step_name == "FastICA Feature Extraction":
            st.markdown("#### 🧠 Auto-Tuning (Optional)")
            st.caption("Optimize the `replace_ratio` using Optuna to find the best balance between original features and ICA components.")
            
            col_tune, col_res = st.columns([1, 2])
            with col_tune:
                if st.button("Run FastICA Tuning", key="tune_fastica_btn"):
                    with st.spinner("Tuning FastICA replace_ratio... (this may take a minute)"):
                        target_col = st.session_state.target_column
                        if target_col:
                            # Simple heuristic for model selection
                            y = current_df[target_col]
                            is_numeric = pd.api.types.is_numeric_dtype(y)
                            if is_numeric and y.nunique() > 20:
                                problem_type = "regression"
                                model = LinearRegression()
                            else:
                                problem_type = "classification"
                                model = LogisticRegression()
                                
                            try:
                                tuning_results = tune_fastica_replace_ratio(
                                    df=current_df,
                                    target_column=target_col,
                                    model=model,
                                    problem_type=problem_type,
                                    n_trials=10
                                )
                                best_ratio = tuning_results['best_replace_ratio']
                                st.session_state.fastica_replace_ratio = best_ratio
                                st.success(f"Best replace_ratio: {best_ratio:.2f}")
                            except Exception as e:
                                st.error(f"Tuning failed: {e}")
                        else:
                            st.error("Target column not defined.")

            with col_res:
                if "fastica_replace_ratio" in st.session_state:
                    st.info(f"✅ Using tuned ratio: **{st.session_state.fastica_replace_ratio:.2f}**")

    # Buttons
    if st.session_state.pre_status is None:
        col1, col2 = st.columns([1, 1])
        
        # Count selected
        n_selected = len(selected_suggestions_list)
        
        apply_selected = col1.button(
            f"Apply Selected Suggestions ({n_selected})", 
            key="apply_selected_btn",
            type="primary",
            disabled=n_selected == 0
        )
        ignore_all = col2.button("Ignore All Suggestions", key="ignore_all_btn")

        if apply_selected:
            # Build pipeline from SELECTED suggestions
            pipeline_to_save = []
            for sug in selected_suggestions_list:
                if sug.get("function_to_call") is not None:
                    # Override replace_ratio if tuned
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

    if st.session_state.pre_status == "applied":
        st.success("All preprocessing steps applied successfully!")
    elif st.session_state.pre_status == "ignored":
        st.info("All preprocessing suggestions were ignored.")

    st.subheader("📊 Final Preprocessed Dataset Preview")
    st.dataframe(st.session_state.pre_df.head())

    return st.session_state.pre_df

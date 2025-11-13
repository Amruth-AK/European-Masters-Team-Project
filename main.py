# main.py

import sys
print(sys.executable)

import streamlit as st
import pandas as pd
from typing import List

from analyze import DataAnalyzer, auto_detect_id_columns
from dashboard import create_dashboard
from pre_dashboard import run_preprocessing_dashboard
from model_suggestion import run_model_suggestions
from feature_selection import select_features_by_importance
from optuna_tuning import tune_model_with_optuna
from prediction_page import display_prediction_page


# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Data Analysis Tool",
    page_icon="🚀",
    layout="wide",
)

# --- Session State Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "id_columns_to_ignore" not in st.session_state:
    st.session_state.id_columns_to_ignore = None
if "pre_status" not in st.session_state:
    st.session_state.pre_status = None
if "pre_df" not in st.session_state:
    st.session_state.pre_df = None
if "modeling_results" not in st.session_state:
    st.session_state.modeling_results = None

# Prediction / pipeline-related state
if "data_schema" not in st.session_state:
    st.session_state.data_schema = None  # dtypes after manual adjustment
if "transformation_pipeline" not in st.session_state:
    st.session_state.transformation_pipeline = None  # logical list of suggestions
if "fitted_pipeline" not in st.session_state:
    st.session_state.fitted_pipeline = None  # fitted steps (for test/prediction)
if "prediction_results_df" not in st.session_state:
    st.session_state.prediction_results_df = None
if "test_metric_name" not in st.session_state:
    st.session_state.test_metric_name = None
if "test_metric_value" not in st.session_state:
    st.session_state.test_metric_value = None


# --- Validation Helper Function ---
def get_plausible_conversion_types(series: pd.Series) -> List[str]:
    """
    Determines which data type conversions are plausible for a given column.
    This prevents users from making obviously incorrect type changes in the UI.
    """
    plausible_types: List[str] = []
    current_dtype = str(series.dtype)

    # All columns can always be treated as Categorical
    plausible_types.append("Categorical")

    # --- Check for Numerical Plausibility ---
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() / max(series.notna().sum(), 1) > 0.75:
        plausible_types.append("Numerical")

    # --- Check for Datetime Plausibility ---
    if pd.api.types.is_numeric_dtype(series.dtype):
        # Heuristic: big numbers may be timestamps
        if series.median() > 1_000_000_000:
            plausible_types.append("Datetime")
    else:
        try:
            datetime_series = pd.to_datetime(series, errors="coerce")
            if datetime_series.notna().sum() / max(series.notna().sum(), 1) > 0.75:
                plausible_types.append("Datetime")
        except (ValueError, TypeError):
            pass

    return list(set(plausible_types))


# --- Home Page ---
def display_home_page():
    """Defines the content of the Home page."""
    st.title("🚀 Interactive Data Analysis Tool")
    st.write("Upload your dataset to begin a comprehensive analysis and model evaluation.")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"], key="file_uploader")

    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            # Reset all dependent states
            keys_to_reset = [
                "analysis_results",
                "pre_status",
                "pre_df",
                "id_columns_to_ignore",
                "modeling_results",
                "target_column",
                "data_schema",
                "transformation_pipeline",
                "fitted_pipeline",
                "prediction_results_df",
                "test_metric_name",
                "test_metric_value",
            ]
            for key in keys_to_reset:
                st.session_state[key] = None
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.session_state.df = None

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(), use_container_width=True)

        with st.expander("🔧 Manually Adjust Column Data Types", expanded=False):
            st.info(
                "The system automatically limits conversion options to prevent errors "
                "(e.g., you cannot convert a text column directly to numbers if most values are non-numeric)."
            )

            dtype_options = {
                "Numerical": "float64",
                "Categorical": "category",
                "Datetime": "datetime64[ns]",
            }

            modified_df = st.session_state.df.copy()
            cols_per_row = 3
            ui_cols = st.columns(cols_per_row)

            for i, col_name in enumerate(modified_df.columns):
                with ui_cols[i % cols_per_row]:
                    inferred_dtype_str = str(modified_df[col_name].dtype)

                    if "int" in inferred_dtype_str or "float" in inferred_dtype_str:
                        current_type_str = "Numerical"
                    elif "datetime" in inferred_dtype_str:
                        current_type_str = "Datetime"
                    else:
                        current_type_str = "Categorical"

                    plausible_options = get_plausible_conversion_types(modified_df[col_name])
                    if current_type_str not in plausible_options:
                        plausible_options.insert(0, current_type_str)

                    try:
                        default_index = plausible_options.index(current_type_str)
                    except ValueError:
                        default_index = 0

                    selected_type_str = st.selectbox(
                        label=f"**{col_name}** (Inferred: *{current_type_str}*)",
                        options=plausible_options,
                        index=default_index,
                        key=f"dtype_override_{col_name}",
                    )

                    new_dtype = dtype_options[selected_type_str]
                    if new_dtype != inferred_dtype_str:
                        try:
                            if new_dtype == "datetime64[ns]":
                                modified_df[col_name] = pd.to_datetime(modified_df[col_name])
                            else:
                                modified_df[col_name] = modified_df[col_name].astype(new_dtype)
                        except Exception as e:
                            st.error(
                                f"Failed to convert '{col_name}' to {selected_type_str}. "
                                f"Reverting. Error: {e}"
                            )
                            modified_df[col_name] = st.session_state.df[col_name]

            st.session_state.df = modified_df
            # Save the final data schema
            st.session_state.data_schema = st.session_state.df.dtypes.to_dict()

        # --- Identifier Column Selection ---
        st.subheader("⚙️ Configure Identifier Columns")
        st.info(
            "The system automatically detects columns that look like identifiers (e.g., 'user_id'). "
            "These columns will be ignored during statistical analysis and duplicate row checks. "
            "You can review and adjust the selection below."
        )
        if st.session_state.id_columns_to_ignore is None:
            detected_ids = auto_detect_id_columns(st.session_state.df, st.session_state.target_column)
            st.session_state.id_columns_to_ignore = detected_ids
        selected_id_cols = st.multiselect(
            label="Select the column(s) to treat as identifiers:",
            options=st.session_state.df.columns.tolist(),
            default=st.session_state.id_columns_to_ignore,
            help="These columns will be excluded from numerical analysis, correlations, and duplicate row detection.",
        )
        st.session_state.id_columns_to_ignore = selected_id_cols

        # --- Target Column Selection ---
        st.session_state.target_column = st.selectbox(
            "🎯 Select the target column for analysis (e.g., prediction target)",
            options=[col for col in st.session_state.df.columns if col not in st.session_state.id_columns_to_ignore],
        )

        col1, col2 = st.columns([1, 1])

        if col1.button("📊 Run Data Analysis", use_container_width=True):
            with st.spinner("Running comprehensive analysis... This may take a moment."):
                analyzer_instance = DataAnalyzer(
                    df=st.session_state.df,
                    target_column=st.session_state.target_column,
                    id_columns_to_ignore=st.session_state.id_columns_to_ignore,
                )
                st.session_state.analysis_results = analyzer_instance.run_full_analysis()
            st.success("✅ Analysis complete! Use the sidebar to explore results.")

        if col2.button("🗑️ Clear Dataset", use_container_width=True):
            keys_to_clear = [
                "df",
                "analysis_results",
                "target_column",
                "pre_status",
                "pre_df",
                "modeling_results",
                "id_columns_to_ignore",
                "data_schema",
                "transformation_pipeline",
                "fitted_pipeline",
                "prediction_results_df",
                "test_metric_name",
                "test_metric_value",
            ]
            for key in keys_to_clear:
                st.session_state[key] = None
            st.rerun()


# --- Sidebar Navigation ---
pages = [
    "Home",
    "General Info",
    "Missing Values",
    "Descriptive Statistics",
    "Distributions",
    "Categorical Info",
    "Correlations",
    "Outlier Info",
    "Duplicate Analysis",
    "Preprocessing Suggestions",
    "Model Suggestions",
    "Make Predictions",
]

st.sidebar.title("📚 Navigation")
selected_page = st.sidebar.radio("Go to", pages)

# --- Page Routing Logic ---
if selected_page == "Home":
    display_home_page()

elif selected_page == "Preprocessing Suggestions":
    if st.session_state.analysis_results and st.session_state.df is not None:
        _ = run_preprocessing_dashboard(
            st.session_state.analysis_results,
            st.session_state.df,
        )
    else:
        st.warning("⚠️ Please upload a dataset and run the analysis first.")

elif selected_page == "Make Predictions":
    display_prediction_page()

elif selected_page == "Model Suggestions":
    if st.session_state.df is not None and st.session_state.target_column:
        if st.session_state.pre_status not in ["applied", "ignored"]:
            st.warning(
                "⚠️ Please finish preprocessing first by applying or ignoring the "
                "suggestions on the Preprocessing page."
            )
        else:
            st.title("🤖 Model Suggestions")
            data_for_modeling = (
                st.session_state.pre_df
                if st.session_state.pre_df is not None
                else st.session_state.df
            )
            results = st.session_state.get("modeling_results")
            if results is None:
                st.info(
                    "This step will:\n"
                    "- Identify the best model family using an internal model search\n"
                    "- Use feature importance to remove irrelevant features\n"
                    "- Run Optuna to find the best hyperparameters for a pure sklearn model."
                )
                if st.button("🚀 Run model search and hyperparameter tuning", use_container_width=True):
                    target_column = st.session_state.target_column
                    st.info("🔍 Identifying the best model.")
                    with st.spinner("Identifying the best model."):
                        ag_results = run_model_suggestions(
                            data_for_modeling,
                            target_column=target_column,
                        )
                    st.info("🧬 Selecting most relevant features.")
                    with st.spinner("Selecting most relevant features."):
                        reduced_df, selected_features = select_features_by_importance(
                            df=data_for_modeling,
                            target_column=target_column,
                            feature_importance=ag_results.get("feature_importance"),
                            importance_threshold=0.0,
                        )
                    st.info("🎯 Finding the best hyperparameters.")
                    with st.spinner("Finding the best hyperparameters."):
                        tuning_results = tune_model_with_optuna(
                            df=reduced_df,
                            target_column=target_column,
                            model_family=ag_results["best_model_family"],
                            problem_type=ag_results["problem_type"],
                            eval_metric=ag_results["eval_metric"],
                            n_trials=30,
                            time_limit=120,
                        )
                    st.session_state.modeling_results = {
                        "problem_type": ag_results["problem_type"],
                        "eval_metric": ag_results["eval_metric"],
                        "auto_best_model_name": ag_results["best_model_name"],
                        "auto_best_model_family": ag_results["best_model_family"],
                        "selected_features": selected_features,
                        "tuned_model_family": ag_results["best_model_family"],
                        "tuned_model_class": tuning_results["best_model_class"],
                        "tuned_params": tuning_results["best_params"],
                        "final_model": tuning_results["best_model"],
                        "eval_score": tuning_results["best_eval_score"],
                    }
                    st.success(
                        "✅ Best model identified and hyperparameters tuned. "
                        "You can now inspect the results below."
                    )
                    results = st.session_state.modeling_results

            if results is not None:
                st.subheader("🏁 Final Model")
                best_model_label = (
                    results.get("auto_best_model_name")
                    or results.get("auto_best_model_family")
                    or "Unknown model"
                )
                st.write(f"**Best Model:** `{best_model_label}`")
                st.write("**Optimal Hyperparameters:**")
                st.json(results["tuned_params"])
                eval_metric = (results.get("eval_metric") or "").lower()
                eval_score = results.get("eval_score", None)
                if eval_score is not None:
                    if eval_metric == "roc_auc":
                        st.write(f"**Validation ROC AUC:** `{eval_score:.4f}`")
                    elif eval_metric == "root_mean_squared_error":
                        st.write(f"**Validation RMSE:** `{eval_score:.4f}`")
                    elif eval_metric == "log_loss":
                        st.write(f"**Validation log_loss:** `{eval_score:.4f}`")
                    else:
                        st.write(f"**Validation score:** `{eval_score:.4f}`")
    else:
        st.warning("⚠️ Please upload a dataset and select a target column on the Home page first.")

else:
    if st.session_state.analysis_results:
        create_dashboard(st.session_state.analysis_results, selected_page)
    else:
        st.warning("⚠️ Please upload a dataset and run analysis on the Home page first.")

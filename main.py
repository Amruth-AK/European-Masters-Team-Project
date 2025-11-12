import sys
print(sys.executable)
import streamlit as st
import pandas as pd
from analyze import DataAnalyzer, auto_detect_id_columns
from dashboard import create_dashboard
from pre_dashboard import run_preprocessing_dashboard
from model_suggestion import run_model_suggestions
from feature_selection import select_features_by_importance
from optuna_tuning import tune_model_with_optuna
from typing import List

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Data Analysis Tool",
    page_icon="🚀",
    layout="wide"
)

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'id_columns_to_ignore' not in st.session_state: # FIXED: Added initialization
    st.session_state.id_columns_to_ignore = None
if 'pre_status' not in st.session_state:
    st.session_state.pre_status = None
if 'pre_df' not in st.session_state:
    st.session_state.pre_df = None
if 'modeling_results' not in st.session_state:
    st.session_state.modeling_results = None


# --- NEW: Validation Helper Function ---
def get_plausible_conversion_types(series: pd.Series) -> List[str]:
    """
    Determines which data type conversions are plausible for a given column.
    This prevents users from making obviously incorrect type changes in the UI.
    """
    plausible_types = []
    current_dtype = str(series.dtype)
    
    # All columns can always be treated as Categorical
    plausible_types.append("Categorical")

    # --- Check for Numerical Plausibility ---
    # Attempt to convert to numeric, coercing errors.
    numeric_series = pd.to_numeric(series, errors='coerce')
    # If less than 25% of the values are lost, consider it a plausible conversion.
    if numeric_series.notna().sum() / series.notna().sum() > 0.75:
        plausible_types.append("Numerical")

    # --- Check for Datetime Plausibility ---
    # Avoid converting pure numbers unless they look like valid timestamps (large numbers)
    if pd.api.types.is_numeric_dtype(series.dtype):
        # Heuristic: if the median value is a huge number, it might be a Unix timestamp.
        # Otherwise, small numbers like 1, 2, 3 are unlikely to be dates.
        if series.median() > 1_000_000_000: # A reasonable cutoff for recent timestamps
             plausible_types.append("Datetime")
    else:
        # For object types, try converting to datetime
        try:
            datetime_series = pd.to_datetime(series, errors='coerce')
            if datetime_series.notna().sum() / series.notna().sum() > 0.75:
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
            keys_to_reset = ['analysis_results', 'pre_status', 'pre_df', 'id_columns_to_ignore', 'modeling_results', 'target_column']
            for key in keys_to_reset:
                st.session_state[key] = None
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.session_state.df = None

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        with st.expander("🔧 Manually Adjust Column Data Types", expanded=False):
            st.info("The system automatically limits conversion options to prevent errors (e.g., you cannot convert a text column to numbers).")

            # --- UPDATED: Simplified data type options ---
            dtype_options = {
                "Numerical": "float64",
                "Categorical": "category",
                "Datetime": "datetime64[ns]",
            }
            full_options_list = list(dtype_options.keys())
            modified_df = st.session_state.df.copy()
            cols_per_row = 3
            ui_cols = st.columns(cols_per_row)

            for i, col_name in enumerate(modified_df.columns):
                with ui_cols[i % cols_per_row]:
                    inferred_dtype_str = str(modified_df[col_name].dtype)
                    
                    # Determine current user-friendly type
                    if 'int' in inferred_dtype_str or 'float' in inferred_dtype_str:
                        current_type_str = "Numerical"
                    elif 'datetime' in inferred_dtype_str:
                        current_type_str = "Datetime"
                    else:
                        current_type_str = "Categorical"

                    # --- UPDATED: Get only plausible options for this column ---
                    plausible_options = get_plausible_conversion_types(modified_df[col_name])
                    
                    # Ensure the current type is always an option
                    if current_type_str not in plausible_options:
                        plausible_options.insert(0, current_type_str)

                    try:
                        default_index = plausible_options.index(current_type_str)
                    except ValueError:
                        default_index = 0
                    
                    # Create the selectbox with the filtered list of options
                    selected_type_str = st.selectbox(
                        label=f"**{col_name}** (Inferred: *{current_type_str}*)",
                        options=plausible_options,
                        index=default_index,
                        key=f"dtype_override_{col_name}"
                    )

                    new_dtype = dtype_options[selected_type_str]
                    if new_dtype != inferred_dtype_str:
                        try:
                            if new_dtype == 'datetime64[ns]':
                                modified_df[col_name] = pd.to_datetime(modified_df[col_name])
                            else:
                                modified_df[col_name] = modified_df[col_name].astype(new_dtype)
                        except Exception as e:
                            st.error(f"Failed to convert '{col_name}' to {selected_type_str}. Reverting. Error: {e}")
                            modified_df[col_name] = st.session_state.df[col_name]

            st.session_state.df = modified_df

        # --- Identifier Column Selection ---
        st.subheader("⚙️ Configure Identifier Columns")
        # ... (This section remains unchanged)
        st.info("The system automatically detects columns that look like identifiers (e.g., 'user_id'). These columns will be ignored during statistical analysis and duplicate row checks. You can review and adjust the selection below.")
        if st.session_state.id_columns_to_ignore is None:
            detected_ids = auto_detect_id_columns(st.session_state.df, st.session_state.target_column)
            st.session_state.id_columns_to_ignore = detected_ids
        selected_id_cols = st.multiselect(
            label="Select the column(s) to treat as identifiers:",
            options=st.session_state.df.columns.tolist(),
            default=st.session_state.id_columns_to_ignore,
            help="These columns will be excluded from numerical analysis, correlations, and duplicate row detection."
        )
        st.session_state.id_columns_to_ignore = selected_id_cols

        # --- Target Column Selection ---
        st.session_state.target_column = st.selectbox(
            "🎯 Select the target column for analysis (e.g., prediction target)",
            options=[col for col in st.session_state.df.columns if col not in st.session_state.id_columns_to_ignore]
        )

        col1, col2 = st.columns([1, 1])

        if col1.button("📊 Run Data Analysis", use_container_width=True):
            with st.spinner("Running comprehensive analysis... This may take a moment."):
                analyzer_instance = DataAnalyzer(
                    df=st.session_state.df,
                    target_column=st.session_state.target_column,
                    id_columns_to_ignore=st.session_state.id_columns_to_ignore
                )
                st.session_state.analysis_results = analyzer_instance.run_full_analysis()
            st.success("✅ Analysis complete! Use the sidebar to explore results.")

        if col2.button("🗑️ Clear Dataset", use_container_width=True):
            keys_to_clear = ['df', 'analysis_results', 'target_column', 'pre_status', 'pre_df', 'modeling_results', 'id_columns_to_ignore']
            for key in keys_to_clear:
                st.session_state[key] = None
            st.rerun()
# --- (The rest of the file remains unchanged) ---

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
    "Model Suggestions"  
]

st.sidebar.title("📚 Navigation")
selected_page = st.sidebar.radio("Go to", pages)

# --- Page Routing Logic ---
if selected_page == "Home":
    display_home_page()

elif selected_page == "Preprocessing Suggestions":
    if st.session_state.analysis_results and st.session_state.df is not None:
        final_df = run_preprocessing_dashboard(
            st.session_state.analysis_results,
            st.session_state.df
        )
        st.session_state.df = final_df
    else:
        st.warning("⚠️ Please upload a dataset and run the analysis first.")

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
                    "- Identify the best base model using AutoGluon\n"
                    "- Use its feature importance to drop irrelevant features\n"
                    "- Show the exact hyperparameters used by that best model"
                )
                if st.button("🚀 Run model search", use_container_width=True):
                    target_column = st.session_state.target_column
                    st.info("🔍 Identifying the best model...")
                    with st.spinner("Identifying the best model..."):
                        ag_results = run_model_suggestions(
                            data_for_modeling,
                            target_column=target_column,
                        )

                    st.info("🧬 Selecting most relevant features...")
                    with st.spinner("Selecting most relevant features..."):
                        reduced_df, selected_features = select_features_by_importance(
                            df=data_for_modeling,
                            target_column=target_column,
                            feature_importance=ag_results.get("feature_importance"),
                            importance_threshold=0.0,
                        )

                    # Save final outputs in session_state
                    st.session_state.modeling_results = {
                        "problem_type": ag_results["problem_type"],
                        "eval_metric": ag_results["eval_metric"],
                        "auto_best_model_name": ag_results["best_model_name"],
                        "auto_best_model_family": ag_results["best_model_family"],
                        "selected_features": selected_features,
                        "auto_best_params": ag_results.get("best_model_params", {}),
                        "auto_best_val_score": ag_results.get("best_val_score", None),
                    }
                    st.success("✅ Best model identified. See details below.")
                    results = st.session_state.modeling_results

            if results is not None:
                st.subheader("🏁 Final Model (AutoGluon)")
                best_model_label = (
                    results.get("auto_best_model_name")
                    or results.get("auto_best_model_family")
                    or "Unknown model"
                )
                st.write(f"**Best Model:** `{best_model_label}`")

                # Show the AutoGluon validation score for the best model
                eval_metric = (results.get("eval_metric") or "").lower()
                eval_score = results.get("auto_best_val_score", None)
                if eval_score is not None:
                    if eval_metric == "roc_auc":
                        st.write(f"**Validation ROC AUC (AutoGluon):** `{eval_score:.4f}`")
                    elif eval_metric == "root_mean_squared_error":
                        st.write(f"**Validation RMSE (AutoGluon):** `{eval_score:.4f}`")
                    elif eval_metric == "log_loss":
                        st.write(f"**Validation log_loss (AutoGluon):** `{eval_score:.4f}`")
                    else:
                        st.write(f"**Validation score (AutoGluon):** `{eval_score:.4f}`")

                # Exact hyperparameters used by the best AG base model
                st.write("**Hyperparameters used by the best model:**")
                st.json(results.get("auto_best_params", {}))

    else:
        st.warning("⚠️ Please upload a dataset and select a target column on the Home page first.")
else:
    if st.session_state.analysis_results:
        create_dashboard(st.session_state.analysis_results, selected_page)
    else:
        st.warning("⚠️ Please upload a dataset and run analysis on the Home page first.")

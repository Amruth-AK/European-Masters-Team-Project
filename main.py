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
from model_scorer import score_and_rank_models
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
if "autogluon_results" not in st.session_state:
    st.session_state.autogluon_results = None
if "ranked_models" not in st.session_state:
    st.session_state.ranked_models = None
if "pre_df_feature_importance" not in st.session_state:
    st.session_state.pre_df_feature_importance = None

# Prediction / pipeline-related state
if "data_schema" not in st.session_state:
    st.session_state.data_schema = None
if "transformation_pipeline" not in st.session_state:
    st.session_state.transformation_pipeline = None
if "fitted_pipeline" not in st.session_state:
    st.session_state.fitted_pipeline = None
if "prediction_results_df" not in st.session_state:
    st.session_state.prediction_results_df = None
if "test_metric_name" not in st.session_state:
    st.session_state.test_metric_name = None
if "test_metric_value" not in st.session_state:
    st.session_state.test_metric_value = None


# --- Validation Helper Function (Unchanged) ---
def get_plausible_conversion_types(series: pd.Series) -> List[str]:
    plausible_types: List[str] = []
    current_dtype = str(series.dtype)
    plausible_types.append("Categorical")
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() / max(series.notna().sum(), 1) > 0.75:
        plausible_types.append("Numerical")
    if pd.api.types.is_numeric_dtype(series.dtype):
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


# --- Home Page (Unchanged) ---
def display_home_page():
    st.title("🚀 Interactive Data Analysis Tool")
    st.write("Upload your dataset to begin a comprehensive analysis and model evaluation.")
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"], key="file_uploader")
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            keys_to_reset = [
                "analysis_results", "pre_status", "pre_df", "id_columns_to_ignore", "modeling_results", 
                "target_column", "data_schema", "transformation_pipeline", "fitted_pipeline", 
                "prediction_results_df", "test_metric_name", "test_metric_value", "autogluon_results", 
                "ranked_models", "pre_df_feature_importance"
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
            # (Logic for type conversion remains the same)
            st.info("The system automatically limits conversion options to prevent errors.")
            dtype_options = {"Numerical": "float64", "Categorical": "category", "Datetime": "datetime64[ns]"}
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
                        options=plausible_options, index=default_index, key=f"dtype_override_{col_name}"
                    )
                    new_dtype = dtype_options[selected_type_str]
                    if new_dtype != inferred_dtype_str:
                        try:
                            if new_dtype == "datetime64[ns]":
                                modified_df[col_name] = pd.to_datetime(modified_df[col_name])
                            else:
                                modified_df[col_name] = modified_df[col_name].astype(new_dtype)
                        except Exception as e:
                            st.error(f"Failed to convert '{col_name}' to {selected_type_str}. Reverting. Error: {e}")
                            modified_df[col_name] = st.session_state.df[col_name]
            st.session_state.df = modified_df
            st.session_state.data_schema = st.session_state.df.dtypes.to_dict()
        st.subheader("⚙️ Configure Identifier Columns")
        st.info("The system automatically detects columns that look like identifiers. These columns will be ignored during statistical analysis.")
        if st.session_state.id_columns_to_ignore is None:
            detected_ids = auto_detect_id_columns(st.session_state.df, st.session_state.target_column)
            st.session_state.id_columns_to_ignore = detected_ids
        selected_id_cols = st.multiselect(
            label="Select the column(s) to treat as identifiers:",
            options=st.session_state.df.columns.tolist(),
            default=st.session_state.id_columns_to_ignore,
        )
        st.session_state.id_columns_to_ignore = selected_id_cols
        st.session_state.target_column = st.selectbox(
            "🎯 Select the target column for analysis",
            options=[col for col in st.session_state.df.columns if col not in st.session_state.id_columns_to_ignore],
        )

# --- Model Recommendations Page (Corrected Logic) ---
def display_model_recommendations_page():
    st.title("🤖 Model Recommendations")

    if st.session_state.df is None or st.session_state.target_column is None:
        st.warning("⚠️ Please upload a dataset and select a target column on the Home page first.")
        return

    # This is the main display logic for when recommendations have NOT been generated yet
    if st.session_state.ranked_models is None:
        st.info(
            "This step combines two approaches to recommend the best model for your data:\n"
            "1.  **Heuristic Analysis:** It scores models based on your dataset's characteristics.\n"
            "2.  **Performance Testing:** It runs a quick automated ML tournament (AutoGluon) on your raw data.\n\n"
            "The final recommendation is a weighted combination of both."
        )
        time_limit_seconds = st.slider(
            "Set a time limit for the performance test (seconds)", 60, 1800, 300, 60,
            help="Longer time allows for more thorough training and may yield better models."
        )
        if st.button("🚀 Generate Model Recommendations", use_container_width=True):
            if st.session_state.analysis_results is None:
                with st.spinner("Step 1/3: Running comprehensive data analysis..."):
                    analyzer = DataAnalyzer(
                        st.session_state.df, st.session_state.target_column, st.session_state.id_columns_to_ignore
                    )
                    st.session_state.analysis_results = analyzer.run_full_analysis()
            try:
                with st.spinner(f"Step 2/3: Running model performance test for {time_limit_seconds}s..."):
                    # Use the main dataframe from the home page
                    ag_results = run_model_suggestions(
                        df=st.session_state.df, target_column=st.session_state.target_column, time_limit=time_limit_seconds
                    )
                    st.session_state.autogluon_results = ag_results
                with st.spinner("Step 3/3: Analyzing results and generating final recommendations..."):
                    ranked = score_and_rank_models(
                        st.session_state.analysis_results, st.session_state.autogluon_results
                    )
                    st.session_state.ranked_models = ranked
                st.success("✅ Recommendations generated!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ An error occurred during performance testing: {e}")
    
    # This is the display logic for AFTER recommendations are generated
    else:
        st.success("Recommendations generated! Here is the ranked list of model families.")
        ranked_models = st.session_state.ranked_models
        top_model = ranked_models[0]
        st.header(f"🥇 Top Recommendation: {top_model['model_family'].replace('_', ' ').title()}")
        st.progress(top_model['normalized_score'] / 100)
        st.metric("Final Score", f"{top_model['normalized_score']}/100")
        with st.expander("Why was this model chosen? (Justification)", expanded=True):
            for reason in top_model['justification']:
                st.markdown(f"- {reason}")
        st.markdown("---")
        st.subheader("🥈 Alternative Models")
        for model in ranked_models[1:]:
            st.markdown(f"#### {model['model_family'].replace('_', ' ').title()}")
            st.progress(model['normalized_score'] / 100)
            st.write(f"**Score:** {model['normalized_score']}/100")
            with st.expander("Justification"):
                for reason in model['justification']:
                    st.markdown(f"- {reason}")
        with st.expander("🔍 View Detailed Performance Test Leaderboard"):
            leaderboard = st.session_state.autogluon_results['leaderboard']
            st.dataframe(leaderboard[['model', 'score_val', 'fit_time', 'pred_time_val']], use_container_width=True)


# --- Final Model Training Page (Unchanged but will now work correctly) ---
def display_final_model_training_page():
    st.title("🏆 Final Model Training")

    if st.session_state.pre_status not in ["applied", "ignored"]:
        st.warning("⚠️ Please finish preprocessing on the 'Preprocessing Suggestions' page first.")
        return

    data_for_modeling = st.session_state.pre_df if st.session_state.pre_df is not None else st.session_state.df

    if st.session_state.pre_df_feature_importance is None:
        st.info("Before tuning, we need to rank the importance of the newly engineered features.")
        if st.button("🔬 Analyze Preprocessed Features", use_container_width=True):
            with st.spinner("Running a quick analysis to rank feature importance..."):
                try:
                    fi_results = run_model_suggestions(
                        df=data_for_modeling, target_column=st.session_state.target_column, 
                        time_limit=120, purpose="feature_importance_only"
                    )
                    st.session_state.pre_df_feature_importance = fi_results.get("feature_importance")
                    st.success("Feature analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during feature analysis: {e}")
        return

    st.header("⚙️ Tune and Train Final Model")
    ranked_models = st.session_state.ranked_models
    autogluon_results = st.session_state.autogluon_results
    top_model_families = [m['model_family'] for m in ranked_models]
    selected_family = st.selectbox(
        "Select a model family to train and tune:", options=top_model_families
    )
    
    if st.button(f"🚀 Train Final {selected_family.replace('_', ' ').title()} Model", use_container_width=True):
        st.session_state.modeling_results = None
        target_column = st.session_state.target_column
        with st.spinner("Step 1/2: Selecting most relevant features..."):
            feature_importance_df = st.session_state.pre_df_feature_importance
            reduced_df, selected_features = select_features_by_importance(
                df=data_for_modeling, target_column=target_column,
                feature_importance=feature_importance_df, importance_threshold=0.0,
            )
        st.success(f"Selected {len(selected_features)} out of {len(data_for_modeling.columns)-1} features.")
        with st.expander("View Selected Features"):
            st.write(selected_features)
        with st.spinner(f"Step 2/2: Finding the best hyperparameters for {selected_family}..."):
            tuning_results = tune_model_with_optuna(
                df=reduced_df, target_column=target_column, model_family=selected_family,
                problem_type=autogluon_results["problem_type"], eval_metric=autogluon_results["eval_metric"],
                n_trials=50, time_limit=300,
            )
        st.session_state.modeling_results = {
            "problem_type": autogluon_results["problem_type"], "eval_metric": autogluon_results["eval_metric"],
            "selected_features": selected_features, "tuned_model_family": selected_family,
            "tuned_model_class": tuning_results["best_model_class"], "tuned_params": tuning_results["best_params"],
            "final_model": tuning_results["best_model"], "eval_score": tuning_results["best_eval_score"],
        }
        st.success("✅ Final model trained and tuned!")
        st.rerun()

    if st.session_state.modeling_results is not None:
        results = st.session_state.modeling_results
        st.subheader("🏁 Final Tuned Model")
        st.write(f"**Model Family:** `{results['tuned_model_family']}`")
        st.metric(f"Validation {results['eval_metric'].upper()}", f"{results['eval_score']:.4f}")
        st.write("**Optimal Hyperparameters:**")
        st.json(results["tuned_params"])


# --- Sidebar Navigation (Unchanged) ---
pages = [
    "Home", "Model Recommendations", "General Info", "Missing Values",
    "Descriptive Statistics", "Distributions", "Categorical Info",
    "Correlations", "Outlier Info", "Duplicate Analysis",
    "Preprocessing Suggestions", "Final Model Training", "Make Predictions",
]
st.sidebar.title("📚 Navigation")
selected_page = st.sidebar.radio("Go to", pages)

# --- Page Routing Logic (Corrected) ---
if selected_page == "Home":
    display_home_page()

elif selected_page == "Model Recommendations":
    display_model_recommendations_page()

elif selected_page == "Preprocessing Suggestions":
    if st.session_state.analysis_results:
        run_preprocessing_dashboard(st.session_state.analysis_results, st.session_state.df)
    else:
        st.warning("⚠️ Please generate recommendations on the 'Model Recommendations' page before proceeding.")

elif selected_page == "Final Model Training":
    if st.session_state.ranked_models:
        display_final_model_training_page()
    else:
        st.warning("⚠️ Please generate recommendations on the 'Model Recommendations' page first.")

elif selected_page == "Make Predictions":
    if st.session_state.modeling_results:
        display_prediction_page()
    else:
        st.warning("⚠️ Please train a final model on the 'Final Model Training' page before making predictions.")

else: # Handles all analysis dashboard pages
    if st.session_state.analysis_results:
        create_dashboard(st.session_state.analysis_results, selected_page)
    else:
        st.warning("⚠️ Please upload a dataset and generate model recommendations on the 'Model Recommendations' page first.")
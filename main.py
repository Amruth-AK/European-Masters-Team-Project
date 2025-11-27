import streamlit as st
import pandas as pd
from typing import List

from analyze import DataAnalyzer, auto_detect_id_columns
from dashboard import create_dashboard
from pre_dashboard import run_preprocessing_dashboard
from download_page import run_download_page

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Interactive Data Analysis Tool",
    page_icon="🚀",
    layout="wide",
)

# ======================================
# SESSION STATE INIT
# ======================================
default_states = {
    "df": None,
    "analysis_results": None,
    "target_column": None,
    "id_columns_to_ignore": None,
    "pre_status": None,
    "pre_df": None,
    "modeling_results": None,
    "data_schema": None,
    "transformation_pipeline": None,
    "fitted_pipeline": None,
    "prediction_results_df": None,
    "test_metric_name": None,
    "test_metric_value": None,
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = {
        "expanded_analysis": True,
        "selected_page": "Home",
        "selected_analysis_subpage": "General Info"
    }

# ======================================
# HELPER FUNCTIONS
# ======================================
def get_plausible_conversion_types(series: pd.Series) -> List[str]:
    plausible_types = ["Categorical"]
    if pd.to_numeric(series, errors="coerce").notna().mean() > 0.75:
        plausible_types.append("Numerical")
    try:
        if pd.to_datetime(series, errors="coerce").notna().mean() > 0.75:
            plausible_types.append("Datetime")
    except Exception:
        pass
    return list(set(plausible_types))

# ======================================
# HOME PAGE
# ======================================
def display_home_page():
    st.title("🚀 Interactive Data Analysis Tool")
    st.write("Upload your dataset to begin a comprehensive analysis and model evaluation.")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            for key in default_states:
                if key != "df":
                    st.session_state[key] = None
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.session_state.df = None

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(), use_container_width=True)

        # Column Type Adjustment
        with st.expander("🔧 Manually Adjust Column Data Types", expanded=False):
            st.info("Options are restricted to prevent invalid conversions.")
            dtype_options = {"Numerical": "float64", "Categorical": "category", "Datetime": "datetime64[ns]"}
            modified_df = st.session_state.df.copy()
            cols_per_row = 3
            ui_cols = st.columns(cols_per_row)

            for i, col_name in enumerate(modified_df.columns):
                with ui_cols[i % cols_per_row]:
                    inferred_dtype_str = str(modified_df[col_name].dtype)
                    current_type_str = (
                        "Numerical" if "int" in inferred_dtype_str or "float" in inferred_dtype_str
                        else "Datetime" if "datetime" in inferred_dtype_str
                        else "Categorical"
                    )
                    plausible = get_plausible_conversion_types(modified_df[col_name])
                    if current_type_str not in plausible:
                        plausible.insert(0, current_type_str)

                    selected_type_str = st.selectbox(
                        f"**{col_name}** (Inferred: *{current_type_str}*)",
                        options=plausible,
                        index=plausible.index(current_type_str),
                        key=f"dtype_override_{col_name}",
                    )

                    new_dtype = dtype_options[selected_type_str]
                    try:
                        if new_dtype == "datetime64[ns]":
                            modified_df[col_name] = pd.to_datetime(modified_df[col_name])
                        else:
                            modified_df[col_name] = modified_df[col_name].astype(new_dtype)
                    except Exception as e:
                        st.error(f"Failed to convert '{col_name}'. Reverting. Error: {e}")
                        modified_df[col_name] = st.session_state.df[col_name]

            st.session_state.df = modified_df
            st.session_state.data_schema = modified_df.dtypes.to_dict()

        # ID Column Selection
        st.subheader("⚙️ Configure Identifier Columns")
        st.info("These columns will be ignored in analysis.")
        if st.session_state.id_columns_to_ignore is None:
            st.session_state.id_columns_to_ignore = auto_detect_id_columns(
                st.session_state.df, st.session_state.target_column
            )

        selected_id_cols = st.multiselect(
            "Select the column(s) to treat as identifiers:",
            options=st.session_state.df.columns.tolist(),
            default=st.session_state.id_columns_to_ignore,
        )
        st.session_state.id_columns_to_ignore = selected_id_cols

        # Target Column Selection
        st.session_state.target_column = st.selectbox(
            "🎯 Select the target column",
            options=[col for col in st.session_state.df.columns if col not in st.session_state.id_columns_to_ignore],
        )

        # Action Buttons
        col1, col2 = st.columns([1, 1])
        if col1.button("📊 Run Data Analysis", use_container_width=True):
            with st.spinner("Running analysis..."):
                analyzer = DataAnalyzer(
                    df=st.session_state.df,
                    target_column=st.session_state.target_column,
                    id_columns_to_ignore=st.session_state.id_columns_to_ignore,
                )
                st.session_state.analysis_results = analyzer.run_full_analysis()
            st.success("✅ Analysis complete! Use the sidebar to explore results.")

        if col2.button("🗑️ Clear Dataset", use_container_width=True):
            for key in default_states:
                st.session_state[key] = None
            st.rerun()


# ======================================
# SIDEBAR NAVIGATION (Modern Tree)
# ======================================
st.sidebar.title("📚 Navigation")

# HOME BUTTON
if st.sidebar.button("🏠 Home"):
    st.session_state.sidebar_state["selected_page"] = "Home"

# ANALYSIS BUTTON
if st.sidebar.button("📊 Analysis"):
    st.session_state.sidebar_state["selected_page"] = "Analysis"
    st.session_state.sidebar_state["expanded_analysis"] = not st.session_state.sidebar_state["expanded_analysis"]

# Analysis Subpages
analysis_subpages = [
    "General Info",
    "Missing Values",
    "Descriptive Statistics",
    "Distributions",
    "Categorical Info",
    "Correlations",
    "Outlier Info",
    "Duplicate Analysis"
]

if st.session_state.sidebar_state["selected_page"] == "Analysis" and st.session_state.sidebar_state["expanded_analysis"]:
    for sub in analysis_subpages:
        if st.sidebar.button(f"   └ {sub}"):
            st.session_state.sidebar_state["selected_analysis_subpage"] = sub

# PREPROCESSING BUTTON
if st.sidebar.button("🛠 Preprocessing Suggestions"):
    st.session_state.sidebar_state["selected_page"] = "Preprocessing"

# DOWNLOAD BUTTON
if st.sidebar.button("⬇ Download Preprocessed Data"):
    st.session_state.sidebar_state["selected_page"] = "Download"

# ======================================
# PAGE ROUTING
# ======================================
selected_page = st.session_state.sidebar_state["selected_page"]

if selected_page == "Home":
    display_home_page()

elif selected_page == "Analysis":
    subpage = st.session_state.sidebar_state["selected_analysis_subpage"]
    if st.session_state.analysis_results:
        create_dashboard(st.session_state.analysis_results, subpage)
    else:
        st.warning("⚠️ Please upload a dataset and run analysis first.")

elif selected_page == "Preprocessing":
    if st.session_state.analysis_results and st.session_state.df is not None:
        run_preprocessing_dashboard(st.session_state.analysis_results, st.session_state.df)
    else:
        st.warning("⚠️ Please upload a dataset and run analysis first.")

elif selected_page == "Download":
    run_download_page()

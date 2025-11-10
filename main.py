import sys
print(sys.executable)
import streamlit as st
import pandas as pd
from analyze import DataAnalyzer
from dashboard import create_dashboard
from pre_dashboard import run_preprocessing_dashboard

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
if 'pre_status' not in st.session_state:
    st.session_state.pre_status = None
if 'pre_df' not in st.session_state:
    st.session_state.pre_df = None

# --- Home Page ---
def display_home_page():
    """Defines the content of the Home page."""
    st.title("🚀 Interactive Data Analysis Tool")
    st.write("Upload your dataset to begin a comprehensive analysis and model evaluation.")

    # File Uploader
    uploaded_file = st.file_uploader(
        "📂 Upload your CSV file",
        type=["csv"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.analysis_results = None
            st.session_state.pre_status = None
            st.session_state.pre_df = None
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.session_state.df = None

    # Show Data Preview
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        # --- NEW: Data Type Override Section ---
        with st.expander("🔧 Manually Adjust Column Data Types", expanded=False):
            st.info("Here you can override the data types inferred by the system. For example, a numerical ID column can be correctly set as 'Categorical'.")

            # Define the mapping from user-friendly names to pandas dtypes
            dtype_options = {
                "Numerical": "float64",
                "Categorical": "category",
                "Datetime": "datetime64[ns]",
                "Object (Text)": "object"
            }
            options_list = list(dtype_options.keys())

            modified_df = st.session_state.df.copy()
            cols_per_row = 3
            ui_cols = st.columns(cols_per_row)

            for i, col_name in enumerate(modified_df.columns):
                with ui_cols[i % cols_per_row]:
                    inferred_dtype = str(modified_df[col_name].dtype)

                    # Determine the default index for the selectbox
                    try:
                        # Find which user-friendly option matches the inferred type
                        default_index = options_list.index(next(key for key, val in dtype_options.items() if val in inferred_dtype))
                    except (StopIteration, ValueError):
                        # If no direct match (e.g., 'int64'), default to Numerical or Object
                        if 'int' in inferred_dtype or 'float' in inferred_dtype:
                             default_index = options_list.index("Numerical")
                        else:
                             default_index = options_list.index("Object (Text)")

                    # Create a selectbox for each column
                    selected_type_str = st.selectbox(
                        label=f"**{col_name}** (Inferred: *{inferred_dtype}*)",
                        options=options_list,
                        index=default_index,
                        key=f"dtype_override_{col_name}" # Unique key is crucial
                    )

                    # Apply the new type if it's different from the original
                    new_dtype = dtype_options[selected_type_str]
                    if new_dtype != inferred_dtype:
                        try:
                            if new_dtype == 'datetime64[ns]':
                                modified_df[col_name] = pd.to_datetime(modified_df[col_name])
                            else:
                                modified_df[col_name] = modified_df[col_name].astype(new_dtype)
                        except Exception as e:
                            st.error(f"Failed to convert '{col_name}' to {selected_type_str}. Reverting. Error: {e}")
                            # Revert on failure
                            modified_df[col_name] = st.session_state.df[col_name]


            # Persist the changes back to the session state
            st.session_state.df = modified_df
        # --- End of New Section ---

        # Target Column Selection
        st.session_state.target_column = st.selectbox(
            "🎯 Select the target column for analysis (e.g., prediction target)",
            options=st.session_state.df.columns
        )

        col1, col2 = st.columns([1, 1])

        # Analyze Button
        if col1.button("📊 Run Data Analysis", use_container_width=True):
            with st.spinner("Running comprehensive analysis... This may take a moment."):
                # Instantiate DataAnalyzer without id_columns_to_ignore for auto-detection
                analyzer_instance = DataAnalyzer(
                    df=st.session_state.df,
                    target_column=st.session_state.target_column
                )
                st.session_state.analysis_results = analyzer_instance.run_full_analysis()
            st.success("✅ Analysis complete! Use the sidebar to explore results.")

        # Remove Dataset Button
        if col2.button("🗑️ Clear Dataset", use_container_width=True):
            for key in ['df', 'analysis_results', 'target_column', 'pre_status', 'pre_df']:
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
    # This page is now a placeholder / info page.
    # Model selection & hyperparameter tuning will be handled
    # internally (e.g. via Optuna) in a separate backend step.
    if st.session_state.df is not None and st.session_state.target_column:
        if st.session_state.pre_status not in ["applied", "ignored"]:
            st.warning(
                "⚠️ Please finish preprocessing first by applying or ignoring the suggestions "
                "on the Preprocessing page."
            )
        else:
            st.title("🤖 Model & Hyperparameter Suggestions")
            st.info(
                "The next step of the pipeline will:\n"
                "- Use the internally selected best model as input to Optuna for hyperparameter tuning\n"
                "- Use feature importance to remove irrelevant features\n\n"
                "This page is currently a placeholder and does not run AutoGluon directly."
            )
    else:
        st.warning(
            "⚠️ Please upload a dataset and select a target column on the Home page first."
        )


else:
    if st.session_state.analysis_results:
        create_dashboard(st.session_state.analysis_results, selected_page)
    else:
        st.warning("⚠️ Please upload a dataset and run analysis on the Home page first.")

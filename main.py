#main.py

import sys
print(sys.executable)
import streamlit as st
import pandas as pd
from analyze import DataAnalyzer
from dashboard import create_dashboard
from pre_dashboard import run_preprocessing_dashboard
from model_suggestion import run_model_suggestions  

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
# The 'id_columns' session state is no longer needed

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

        # Target Column Selection
        st.session_state.target_column = st.selectbox(
            "🎯 Select the target column for analysis (e.g., prediction target)",
            options=st.session_state.df.columns
        )
        
        # ID Column Selection is now removed for automatic detection

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
    if st.session_state.df is not None and st.session_state.target_column:
        if st.session_state.pre_status not in ["applied", "ignored"]:
            st.warning("⚠️ Please finish preprocessing first by applying or ignoring the suggestions.")
        else:
            if "model_results" not in st.session_state or st.session_state.model_results is None:
                st.info("💡 Comparing multiple models to find the best one...")
                st.session_state.model_results = run_model_suggestions(
                    st.session_state.df,
                    st.session_state.target_column
                )
            else:
                st.success("✅ Model training already completed. Showing previous results.")
                model_results = st.session_state.model_results

                if model_results and "leaderboard" in model_results:
                    st.dataframe(model_results["leaderboard"])
                    st.write(f"**Best model:** {model_results['best_model']}")
    else:
        st.warning("⚠️ Please upload a dataset and select a target column on the Home page first.")

else:
    if st.session_state.analysis_results:
        create_dashboard(st.session_state.analysis_results, selected_page)
    else:
        st.warning("⚠️ Please upload a dataset and run analysis on the Home page first.")
# main.py

import streamlit as st
import pandas as pd
from analyze import DataAnalyzer  # Corrected import name
from dashboard import create_dashboard
from pre_dashboard import run_preprocessing_dashboard


# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Data Analysis Tool",
    page_icon="🚀",
    layout="wide"
)

# --- Session State Initialization ---
# This ensures that the variables persist across page navigations
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None


# --- Home Page Function ---
def display_home_page():
    """Defines the content of the Home page."""
    st.title("🚀 Interactive Data Analysis Tool")
    st.write("Upload your dataset to begin a comprehensive analysis.")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here",
        type=["csv"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            # When a new file is uploaded, reset previous analysis
            st.session_state.analysis_results = None
            st.session_state.df = pd.read_csv(uploaded_file)

        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            st.session_state.df = None

    # Logic to display content if a DataFrame is loaded
    if st.session_state.df is not None:
        st.success("File uploaded successfully! Here's a preview of your data:")
        st.dataframe(st.session_state.df.head())

        # Target Column Selection
        st.session_state.target_column = st.selectbox(
            "Select the target column for focused analysis (e.g., for correlations)",
            options=st.session_state.df.columns
        )

        col1, col2 = st.columns([1, 5]) # Create columns for buttons

        # Analyze Data Button
        if col1.button("📊 Analyze Data", use_container_width=True):
            with st.spinner('Performing comprehensive analysis... This might take a moment.'):
                analyzer_instance = DataAnalyzer(
                    df=st.session_state.df,
                    target_column=st.session_state.target_column
                )
                st.session_state.analysis_results = analyzer_instance.run_full_analysis()
            st.success("Analysis Complete! Navigate to other sections using the sidebar.")

        # Remove Dataset Button
        if col2.button("🗑️ Remove Dataset", use_container_width=True):
            st.session_state.df = None
            st.session_state.analysis_results = None
            st.session_state.target_column = None
            # Reruns the script to reflect the cleared state
            st.rerun()


# --- Main App Navigation ---
# Define all pages
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
    "Preprocessing Suggestions"
]

st.sidebar.title("Analysis Sections")
selected_page = st.sidebar.radio("Go to", pages)

if selected_page == "Home":
    display_home_page()

elif selected_page == "Preprocessing Suggestions":
    if st.session_state.analysis_results and st.session_state.df is not None:
        # Call the new preprocessing dashboard and get the final dataset
        final_df = run_preprocessing_dashboard(
            st.session_state.analysis_results,
            st.session_state.df
        )
        
    else:
        st.warning("Please upload a dataset and run analysis on the Home page first.")

else:
    if st.session_state.analysis_results:
        create_dashboard(st.session_state.analysis_results, selected_page)
    else:
        st.warning("Please upload a dataset and run analysis on the Home page first.")




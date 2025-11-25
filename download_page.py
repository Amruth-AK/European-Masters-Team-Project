import streamlit as st
import pandas as pd
from preprocessing_pipeline import apply_fitted_pipeline

def run_download_page():
    st.title("📄 Download Preprocessed Data")

    # --- 1. Check if preprocessing has been applied ---
    if "pre_df" not in st.session_state or st.session_state.pre_df is None:
        st.warning("⚠️ Please upload a TRAIN dataset and apply preprocessing first.")
        return
    
    if "fitted_pipeline" not in st.session_state or st.session_state.fitted_pipeline is None:
        st.warning("⚠️ No fitted preprocessing pipeline found. Apply preprocessing first.")
        return

    pre_df = st.session_state.pre_df
    fitted_pipeline = st.session_state.fitted_pipeline
    analysis_results = st.session_state.analysis_results

    # ------------------------------------------------------------
    # SECTION A — Download Preprocessed TRAIN Dataset
    # ------------------------------------------------------------
    st.subheader("⬇️ Download Preprocessed TRAIN Dataset")

    train_csv = pre_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="💾 Download Processed Train Data",
        data=train_csv,
        file_name="processed_train.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # SECTION B — Upload TEST Dataset
    # ------------------------------------------------------------
    st.subheader("📂 Upload TEST Dataset (Optional)")

    test_file = st.file_uploader("Upload test.csv", type=["csv"])

    if test_file is not None:
        try:
            test_df = pd.read_csv(test_file)
            st.write("### Test Data Preview")
            st.dataframe(test_df.head())

            # Button to transform test set
            if st.button("🚀 Transform Test Set", use_container_width=True):

                with st.spinner("Applying fitted preprocessing pipeline to TEST data..."):
                    processed_test = apply_fitted_pipeline(
                        test_df.copy(),
                        fitted_pipeline,
                        analysis_results=analysis_results,
                    )

                st.success("Test set transformed successfully!")

                st.write("### Preview of Transformed Test Data")
                st.dataframe(processed_test.head())

                # Store for download
                csv_test = processed_test.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Processed Test Data",
                    data=csv_test,
                    file_name="processed_test.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Error processing test set: {e}")

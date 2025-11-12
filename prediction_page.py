# prediction_page.py

import streamlit as st
import pandas as pd
from pre_dashboard import apply_suggestion # We need this function to run the pipeline

def display_prediction_page():
    """
    Renders the page for uploading new data and making predictions.
    """
    st.title("🔮 Make New Predictions")

    # --- 1. Check if a model is ready ---
    if 'modeling_results' not in st.session_state or st.session_state.modeling_results is None:
        st.warning("⚠️ Please train a model first by going through the full workflow on the Home page.")
        st.info("You need to upload a dataset, run analysis, apply preprocessing, and run model suggestions before you can make predictions.")
        return

    st.info("Upload a new CSV file with the same columns as your original training data to make predictions.")
    
    uploaded_file = st.file_uploader("📂 Upload new data for prediction", type=["csv"], key="prediction_uploader")

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(new_df.head())

            if st.button("🚀 Predict on New Data", use_container_width=True):
                # --- NEW: Reset previous results when starting a new prediction ---
                st.session_state.prediction_results_df = None
                
                with st.spinner("Applying transformations and making predictions..."):
                    # ... (The prediction logic remains exactly the same as before) ...
                    recipe = {
                        "schema": st.session_state.data_schema,
                        "pipeline": st.session_state.transformation_pipeline,
                        "final_features": st.session_state.modeling_results['selected_features'],
                        "model": st.session_state.modeling_results['final_model'],
                        "analysis_results": st.session_state.analysis_results
                    }
                    
                    processed_df = new_df.copy()
                    for col, dtype in recipe["schema"].items():
                        if col in processed_df.columns:
                            try:
                                if 'datetime' in str(dtype):
                                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
                                else:
                                    processed_df[col] = processed_df[col].astype(dtype)
                            except Exception:
                                pass # Ignore conversion errors on new data
                    
                    for step in recipe["pipeline"]:
                        processed_df = apply_suggestion(processed_df, step, recipe["analysis_results"])

                    X_pred = pd.DataFrame(columns=recipe["final_features"], index=processed_df.index)
                    for col in recipe["final_features"]:
                        if col in processed_df.columns:
                            X_pred[col] = processed_df[col]
                        else:
                            X_pred[col] = 0
                    
                    X_pred = X_pred.fillna(0)
                    model = recipe["model"]
                    predictions = model.predict(X_pred)

                    results_df = new_df.copy()
                    results_df['prediction'] = predictions

                    # --- MODIFIED: Store results in session state instead of displaying directly ---
                    st.session_state.prediction_results_df = results_df

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
            st.session_state.prediction_results_df = None # Clear on error

    # --- NEW: Display results and download options if predictions exist ---
    if st.session_state.prediction_results_df is not None:
        st.success("✅ Predictions complete!")
        st.subheader("Prediction Results")
        results_df = st.session_state.prediction_results_df
        st.dataframe(results_df)

        st.markdown("---")
        
        # --- NEW: Download Configuration Expander ---
        with st.expander("📥 Configure and Download Predictions", expanded=True):
            st.write("Select the columns you want to include in the download file and rename them if needed.")

            all_columns = results_df.columns.tolist()
            
            # 1. Column Selection
            selected_columns = st.multiselect(
                "Select columns to download:",
                options=all_columns,
                default=all_columns
            )

            # 2. Column Renaming
            rename_mapping = {}
            st.write("**Rename Columns (optional):**")
            
            cols_per_row = 2
            ui_cols = st.columns(cols_per_row)
            
            for i, col in enumerate(selected_columns):
                with ui_cols[i % cols_per_row]:
                    new_name = st.text_input(
                        f"Rename '{col}':",
                        value=col,
                        key=f"rename_{col}"
                    )
                    if new_name != col and new_name: # Ensure new name is not empty
                        rename_mapping[col] = new_name
            
            # 3. Prepare the final DataFrame for download
            if not selected_columns:
                st.warning("Please select at least one column to download.")
            else:
                df_to_download = results_df[selected_columns].copy()
                if rename_mapping:
                    df_to_download.rename(columns=rename_mapping, inplace=True)

                st.subheader("Download Preview")
                st.dataframe(df_to_download.head())
                
                # Convert DataFrame to CSV for download button
                csv = df_to_download.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
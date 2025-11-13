# prediction_page.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss

from preprocessing_pipeline import apply_fitted_pipeline


def display_prediction_page():
    """
    Renders the page for uploading new data and making predictions.
    """
    st.title("🔮 Make New Predictions")

    # --- 1. Check if a model is ready ---
    if "modeling_results" not in st.session_state or st.session_state.modeling_results is None:
        st.warning("⚠️ Please train a model first by going through the full workflow on the Home page.")
        st.info(
            "You need to upload a dataset, run analysis, apply preprocessing, "
            "and run model suggestions before you can make predictions."
        )
        return

    if "fitted_pipeline" not in st.session_state or st.session_state.fitted_pipeline is None:
        st.warning("⚠️ No preprocessing pipeline found. Please apply preprocessing suggestions first.")
        return

    modeling_results = st.session_state.modeling_results
    fitted_pipeline = st.session_state.fitted_pipeline
    data_schema = st.session_state.data_schema
    analysis_results = st.session_state.analysis_results
    target_column = st.session_state.target_column

    st.info("Upload a new CSV file with the same columns as your original training data to make predictions.")

    uploaded_file = st.file_uploader(
        "📂 Upload new data for prediction", type=["csv"], key="prediction_uploader"
    )

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(new_df.head())

            if st.button("🚀 Predict on New Data", use_container_width=True):
                # Reset previous results when starting a new prediction
                st.session_state.prediction_results_df = None

                with st.spinner("Applying transformations and making predictions..."):
                    # 1) Coerce dtypes using stored schema
                    processed_df = new_df.copy()
                    if data_schema is not None:
                        for col, dtype in data_schema.items():
                            if col in processed_df.columns:
                                try:
                                    if "datetime" in str(dtype):
                                        processed_df[col] = pd.to_datetime(
                                            processed_df[col], errors="coerce"
                                        )
                                    else:
                                        processed_df[col] = processed_df[col].astype(dtype)
                                except Exception:
                                    # Ignore conversion errors on new data
                                    pass

                    # 2) Apply the fitted preprocessing pipeline
                    processed_df = apply_fitted_pipeline(
                        processed_df,
                        fitted_pipeline,
                        analysis_results=analysis_results,
                    )

                    # 3) Align to final feature set used during training
                    final_features = modeling_results["selected_features"]
                    X_pred = pd.DataFrame(
                        columns=final_features, index=processed_df.index
                    )

                    for col in final_features:
                        if col in processed_df.columns:
                            X_pred[col] = processed_df[col]
                        else:
                            # If a feature is missing in new data, fill with 0
                            X_pred[col] = 0

                    X_pred = X_pred.fillna(0)

                    # 4) Predict
                    model = modeling_results["final_model"]
                    predictions = model.predict(X_pred)

                    results_df = new_df.copy()
                    results_df["prediction"] = predictions

                    # Store base prediction results in session state
                    st.session_state.prediction_results_df = results_df

                    # 5) Optional: Evaluate if target column is present in the uploaded file
                    if target_column in new_df.columns:
                        y_true = new_df[target_column]
                        # Drop rows with missing target in eval
                        mask = y_true.notna()
                        if mask.sum() > 0:
                            y_eval = y_true[mask]
                            X_eval = X_pred.loc[mask]

                            problem_type = (modeling_results.get("problem_type") or "").lower()
                            eval_metric = (modeling_results.get("eval_metric") or "").lower()

                            try:
                                if problem_type == "regression":
                                    rmse = mean_squared_error(y_eval, model.predict(X_eval), squared=False)
                                    st.success(f"📏 Test RMSE on uploaded data: `{rmse:.4f}`")

                                else:
                                    # classification
                                    if hasattr(model, "predict_proba"):
                                        proba = model.predict_proba(X_eval)
                                        if problem_type == "binary" or len(np.unique(y_eval)) == 2:
                                            # Use positive class probabilities
                                            if proba.shape[1] > 1:
                                                pos_proba = proba[:, 1]
                                            else:
                                                pos_proba = proba[:, 0]

                                            if eval_metric == "roc_auc":
                                                auc = roc_auc_score(y_eval, pos_proba)
                                                st.success(f"📏 Test ROC AUC on uploaded data: `{auc:.4f}`")
                                            else:
                                                # default to AUC if metric doesn't match
                                                auc = roc_auc_score(y_eval, pos_proba)
                                                st.success(f"📏 Test ROC AUC on uploaded data: `{auc:.4f}`")
                                        else:
                                            # multiclass
                                            ll = log_loss(y_eval, proba)
                                            st.success(f"📏 Test Log Loss on uploaded data: `{ll:.4f}`")
                                    else:
                                        st.info("Model does not support probability predictions; skipping test metric.")
                            except Exception as e:
                                st.warning(f"Could not compute test metric on uploaded data: {e}")

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
            st.session_state.prediction_results_df = None

    # --- 2. Post-processing: select / rename columns & download ---
    if "prediction_results_df" in st.session_state and st.session_state.prediction_results_df is not None:
        st.success("✅ Predictions complete!")

        base_df = st.session_state.prediction_results_df

        st.subheader("Prediction Results (first 10 rows)")
        st.dataframe(base_df.head(10))

        st.markdown("### ✂️ Customize columns before download")

        all_columns = list(base_df.columns)

        # 2.1 Select which columns to include
        selected_cols = st.multiselect(
            "Select columns to include in the final prediction file:",
            options=all_columns,
            default=all_columns,
        )

        if not selected_cols:
            st.warning("Please select at least one column to include in the download.")
            return

        df_to_download = base_df[selected_cols].copy()

        # 2.2 Rename columns (optional)
        with st.expander("✏️ Rename columns (optional)", expanded=False):
            rename_mapping = {}
            for col in selected_cols:
                new_name = st.text_input(
                    f"Rename column '{col}'",
                    value=col,
                    key=f"rename_{col}",
                )
                if new_name and new_name != col:
                    rename_mapping[col] = new_name

            if rename_mapping:
                df_to_download = df_to_download.rename(columns=rename_mapping)

        st.subheader("📥 Download Preview")
        st.dataframe(df_to_download.head(10))

        # Convert DataFrame to CSV for download button
        csv_data = df_to_download.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="💾 Download Predictions as CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

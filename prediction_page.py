from autogluon.tabular import TabularPredictor
import streamlit as st
import pandas as pd


def display_prediction_page():
    st.title("🔮 Make New Predictions")

    raw_model_path = st.session_state.get("raw_ag_model_path")
    raw_best_model_name = st.session_state.get("raw_ag_best_model_name")
    target_column = st.session_state.get("target_column")
    raw_eval_metric = st.session_state.get("raw_ag_eval_metric")

    if raw_model_path is None or raw_best_model_name is None or target_column is None:
        st.warning(
            "⚠️ No trained AutoGluon model available yet.\n\n"
            "Please go to **Home**, upload the train dataset, select the target, "
            "and click **📊 Run Data Analysis**."
        )

    uploaded_file = st.file_uploader(
        "📂 Upload test dataset (CSV)", type=["csv"], key="prediction_uploader"
    )

    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading test file: {e}")
            st.session_state.prediction_results_df = None
            return

        st.subheader("Test Data Preview")
        st.dataframe(test_df.head())

        if st.button("🚀 Predict on Test Data", use_container_width=True):
            st.session_state.prediction_results_df = None

            if raw_model_path is None or raw_best_model_name is None or target_column is None:
                st.error(
                    "❌ No trained AutoGluon model is available.\n\n"
                    "Please make sure you have run **📊 Run Data Analysis** on the train dataset first."
                )
                return

            with st.spinner("Making predictions with the trained AutoGluon model..."):
                try:
                    predictor = TabularPredictor.load(raw_model_path)

                    # 👉 RAW test data, just drop target if present
                    if target_column in test_df.columns:
                        X_test = test_df.drop(columns=[target_column])
                        has_target = True
                    else:
                        X_test = test_df.copy()
                        has_target = False

                    preds = predictor.predict(X_test, model=raw_best_model_name)

                    results_df = test_df.copy()
                    results_df["prediction"] = preds
                    st.session_state.prediction_results_df = results_df

                    # Optional: evaluate if test has the true target
                    if has_target:
                        try:
                            eval_df = test_df.copy()
                            eval_metrics = predictor.evaluate(
                                eval_df,
                                model=raw_best_model_name,
                                silent=True,
                            )

                            if raw_eval_metric and raw_eval_metric in eval_metrics:
                                metric_name = raw_eval_metric
                                metric_value = eval_metrics[metric_name]
                            else:
                                metric_name, metric_value = next(iter(eval_metrics.items()))

                            st.success(
                                f"📏 Test {metric_name}: `{metric_value:.4f}` "
                                f"(computed by AutoGluon on your test data)"
                            )
                        except Exception as e:
                            st.warning(f"Could not compute test metric on test data: {e}")

                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {e}")
                    st.session_state.prediction_results_df = None


    # --- Post-processing UI (unchanged in spirit) ---
    if "prediction_results_df" in st.session_state and st.session_state.prediction_results_df is not None:
        st.success("✅ Predictions complete!")

        base_df = st.session_state.prediction_results_df

        st.subheader("Prediction Results (first 10 rows)")
        st.dataframe(base_df.head(10))

        st.markdown("### ✂️ Customize columns before download")

        all_columns = list(base_df.columns)

        selected_cols = st.multiselect(
            "Select columns to include in the final prediction file:",
            options=all_columns,
            default=all_columns,
        )

        if not selected_cols:
            st.warning("Please select at least one column to include in the download.")
            return

        df_to_download = base_df[selected_cols].copy()

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

        csv_data = df_to_download.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="💾 Download Predictions as CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

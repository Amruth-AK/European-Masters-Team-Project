# model_suggestion.py

import pandas as pd
import streamlit as st
from autogluon.tabular import TabularPredictor
import plotly.express as px


def run_model_suggestions(df: pd.DataFrame, target_column: str, time_limit: int = None):
    """
    Trains AutoGluon models and suggests the best-performing one.
    """
    st.title("🤖 Model Suggestion (AutoGluon)")
    st.write(f"Target column selected: **{target_column}**")

    # --- Validation ---
    if target_column not in df.columns:
        st.error("❌ Target column not found in the dataset.")
        return None

    # --- Determine problem type ---
    if pd.api.types.is_numeric_dtype(df[target_column]):
        problem_type = 'regression'
        eval_metric = 'r2'
    else:
        problem_type = 'classification'
        eval_metric = 'accuracy'

    st.write(f"Detected problem type: **{problem_type}**")
    st.write(f"Using evaluation metric: **{eval_metric}**")

    # --- Split data ---
    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)

    st.write(f"Training set: {train_data.shape} rows")
    st.write(f"Test set: {test_data.shape} rows")

    # --- Dynamic model selection ---
    n_rows = df.shape[0]
    if n_rows > 5000:
        model_list = ['GBM', 'RF']
    elif n_rows > 1000:
        model_list = ['GBM', 'RF', 'XGB']
    else:
        model_list = ['GBM', 'RF', 'XGB', 'CAT', 'NN_TORCH']

    hyperparameters = {model: {} for model in model_list}

    # --- Dynamic time limit ---
    if time_limit is None:
        time_limit = max(60, min(600, n_rows // 10))

    st.info(f"Training models: {model_list} for up to {time_limit} seconds total...")

    # --- Train models ---
    try:
        predictor = TabularPredictor(
            label=target_column,
            problem_type=problem_type,
            eval_metric=eval_metric,
            verbosity=1
        ).fit(
            train_data=train_data,
            time_limit=time_limit,
            presets='medium_quality_faster_train',
            hyperparameters=hyperparameters
        )

    except Exception as e:
        st.error(f"❌ Model training failed: {e}")
        return None

    # --- Evaluate leaderboard ---
    leaderboard = None
    try:
        leaderboard = predictor.leaderboard(test_data, silent=True)

        if leaderboard is not None and not leaderboard.empty:
            # Select column for sorting
            score_column = 'score_test' if 'score_test' in leaderboard.columns else 'score_val'

            # Determine if higher values are better
            higher_is_better_metrics = ['r2', 'accuracy', 'roc_auc', 'f1']
            ascending_order = False if eval_metric in higher_is_better_metrics else True

            # Sort leaderboard
            leaderboard = leaderboard.sort_values(by=score_column, ascending=ascending_order).reset_index(drop=True)

            st.subheader("🏆 Model Leaderboard (Sorted by Performance)")
            st.dataframe(leaderboard)

            # Display model ranking
            st.write("### 📋 Model Performance Ranking:")
            for i, row in leaderboard.iterrows():
                model_name = row['model']
                score = row.get(score_column, None)
                if score is not None and not pd.isna(score):
                    st.write(f"{i + 1}. **{model_name}** - {score_column}: {score:.4f}")
                else:
                    st.write(f"{i + 1}. **{model_name}** - {score_column}: N/A")
        else:
            st.warning("⚠️ Leaderboard is empty or None")

    except Exception as e:
        st.warning(f"Leaderboard not available: {e}")

    # --- Determine best model ---
    best_model = None
    try:
        if leaderboard is not None and not leaderboard.empty:
            # Top-ranked model from sorted leaderboard
            best_model = leaderboard.iloc[0]['model']
            st.success(f"✅ Best model: **{best_model}**")

            best_score = leaderboard.iloc[0].get(score_column, None)
            if best_score is not None and not pd.isna(best_score):
                st.write(f"**Performance**: {score_column} = {best_score:.4f}")
        else:
            # Fallback: AutoGluon's built-in best model
            best_model = predictor.get_model_best()
            st.success(f"✅ Best model (fallback): **{best_model}**")
    except Exception as e:
        st.warning(f"Could not determine best model: {e}")

    # --- Feature importance ---
    if best_model is not None:
        st.markdown("### 🔍 Feature Importance (Top 10)")
        try:
            importance_df = predictor.feature_importance(test_data, model=best_model)
            if importance_df is not None and not importance_df.empty:
                st.dataframe(importance_df.head(10))
        except Exception as e:
            st.warning(f"Feature importance not available: {e}")

    # --- Performance comparison chart ---
    if leaderboard is not None and not leaderboard.empty:
        st.markdown("### 📊 Performance Comparison")
        if score_column in leaderboard.columns:
            chart_data = leaderboard[['model', score_column]].copy()
            chart_data = chart_data.sort_values(
                by=score_column,
                ascending=(eval_metric not in higher_is_better_metrics)
            )
            fig = px.bar(
                chart_data,
                x='model',
                y=score_column,
                title=f"Model Performance ({score_column})",
                color=score_column,
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_title="Model", yaxis_title=score_column)
            st.plotly_chart(fig, use_container_width=True)

    return {
        "leaderboard": leaderboard,
        "best_model": best_model,
        "predictor": predictor,
        "problem_type": problem_type
    }

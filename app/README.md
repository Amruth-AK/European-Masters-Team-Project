# ML Compass — Feature Engineering Recommendation Tool

A Streamlit application that analyzes classification datasets and recommends feature engineering transformations using pre-trained meta-models.

## What It Does

Upload a CSV dataset, select your target column, and ML Compass will:

1. **Analyze** your dataset's statistical properties (column types, distributions, meta-features)
2. **Recommend** preprocessing and feature engineering transformations ranked by predicted impact
3. **Train** two LightGBM classifiers — a baseline model and an enhanced model with the recommended transformations applied
4. **Evaluate** both models on a held-out test set, comparing metrics like AUC, log loss, accuracy, and F1
5. **Export** detailed HTML/Markdown/PDF reports summarizing the results

An optional AI chat assistant (powered by Google Gemini) is available in the sidebar for contextual questions about your dataset and recommendations.

## Live Demo

👉 **[Launch the app on Streamlit Cloud](https://mlcompass.streamlit.app)**

## Run Locally

```bash
# 1. Go to your Repo
# E.g.
cd /work/inestp05/Abgabe

# 2. Install dependencies
pip install -e ".[reports]"
pip install streamlit optuna google-generativeai

# 3. Launch the app
streamlit run app/recommend_app.py
```

Requires Python 3.9+.

## Gemini Chat Assistant

The sidebar includes an optional LLM-powered chat assistant. To use it, enter a free Google Gemini API key (get one at [aistudio.google.com](https://aistudio.google.com/app/apikey)) in the sidebar input field. The chat provides contextual guidance about your dataset and the recommended transformations.

## How It Works

ML Compass is built on the `mlcompass` library, which bundles pre-trained LightGBM meta-models. These meta-models were trained on evaluation results from hundreds of OpenML classification tasks, learning which transformations tend to help for datasets with particular statistical characteristics. When you upload a new dataset, the app extracts meta-features, queries the meta-models, and ranks transformations by predicted benefit.

## Project Structure

```
├── app/
│   ├── recommend_app.py     # Main Streamlit app (5-step workflow)
│   ├── ui_components.py     # Reusable UI rendering helpers
│   ├── chat_component.py    # Gemini chat assistant
│   └── report_buttons.py    # Report download buttons
├── mlcompass/                # Core library (analysis, recommendations, transforms, evaluation)
├── requirements.txt         # Dependencies for Streamlit Cloud deployment
└── pyproject.toml           # Package configuration
```

# Auto-ML Meta-Learning System

## Overview

Training models on the results of thousands of experiments so they can predict what will work on new datasets.

---

## Data Collectors

The Data Collectors (`DataCollector_3.py`, `HPCollector.py`, `PipelineDataCollector.py`) are scripts that run large-scale experiments across hundreds of public datasets. They systematically try different feature transforms, hyperparameter configurations, and full pipelines, measure whether each one improves model performance, and log all results to a database. This database becomes the training data for the meta-models.

---

## Meta-Models

There are three families of meta-models, each trained on the data collected above:

- **Feature Engineering (FE) Meta-Model** (`train_meta_model_3.py`) — Learns patterns about which data transforms (e.g. log transforms, group-by aggregations) tend to help or hurt, given the properties of a dataset and column.
- **Hyperparameter (HP) Meta-Model** (`train_hp_meta_model.py`) — Learns to predict optimal hyperparameter configurations for LightGBM given dataset characteristics.
- **Pipeline Meta-Model** (`train_pipeline_meta_model.py`) — Learns to score and rank entire feature engineering pipelines (combinations of transforms), capturing interaction effects between transforms.

Each family includes complementary models: a scorer, a gate (to filter bad candidates), a ranker, and additional specialist models.

---

## The App

`auto_fe_app_3.py` is a Streamlit web app that puts the meta-models to work. A user uploads their own dataset, the app computes its properties, runs candidates through the meta-models, and presents a ranked list of recommended feature transforms and hyperparameters. The user selects what they want, and the app trains a final model with those choices applied.

---

## Running the Files

Commands to run each file are documented in the docstring/header of the respective file.

# MLCompass

A meta-learning system that recommends feature engineering transformations for classification datasets.

MLCompass analyzes a dataset's statistical properties, runs pre-trained meta-models to predict which transformations will improve model performance, applies those transformations, trains and evaluates LightGBM classifiers, and generates detailed reports comparing baseline vs. enhanced results.

## Installation

```bash
pip install -e .

# With report generation support (matplotlib):
pip install -e ".[reports]"
```

Requires Python 3.9+ and the following core dependencies: numpy, pandas, scikit-learn, scipy, and lightgbm.

For a complete working example, see `Library Test/test_showcase.py`.

## Project Structure

```
Abgabe/
├── mlcompass/                  # Installable Python library
│   ├── analysis/              # Column profiling, meta-feature extraction, advisories
│   ├── recommendation/        # Meta-model loading, suggestion generation, verdicts
│   ├── transforms/            # Transform application, column detection, helpers
│   ├── evaluation/            # Model training, metrics, threshold optimization
│   ├── reporting/             # HTML/Markdown/PDF report generation
│   ├── data/meta_models/      # Bundled pre-trained LightGBM meta-models
│   └── constants.py           # Feature schemas, method catalogs, hyperparameters
├── Data Collection and Meta Model Training/
│   ├── generate_task_list.py          # Step 0: Generate OpenML task list
│   ├── count_classification_tasks.py  # Step 0b: Count tasks, print sbatch commands
│   ├── collector_utils.py             # Shared evaluation engine (CV, metrics, I/O)
│   ├── collector_slurm.py             # Unified SLURM worker entry point
│   ├── collect_numerical_transforms.py    # 8 numerical transforms
│   ├── collect_categorical_transforms.py  # 5 categorical encodings
│   ├── collect_interaction_features.py    # Pairwise interaction features
│   ├── collect_row_features.py            # Row-level aggregation features
│   ├── merge_collector_results.py     # Step 2: Merge per-worker CSVs
│   ├── ablate_slurm.py               # Step 3: Ablation study over target formulas
│   ├── train_meta_models.py           # Step 4: Train LightGBM meta-models
│   └── meta_models/                   # Trained model outputs
├── Library Test/
│   ├── test_detailed.py       # 40 test functions covering all library modules
│   └── test_showcase.py       # End-to-end demo pipeline
├── app/
│   ├── recommend_app.py       # Streamlit app: 5-step guided workflow
│   ├── ui_components.py       # Reusable Streamlit rendering helpers
│   ├── report_buttons.py      # Report download buttons
│   └── chat_component.py      # LLM chat assistant (Gemini)
├── pyproject.toml             # Package configuration
├── mlcompass_reference.md     # API reference documentation
└── DOCUMENTATION.docx         # Full reproducibility documentation
```

## Pipeline Overview

The system has two phases:

**Phase 1 — Data Collection and Meta-Model Training** (SLURM cluster):

1. Generate a filtered list of OpenML classification tasks
2. Run four collectors (numerical, categorical, interaction, row-level) across all tasks using repeated 5-fold CV with LightGBM
3. Merge per-worker results into consolidated CSVs
4. Run an ablation study to select the best composite target formula per collector type
5. Train LightGBM meta-models (one regressor + one classifier per collector type)

**Phase 2 — Library Usage** (local):

1. Install mlcompass and load the pre-trained meta-models
2. Profile a new dataset (column types, meta-features, advisories)
3. Generate ranked transformation suggestions via the meta-models
4. Apply selected transformations and train baseline vs. enhanced models
5. Evaluate, compare metrics, and generate reports

## Streamlit Application

```bash
streamlit run app/recommend_app.py
```

The app provides a guided five-step workflow: upload data, review suggestions, train models, evaluate on test data, and export reports. An optional LLM chat assistant (Google Gemini) is available in the sidebar for contextual Q&A.

## Documentation

- **DOCUMENTATION.docx** — Complete reproducibility guide covering all scripts, modules, arguments, and execution order
- **mlcompass_reference.md** — API reference with function signatures, parameters, and return types
- **Library Test/TEST_GUIDE.md** — Detailed per-test documentation
- **Library Test/INDEX.md** — Complete function index



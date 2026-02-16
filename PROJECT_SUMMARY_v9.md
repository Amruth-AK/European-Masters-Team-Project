# Auto-ML Meta-Learning System â€” Project Summary v9

## What Is This Project?

When you build a machine learning model, two critical decisions determine performance: **feature engineering** (which transforms to apply to your data) and **hyperparameter tuning** (which model settings to use). Both are traditionally manual, expert-driven processes.

This project **automates both decisions** using meta-learning. It consists of three complementary systems:

### System 1: Feature Engineering (FE) MetaModel

**Stage 1 (Offline â€” Data Collection + Training):**
We run thousands of experiments across hundreds of public datasets. For each experiment, we try one transform (e.g., "take the log of column X," or "multiply column A by column B") and measure whether the model got better or worse. All results are logged into a database. Then we train a set of predictive models (the "meta-models") on this database â€” they learn patterns like "log transforms tend to help right-skewed numeric columns" or "group-by-mean features help when you have informative categorical columns."

The FE meta-learning system includes **four complementary models** (v5):
1. **Score Regressor** - Predicts mean recommendation score (0-100) for a transform
2. **Gate Classifier** - Filters out likely-harmful transforms using effect size + significance
3. **Ranker** - Sorts surviving candidates by predicted quality
4. **Upside Model** - Quantile regressor predicting 90th percentile of effect size, capturing "high-risk/high-reward" methods that the mean regressor washes out (e.g., arithmetic interactions with high variance)

Additional v6 improvements include **RÂ² scoring for regression** (scale-invariant), **noise floor clamping** (stability), and **richer interaction metadata** (partner column properties included in training).

v7 adds **FeatureWiz** (SULOV + XGBoost column pre-ranking, blended 20% into column selection) and **AutoFeat** (polynomial L1 interaction discovery) as optional integrations. An **archetype guidance module** (`archetype_guidance.py`) bridges the Pipeline meta-model into the FE recommendation system via soft per-method score boosts.

**Stage 2 (Online â€” The App):**
A user uploads their own dataset into a Streamlit web app. The app analyzes the dataset's properties (number of rows, column types, correlations, etc.), generates candidate transforms, and runs them through the meta-models to predict which ones will actually help. The user sees a ranked list of recommendations, selects the ones they want, and the app trains a final model with those transforms applied.

### System 2: Hyperparameter (HP) MetaModel

**Stage 1 (Offline â€” Data Collection + Training):**
We evaluate ~26 different hyperparameter configurations per dataset across hundreds of OpenML datasets using Latin Hypercube Sampling. Each configuration is evaluated via repeated cross-validation, recording multiple performance metrics (AUC, accuracy, RMSE, training time, etc.). We then train four complementary meta-models (v2): (1) an HP Config Scorer that predicts performance given dataset properties and HP settings, (2) a Direct HP Predictor that directly predicts optimal HP values from dataset properties, (3) an HP Config Ranker that ranks candidate configurations, and (4) an Ensemble Predictor that combines all three models for robust HP selection.

HP Meta-Model v2 key improvements:
- **GroupKFold in DirectHPPredictor**: prevents top-K rows from the same dataset leaking across train/val folds
- **GroupKFold CV in ranker**: proper cross-validation instead of single 80/20 split
- **Ensemble inference**: orchestrates scorer + predictor + ranker together (perturbation + scoring + ranking pipeline)
- **HP consistency constraints**: enforces num_leaves â‰¤ 2^max_depth, learning budget bounds, integer rounding

**Stage 2 (Online â€” The App):**
When a user uploads a dataset, the app computes dataset meta-features and uses the HP meta-models to recommend optimal hyperparameters. The meta-model approach (Phase 5) replaces the earlier rule-based archetype system with a data-driven approach that adapts to the specific characteristics of each dataset. If HP meta-models are unavailable, the app falls back to the rule-based archetype system.

### System 3: Pipeline MetaModel (NEW)

**Stage 1 (Offline â€” Data Collection + Training):**
While Systems 1 and 2 evaluate individual transforms and individual HP configs, the Pipeline system evaluates **entire FE pipelines** â€” combinations of multiple transforms applied together end-to-end. This captures interaction effects between transforms (e.g., "log_transform + group_mean together helps more than either alone"). For each dataset, the system generates 10â€“15 diverse pipelines from 7 archetypes and evaluates each via repeated cross-validation. We then train five complementary meta-models:

1. **Pipeline Scorer** (LightGBM regressor) â€” predicts pipeline_delta (improvement over baseline) from dataset + pipeline config features
2. **Pipeline Gate** (LightGBM classifier) â€” filters out pipelines predicted to hurt performance; optimized for high precision
3. **Pipeline Ranker** (LightGBM lambdarank) â€” ranks candidate pipelines against each other within a dataset
4. **FE Sensitivity Predictor** (LightGBM regressor) â€” predicts max achievable delta from dataset features alone ("is FE worth trying at all?")
5. **Archetype Recommender** (LightGBM multiclass) â€” recommends the best pipeline archetype family without scoring all candidates

Key design decisions: target is pipeline_delta (transfers across datasets), task-type normalization on delta, dataset Ã— pipeline interaction features, within-dataset relevance for ranker, gate optimized for precision.

**Stage 2 (Online â€” The App):**
Not yet integrated into the Streamlit app. This is the top remaining priority.

---

## Glossary: Key Terms Explained

### The Data Pipeline

| Term | Plain-English Meaning |
|------|----------------------|
| **Feature** | A column in your dataset. "Age," "Income," and "Zip Code" are all features. |
| **Feature Engineering (FE)** | Creating new columns from existing ones to help the model learn better patterns. For example, creating "Income Ã· Age" from two existing columns. |
| **Transform / Method** | A specific operation applied to one or more columns. Examples: `log_transform` (take the logarithm), `group_mean` (average of column B grouped by category A). See the full list in [Methods Tested](#methods-tested). |
| **Candidate** | A proposed transform that *could* be applied. The system generates many candidates (often 100-400), then scores and filters them down to the best ~10-20 recommendations. |
| **Recommendation** | A candidate that survived scoring and filtering â€” the system is suggesting you actually use it. |
| **Pipeline** | A combination of multiple transforms applied together to a dataset. The pipeline system evaluates these as a unit rather than individually. |
| **Baseline** | The model trained on your raw data with simple default settings. This is the "before" to compare against. |
| **HP-Only** | The model trained on your raw data but with smarter hyperparameters (learning rate, tree depth, etc.) selected by the meta-model (or archetype fallback). Isolates the effect of better settings vs. better features. |
| **Enhanced** | The model trained on data that includes the recommended feature transforms AND smarter hyperparameters. This is the "after." |
| **RÂ² (R-Squared)** | "Coefficient of Determination." A regression metric where 1.0 is perfect and 0.0 is equivalent to predicting the mean. Used in v6 to make regression scores scale-invariant and comparable to classification AUC. |

### The FE Meta-Models

The FE system uses **four separate models** that work as a pipeline, each with a different job:

| Model | What It Does | Analogy |
|-------|-------------|---------|
| **Score Regressor** | Predicts a score (0-100) for how helpful a transform will be (mean effect). | Like predicting a restaurant's average star rating |
| **Gate Model** | Predicts the probability a transform is genuinely helpful vs. noise. Filters out bad candidates. | Like a bouncer at a club â€” "are you on the list?" |
| **Ranker** | Among the candidates that pass the gate, sorts them by expected quality. | Like ranking the VIP guests by importance |
| **Upside Model** | Predicts the 90th percentile effect size â€” captures high-variance "boom or bust" methods. | Like predicting a restaurant's best possible experience (not just average) |

### Scores and Metrics Shown in the App

| Column in App | What It Means | Range | How It's Calculated |
|---------------|---------------|-------|---------------------|
| **Gate P** | Probability that this transform will genuinely improve your model. Higher = more confident. | 0.0 â€” 1.0 | The gate model (LightGBM classifier) outputs a probability based on the transform's metadata â€” column statistics, dataset properties, method type, and pair correlation metrics. |
| **Rank** | Relative ranking score among candidates that passed the gate. Higher = predicted to be more beneficial. | Unbounded (relative) | The ranker model (LightGBM lambdarank) predicts relative ordering. It was trained only on transforms that were empirically helpful, so it specializes in distinguishing "good from great." |
| **Prior** | The historical average performance of this *type* of transform across all training data. | 0-100 | Computed from the training database: average recommendation score of all rows with this method. Green (ðŸŸ¢ â‰¥53) = this method type usually helps. Yellow (ðŸŸ¡ 49-53) = neutral. Red (ðŸ”´ <49) = often harmful. |
| **Score** | The regressor's predicted quality score for this specific transform on your specific dataset. | ~30-70 typical | XGBoost regressor output, combining dataset properties, column statistics, and method type. Adjusted by the method prior (Â±30% of the prior's deviation from 50). |
| **Impact** | Whether this transform creates genuinely new information or just re-encodes existing data. | ðŸŸ¢ additive / ðŸ”„ encoding / ðŸ§¹ cleaning | Hardcoded per method type. Tree-based models (like LightGBM) can already approximate many encoding transforms via histogram binning, so additive transforms are generally more valuable. |
| **Type** | Whether this transform operates on a single column, combines multiple columns, or is global. | ðŸ“Š Single / ðŸ”— Inter. / ðŸŒ Global | Determined by the method â€” interactions combine two or more columns, row_stats is global across all numeric columns. |

### How the Recommendation Score (0-100) Is Calculated

This is the target variable the meta-models are trained to predict. Here's how a single training data point gets its score:

1. **Run the experiment:** Apply transform X to dataset D. Train a model WITH it and WITHOUT it. Measure the difference (called "delta").

2. **Compute raw delta:**
   - **Full-model delta:** Train on ALL features + the new one vs. ALL features without it. Measures marginal contribution.
   - **Individual delta:** Train on ONLY this feature vs. a simple baseline. Measures standalone predictive power.
   - **Blend:** Most methods use 40% full-model + 60% individual. Some methods (like `group_mean`) only use full-model because individual eval doesn't make sense for them.

3. **Soft p-value modifier:** Statistical significance of the improvement. Not a hard cutoff â€” just nudges the score Â±20-40%. A very significant result (p < 0.001) gets +20-40% boost. Insignificant results get -20-30% penalty.

4. **Normalize to 0-100:** Uses the distribution of all raw deltas across the entire training database. The median delta maps to 50. Better-than-average transforms score above 50; worse-than-average score below 50. The mapping uses a tanh curve so extreme outliers don't blow up the scale.

**Key insight:** A score of 50 means "average" â€” this transform neither helps nor hurts compared to the typical transform. A score of 55+ means it's in the top 20% (helpful). Below 45 means it's likely harmful.

### How the Gate Model Decides "Helpful vs. Not"

1. **Training target:** During training, any transform with a recommendation score above the 80th percentile (AND above 50.5) is labeled "helpful" (= 1). Everything else is "not helpful" (= 0). Typically about 20% of transforms are labeled helpful.

2. **Features it uses:** 82+ features about the transform, including:
   - Dataset properties (size, column count, missing data ratio, class imbalance)
   - Column properties (skewness, entropy, correlation with target, outlier ratio)
   - **Partner column properties (v6):** For interactions (AÃ—B), properties of column B (skewness, uniqueness, etc.)
   - Method type (encoded as a number)
   - Pair statistics for interactions (correlation between the two columns, mutual information, scale ratio)

3. **Output:** A probability between 0 and 1. The **calibrated threshold** (typically around 0.20-0.26) determines the cutoff â€” candidates below this probability are filtered out.

4. **Safety net:** If the gate is too strict and filters out almost everything, the system falls back to including the top candidates by gate probability anyway, so you always get at least 5 recommendations.

### How Method Priors Work

The training data clearly shows that some methods are consistently helpful and others are consistently harmful:

| Method | Avg Score | Verdict |
|--------|-----------|---------|
| `row_stats` | 61.1 | ðŸŸ¢ Usually helpful â€” adds row-level summary statistics |
| `group_mean` | 54.6 | ðŸŸ¢ Reliably positive â€” target-encoding-like features |
| `group_std` | 52.2 | ðŸŸ¢ Modestly positive â€” captures within-group variance |
| `division_interaction` | 50.3 | ðŸŸ¡ Neutral â€” it's a coin flip |
| `subtraction_interaction` | 49.4 | ðŸŸ¡ Slightly negative |
| `product_interaction` | 48.7 | ðŸ”´ Tends to hurt â€” often just adds noise |
| `cat_concat` | 43.3 | ðŸ”´ Harmful â€” creates high-cardinality features |
| `impute_median` | 28.0 | ðŸ”´ Very harmful â€” destroys information |

**How priors are used at inference time:**

- **Filtering:** Methods with avg score < 46 AND â‰¥ 10 training samples are automatically excluded from candidates. Currently blocks `cat_concat` and `impute_median`.
- **Score adjustment:** Each candidate's predicted score is nudged by 30% of its method's deviation from neutral (50). So `row_stats` (+11.1 from neutral) gets a +3.3 score boost, while `product_interaction` (-1.4 from neutral) gets a -0.4 penalty. This ensures the system prefers empirically proven methods when the model can't otherwise distinguish between candidates.

### Other Technical Terms

| Term | Meaning |
|------|---------|
| **Archetype** | One of 6 pre-defined dataset "profiles" (Tiny, Small-wide, Medium, etc.) with battle-tested hyperparameter configurations. Used as a fallback when HP meta-models are unavailable. |
| **Cross-validation (CV)** | Splitting data into K parts, training on K-1 and testing on 1, rotating through all splits. Gives a more honest performance estimate than a single train/test split. |
| **GroupKFold** | A variant of cross-validation where rows from the same dataset always stay together in the same fold. Prevents "cheating" where the model memorizes dataset-specific patterns. |
| **AUC** | Area Under the ROC Curve. The primary metric for classification tasks. 0.5 = random guessing, 1.0 = perfect. |
| **NDCG** | Normalized Discounted Cumulative Gain. Measures ranking quality â€” how well the top-ranked items match the actual best items. |
| **Mutual Information (MI)** | A measure of how much knowing column A tells you about column B. Higher MI = stronger relationship. |
| **Sentinel value** | A special placeholder value (like -1) used to explicitly mark "this field doesn't apply" rather than leaving it as missing data. Important because missing values get filled with the median, which can mislead the model. |
| **Calibrated threshold** | The gate probability cutoff that optimizes for finding truly helpful transforms. Determined from training data, typically 0.20-0.26 (not the default 0.50). |
| **Cohen's d** | An effect size metric that measures how many standard deviations a delta is from zero, adjusted by noise floor. Used to compare improvements across datasets with different scales. |
| **Noise floor** | The natural variance in model performance caused by random seed differences, not by the transform itself. Measured via a "null intervention" (re-evaluate without changing anything). Deltas smaller than the noise floor are unreliable. |
| **Tree-guided interaction** | Instead of brute-force testing all column pairs, fit a shallow decision tree to find which pairs have conditional dependencies, then test those. Much more efficient and targeted. |
| **Pipeline archetype** | One of 7 pre-defined pipeline templates (e.g., "minimal-encoding," "heavy-interaction," "kitchen-sink") used by the Pipeline system to generate diverse candidate pipelines for evaluation. |
| **FeatureWiz** | External library (v7) that uses SULOV (Searching for Uncorrelated List Of Variables) + recursive XGBoost to rank columns by independent predictive signal. Blended 20% into composite scores. Optional dependency. |
| **AutoFeat** | External library (v7) that generates polynomial candidate features and selects useful ones via L1 regularization. Used to discover non-obvious interaction pairs that tree-guided and importance-based heuristics miss. Optional dependency. |
| **Archetype guidance** | The bridging mechanism (`archetype_guidance.py`) that converts Pipeline meta-model outputs (archetype probabilities + FE sensitivity) into per-method score boosts for the FE recommendation system. Uses soft weighting: all candidates are generated, but scores are adjusted based on the recommended pipeline strategy. |

---

## Architecture

The project consists of three parallel meta-learning pipelines:

### Feature Engineering (FE) MetaModel Pipeline

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  STAGE 1: Parallel Data Collection (SLURM HPC cluster)              â”‚
â”‚                                                                      â”‚
â”‚  generate_task_list.py                                               â”‚
â”‚      â†’ fetches ALL supervised tasks from OpenML                      â”‚
â”‚      â†’ filters by size (500-5M rows, 4-6000 cols, <100M cells)      â”‚
â”‚      â†’ deduplicates by dataset (one task per dataset)                â”‚
â”‚      â†’ writes task_list.json (~2000+ tasks)                          â”‚
â”‚                                                                      â”‚
â”‚  prefetch_data.py                                                    â”‚
â”‚      â†’ pre-downloads OpenML datasets into shared cache               â”‚
â”‚      â†’ prevents redundant downloads across SLURM workers             â”‚
â”‚                                                                      â”‚
â”‚  run_slurm.sh  (sbatch --array=0-N%50)                              â”‚
â”‚      â†’ launches one DataCollector_slurm.py per task                  â”‚
â”‚      â†’ each worker: reads task_list.json[SLURM_ARRAY_TASK_ID]       â”‚
â”‚      â†’ supports OFFSET for batched submissions                       â”‚
â”‚      â†’ 8 CPUs, 32GB RAM, 4hr wall-time per worker                   â”‚
â”‚                                                                      â”‚
â”‚  DataCollector_slurm.py                                              â”‚
â”‚      â†’ thin wrapper: loads dataset, calls DataCollector_3.py         â”‚
â”‚      â†’ writes per-worker CSV + checkpoint (no file clashes)          â”‚
â”‚      â†’ skips already-completed tasks via checkpoint                  â”‚
â”‚                                                                      â”‚
â”‚  DataCollector_3.py  (the core experiment engine)                    â”‚
â”‚      â†’ 3Ã—5-fold repeated CV per transform                            â”‚
â”‚      â†’ RÂ² scoring for regression (scale invariant)                   â”‚
â”‚      â†’ noise-floor calibration (null intervention)                   â”‚
â”‚      â†’ tree-guided interaction selection + tree_pair_score           â”‚
â”‚      â†’ stratified column selection for wide datasets                 â”‚
â”‚      â†’ adaptive method disabling (stops wasting time)                â”‚
â”‚      â†’ per-dataset time budget (default 30min)                       â”‚
â”‚      â†’ produces CSV rows with 110+ fields each (v6 schema)          â”‚
â”‚                                                                      â”‚
â”‚  merge_results.py                                                    â”‚
â”‚      â†’ merges all per-worker CSVs into meta_learning_db.csv          â”‚
â”‚      â†’ prints worker status summary (done/failed/skipped)            â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  STAGE 1b: FE Meta-Model Training                                    â”‚
â”‚                                                                      â”‚
â”‚  train_meta_model_3.py                                               â”‚
â”‚      â†’ loads merged meta_learning_db.csv                             â”‚
â”‚      â†’ computes recommendation scores (weighted delta + p-value)     â”‚
â”‚      â†’ adaptive normalization to 0-100 (z-score + tanh)              â”‚
â”‚      â†’ trains 4 models:                                              â”‚
â”‚          â€¢ Score Regressor (XGBoost/LightGBM)                        â”‚
â”‚          â€¢ Gate Classifier (LightGBM + calibrated threshold)         â”‚
â”‚          â€¢ Ranker (LightGBM lambdarank on helpful-only rows)         â”‚
â”‚          â€¢ Upside Model (quantile regressor, 90th percentile)        â”‚
â”‚      â†’ learned ensemble weights (Nelder-Mead on OOF predictions)     â”‚
â”‚      â†’ richer method priors (p90, p10, range)                        â”‚
â”‚      â†’ outputs:                                                      â”‚
â”‚          preparator.pkl, meta_model.pkl, gate_model.pkl,             â”‚
â”‚          ranker_model.pkl, upside_model.pkl, ensemble_weights.json,  â”‚
â”‚          method_priors.json, metadata.json                           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Hyperparameter (HP) MetaModel Pipeline

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  STAGE 1: HP Data Collection (SLURM HPC cluster)                    â”‚
â”‚                                                                      â”‚
â”‚  generate_task_list.py  (shared with FE pipeline)                    â”‚
â”‚      â†’ same task list used for both FE and HP collection             â”‚
â”‚                                                                      â”‚
â”‚  run_hp_slurm.sh  (sbatch --array=0-N%50)                           â”‚
â”‚      â†’ launches one HPCollector.py per task                          â”‚
â”‚      â†’ 8 CPUs, 32GB RAM, 1hr wall-time per worker                   â”‚
â”‚                                                                      â”‚
â”‚  HPCollector.py  (HP tuning experiment engine)                       â”‚
â”‚      â†’ computes 30+ dataset-level meta-features                      â”‚
â”‚      â†’ generates 26 HP configs via Latin Hypercube Sampling          â”‚
â”‚          (25 sampled + 1 default config)                             â”‚
â”‚      â†’ evaluates each config via 3Ã—5-fold repeated CV                â”‚
â”‚      â†’ records multiple metrics per config:                          â”‚
â”‚          - Classification: accuracy, balanced_acc, F1, AUC           â”‚
â”‚          - Regression: RMSE, MAE, RÂ², MAPE                           â”‚
â”‚          - Training time, actual n_estimators                        â”‚
â”‚      â†’ computes derived targets:                                     â”‚
â”‚          - primary_score (AUC or neg-RMSE)                           â”‚
â”‚          - rank_in_dataset, delta_vs_default                         â”‚
â”‚          - normalized_score, pct_of_best                             â”‚
â”‚      â†’ produces CSV rows with 70+ fields per HP config               â”‚
â”‚                                                                      â”‚
â”‚  merge_hp_results.py                                                 â”‚
â”‚      â†’ merges all per-worker CSVs into hp_tuning_db.csv              â”‚
â”‚      â†’ prints worker status summary                                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  STAGE 1b: HP Meta-Model Training (v2)                               â”‚
â”‚                                                                      â”‚
â”‚  train_hp_meta_model.py                                              â”‚
â”‚      â†’ loads merged hp_tuning_db.csv                                 â”‚
â”‚      â†’ trains 4 complementary models:                                â”‚
â”‚          â€¢ HP Config Scorer (LightGBM regressor)                     â”‚
â”‚            - predicts primary_score from dataset + HP features       â”‚
â”‚            - trained on ALL configs (full HP landscape)              â”‚
â”‚          â€¢ Direct HP Predictor (10 LightGBM regressors)              â”‚
â”‚            - one model per HP parameter                              â”‚
â”‚            - predicts optimal HP values from dataset features        â”‚
â”‚            - trained on best configs per dataset                     â”‚
â”‚            - GroupKFold CV (v2: prevents dataset leakage)            â”‚
â”‚          â€¢ HP Config Ranker (LightGBM lambdarank)                    â”‚
â”‚            - ranks candidate configs within a dataset                â”‚
â”‚            - trained on ALL configs, grouped by dataset              â”‚
â”‚            - GroupKFold CV (v2: proper cross-validation)             â”‚
â”‚          â€¢ Ensemble Predictor (v2: combines all 3 models)            â”‚
â”‚            - Predictor generates initial config                      â”‚
â”‚            - perturbations scored and ranked                         â”‚
â”‚            - weighted combination of scorer + ranker                 â”‚
â”‚      â†’ HP consistency constraints (v2):                              â”‚
â”‚          num_leaves â‰¤ 2^max_depth, learning budget bounds,           â”‚
â”‚          proper integer rounding                                     â”‚
â”‚      â†’ computes dataset archetypes via K-Means clustering            â”‚
â”‚          (6 clusters on optimal HP profiles)                         â”‚
â”‚      â†’ outputs:                                                      â”‚
â”‚          hp_preparator.pkl, hp_scorer.pkl, hp_predictor.pkl,         â”‚
â”‚          hp_ranker.pkl, hp_archetypes.json, hp_metadata.json         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Pipeline MetaModel Pipeline (NEW)

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  STAGE 1: Pipeline Data Collection (SLURM HPC cluster)              â”‚
â”‚                                                                      â”‚
â”‚  generate_task_list.py  (shared with FE + HP pipelines)              â”‚
â”‚      â†’ same task list used for all three collection systems          â”‚
â”‚                                                                      â”‚
â”‚  run_pipeline_slurm.sh  (sbatch --array=0-N%50)                     â”‚
â”‚      â†’ launches one PipelineCollector_slurm.py per task              â”‚
â”‚      â†’ 8 CPUs, 32GB RAM, 5hr wall-time per worker                   â”‚
â”‚                                                                      â”‚
â”‚  PipelineCollector_slurm.py                                          â”‚
â”‚      â†’ thin wrapper: loads dataset, calls PipelineDataCollector      â”‚
â”‚      â†’ writes per-worker CSV + checkpoint (no file clashes)          â”‚
â”‚      â†’ reuses DataCollector_3.py's MetaDataCollector as transform    â”‚
â”‚        engine for shared functionality                               â”‚
â”‚                                                                      â”‚
â”‚  PipelineDataCollector.py  (pipeline experiment engine)              â”‚
â”‚      â†’ data cleanup (ID removal, pruning, leakage detection)         â”‚
â”‚      â†’ baseline CV evaluation                                        â”‚
â”‚      â†’ column profiling (types, importance, properties)              â”‚
â”‚      â†’ generates 10-15 diverse pipelines from 7 archetypes           â”‚
â”‚      â†’ evaluates each pipeline end-to-end with repeated K-fold CV    â”‚
â”‚      â†’ captures 29 dataset metadata + 23 pipeline config features    â”‚
â”‚        + 8 evaluation results per pipeline                           â”‚
â”‚      â†’ tracks transform order, feature expansion, importance         â”‚
â”‚        coverage, method diversity                                    â”‚
â”‚      â†’ produces CSV rows following PIPELINE_CSV_SCHEMA (65 fields)   â”‚
â”‚                                                                      â”‚
â”‚  merge_pipeline_results.py                                           â”‚
â”‚      â†’ merges all per-worker CSVs into pipeline_meta_learning_db.csv â”‚
â”‚      â†’ prints pipeline-level statistics (improvement rates,          â”‚
â”‚        archetype breakdown, etc.)                                    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  STAGE 1b: Pipeline Meta-Model Training                              â”‚
â”‚                                                                      â”‚
â”‚  train_pipeline_meta_model.py                                        â”‚
â”‚      â†’ loads merged pipeline_meta_learning_db.csv                    â”‚
â”‚      â†’ task-type normalization on delta (z-score within clf/reg)     â”‚
â”‚      â†’ dataset Ã— pipeline interaction features                       â”‚
â”‚      â†’ trains 5 complementary models:                                â”‚
â”‚          â€¢ Pipeline Scorer (LightGBM regressor)                      â”‚
â”‚            - predicts pipeline_delta from dataset + pipeline feats   â”‚
â”‚            - trained on ALL rows (full pipeline landscape)           â”‚
â”‚          â€¢ Pipeline Gate (LightGBM binary classifier)                â”‚
â”‚            - P(pipeline significantly improves over baseline)        â”‚
â”‚            - optimized for precision (better to skip than to harm)   â”‚
â”‚          â€¢ Pipeline Ranker (LightGBM lambdarank)                     â”‚
â”‚            - within-dataset relevance labels                         â”‚
â”‚            - ranks candidate pipelines against each other            â”‚
â”‚          â€¢ FE Sensitivity Predictor (LightGBM regressor)             â”‚
â”‚            - dataset features only (no pipeline features)            â”‚
â”‚            - "is FE worth trying at all for this dataset?"           â”‚
â”‚            - trained on best-pipeline-per-dataset                    â”‚
â”‚          â€¢ Archetype Recommender (LightGBM multiclass)               â”‚
â”‚            - predicts best pipeline archetype family                  â”‚
â”‚            - fast path without scoring all candidates                â”‚
â”‚      â†’ pipeline archetype profiles (summary stats per archetype)     â”‚
â”‚      â†’ outputs:                                                      â”‚
â”‚          pipeline_scorer.pkl, pipeline_gate.pkl,                     â”‚
â”‚          pipeline_ranker.pkl, sensitivity_predictor.pkl,             â”‚
â”‚          archetype_recommender.pkl, pipeline_metadata.json           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### User-Facing Application

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  STAGE 2: User-Facing App                                            â”‚
â”‚                                                                      â”‚
â”‚  auto_fe_app_3.py  (Streamlit)                                       â”‚
â”‚      â†’ loads all FE models + HP models + priors                      â”‚
â”‚      â†’ full recommendation pipeline:                                 â”‚
â”‚                                                                      â”‚
â”‚      User uploads CSV                                                â”‚
â”‚          â†’ Analyze dataset properties (50+ features)                 â”‚
â”‚                                                                      â”‚
â”‚          [HP RECOMMENDATION PHASE]                                   â”‚
â”‚          â†’ Compute dataset meta-features                             â”‚
â”‚          â†’ Use HP meta-models to recommend hyperparameters:          â”‚
â”‚              - Direct predictor: fast single-pass prediction         â”‚
â”‚              - Scorer + Ranker: evaluate candidate configs           â”‚
â”‚              - Archetype fallback: if meta-models unavailable        â”‚
â”‚                                                                      â”‚
â”‚          [FE RECOMMENDATION PHASE]                                   â”‚
â”‚          â†’ Generate candidates (100-400 transform proposals)         â”‚
â”‚          â†’ Filter by method priors (remove known-harmful)            â”‚
â”‚          â†’ Score each candidate (regressor + prior adjustment)       â”‚
â”‚          â†’ Gate filter (keep candidates likely to help)              â”‚
â”‚          â†’ Rank surviving candidates (sort by quality)               â”‚
â”‚          â†’ Deduplicate & diversify (avoid redundant transforms)      â”‚
â”‚          â†’ Present top-k recommendations with checkboxes             â”‚
â”‚                                                                      â”‚
â”‚          [MODEL TRAINING & COMPARISON]                               â”‚
â”‚          â†’ User selects transforms â†’ Apply transforms                â”‚
â”‚          â†’ Train 3 models with recommended HPs:                      â”‚
â”‚              - Baseline (raw data + default HPs)                     â”‚
â”‚              - HP-Only (raw data + recommended HPs)                  â”‚
â”‚              - Enhanced (transformed data + recommended HPs)         â”‚
â”‚          â†’ Compare: Baseline vs HP-Only vs Enhanced                  â”‚
â”‚                                                                      â”‚
â”‚          [PIPELINE META-MODEL â€” NOT YET INTEGRATED]                  â”‚
â”‚          â†’ Future: use Pipeline models to recommend whole pipelines  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## All Project Files

### Feature Engineering (FE) MetaModel

| File | Lines | Purpose |
|------|-------|---------|
| `DataCollector_3.py` | ~4084 | Core FE experiment engine. **v7 Updates:** FeatureWiz integration (SULOV + XGBoost column pre-ranking, blends 20% into composite_predictive_score, adds `featurewiz_selected` and `featurewiz_importance` CSV fields), AutoFeat integration (polynomial L1 interaction discovery, adds `autofeat_discovered` interaction source). Both optional with graceful fallback. New init flags: `use_autofeat`, `use_featurewiz`, `autofeat_max_gb`, `autofeat_feateng_steps`, `featurewiz_corr_limit`. **v6:** RÂ² scoring for regression, noise-floor clamping, 16 new interaction metadata fields (`col_b_*`, `col_c_*`), `tree_pair_score` logging, V6 CSV schema. |
| `DataCollector_slurm.py` | ~299 | SLURM array worker wrapper: reads task_list.json, processes one dataset per worker, writes per-worker CSV and checkpoint files. Handles smart size reduction, runtime filters, and OpenML cache management. |
| `generate_task_list.py` | ~251 | Pre-generates a shared task list from OpenML for SLURM array jobs. Fetches all supervised tasks, filters by size, deduplicates by dataset, and writes task_list.json. |
| `run_slurm.sh` | ~93 | SLURM batch script: configures resources (8 CPUs, 32GB RAM, 4hr), sets threading variables, supports OFFSET for batched submissions, launches one DataCollector_slurm.py per array index. |
| `merge_results.py` | ~107 | Post-processing: merges all per-worker CSVs into a single meta_learning_db.csv with schema enforcement, prints worker status summary (done/failed/skipped). |
| `prefetch_data.py` | ~37 | Pre-downloads OpenML datasets into a shared cache directory before SLURM array submission, preventing redundant downloads across SLURM workers. |
| `train_meta_model_3.py` | ~1739 | Trains all FE meta-models (v5). Includes Score Regressor, Gate Classifier, Ranker, Upside Model, learned ensemble weights, method family features, missingness indicators, feature interactions, continuous ranker relevance, and richer method priors. |
| `auto_fe_app_3.py` | ~3015 | Streamlit web app. Loads FE + HP meta-models, generates candidates, scores/gates/ranks them, trains baseline/HP-only/enhanced models. **Phase 5:** Integrates HP meta-models with fallback to rule-based archetypes. |

### Hyperparameter (HP) MetaModel

| File | Lines | Purpose |
|------|-------|---------|
| `HPCollector.py` | ~1629 | Core HP tuning experiment engine: computes dataset-level meta-features, generates 26 HP configs via Latin Hypercube Sampling (25 sampled + 1 default), evaluates each via 3Ã—5-fold repeated CV, records multiple performance metrics (AUC, accuracy, RMSE, MAE, F1, training time), computes derived targets (rank, delta vs default, normalized score), produces CSV rows with 70+ fields per HP config. |
| `run_hp_slurm.sh` | ~100 | SLURM batch script for HP collection: configures resources (8 CPUs, 32GB RAM, 1hr), launches one HPCollector.py per array index. |
| `merge_hp_results.py` | ~164 | Post-processing: merges all per-worker HP CSVs into hp_tuning_db.csv with schema enforcement, deduplication, prints worker status summary. |
| `train_hp_meta_model.py` | ~1727 | Trains all HP meta-models **(v2)**. Produces four complementary models: (1) HP Config Scorer, (2) Direct HP Predictor (GroupKFold CV), (3) HP Config Ranker (GroupKFold CV), (4) Ensemble Predictor (combines all three). Adds HP consistency constraints and dataset archetype clustering via K-Means. |

### Pipeline MetaModel (NEW)

| File | Lines | Purpose |
|------|-------|---------|
| `PipelineDataCollector.py` | ~1217 | Core pipeline experiment engine: generates 10-15 diverse FE pipelines from 7 archetypes, evaluates each end-to-end with repeated K-fold CV. Reuses MetaDataCollector as transform engine. Tracks transform order, feature expansion ratio, importance coverage, method diversity. Produces CSV rows with 65 fields per (dataset, pipeline) pair. |
| `PipelineCollector_slurm.py` | ~278 | SLURM array worker wrapper for pipeline collection: reads task_list.json, processes one dataset with PipelineDataCollector, writes per-worker CSV and checkpoint files. |
| `run_pipeline_slurm.sh` | ~93 | SLURM batch script for pipeline collection: configures resources (8 CPUs, 32GB RAM, 5hr), launches one PipelineCollector_slurm.py per array index. |
| `merge_pipeline_results.py` | ~273 | Post-processing: merges all per-worker pipeline CSVs into pipeline_meta_learning_db.csv with schema enforcement, prints pipeline-level statistics (improvement rates, archetype breakdown, etc.). |
| `train_pipeline_meta_model.py` | ~1416 | Trains all pipeline meta-models. Produces five complementary models: (1) Pipeline Scorer, (2) Pipeline Gate, (3) Pipeline Ranker, (4) FE Sensitivity Predictor, (5) Archetype Recommender. Uses task-type normalization, datasetÃ—pipeline interaction features, within-dataset relevance labels. |

| `archetype_guidance.py` | ~370 | Bridges the Pipeline Meta-Model and FE Meta-Model. Converts pipeline archetype probabilities into per-method score boosts for FE candidate scoring. Uses sensitivity predictor ("is FE worth trying?") and archetype recommender (7 archetype probabilities) to compute soft method-level boosts weighted by prediction confidence and FE sensitivity. Includes `ArchetypeGuidance` class, `ARCHETYPE_METHOD_WEIGHTS` mapping, and `load_pipeline_guidance_models()` loader. |

### Cross-Cutting / Integration

| File | Lines | Purpose |
|------|-------|---------|
| `v7_integration_roadmap.md` | ~184 | Downstream integration roadmap for DataCollector v7 (AutoFeat + FeatureWiz). Covers update sequence across 5 phases: (1) Data infrastructure (merge_results.py, DataCollector_slurm.py), (2) Meta-model trainer (feature list, sentinel handling, interaction_source encoding), (3) Streamlit app (sentinel vs. full FeatureWiz path), (4) Pipeline system (no changes needed), (5) Backward compatibility. Includes task checklist with effort estimates and recommended execution order. |

---

## Implementation Status

### âœ… Phase 0: Quick Fixes
| Fix | What It Does |
|-----|-------------|
| 0.1 Deduplicate training data | Tracks tested interactions with commutative awareness (AÃ—B = BÃ—A) |
| 0.2 Deduplicate recommendations | Diversity constraint + column-overlap penalty in recommendation engine |
| 0.3 Conditional row-stats | Only computes informative stats (skips `row_missing_ratio` if no nulls, etc.) |
| 0.4 Weaken baseline model | Simple defaults (max_bin=63, no regularization) so there's room to improve |

### âœ… Phase 1: Meta-Model Architecture
| Fix | What It Does |
|-----|-------------|
| 1.1 Binary gate target | Data-driven threshold (top 20% AND above neutral) instead of hardcoded cutoff |
| 1.2 Gate model | LightGBM classifier with StratifiedKFold, AUC optimization, calibrated threshold |
| 1.3 Ranking model | LightGBM lambdarank trained on helpful-only rows, NDCG evaluation |
| 1.4 App integration | Full Gateâ†’Ranker pipeline with adaptive display and backward compatibility |

### âœ… Phase 2 (Partial): Data Collection Improvements
| Fix | What It Does |
|-----|-------------|
| 2.1 Aggregate features | Added `n_numeric_cols`, `n_cat_cols`, feature importance statistics to every row |
| 2.3 Wider interaction pools | Expanded from top-4/3/3 to top-6/5/5 for arithmetic/categorical/group-by |
| 2.5 Relative headroom | Added `relative_headroom = 1.0 - baseline_score` â€” how much room the dataset has to improve |

### âœ… Phase 3: HP Prediction via Archetypes
| Fix | What It Does |
|-----|-------------|
| 3.1 Define archetypes | 6 dataset profiles with battle-tested HP configurations |
| 3.2 Archetype matching | Hard-rule matching on size/shape/categorical ratio |
| 3.3 HP meta-model | Full HP meta-model system with scorer, predictor, ranker, and archetype clustering |

### âœ… Phase 4: App & UX
| Fix | What It Does |
|-----|-------------|
| 4.1 Adaptive recommendation count | Smart default `min(max(n_cols * 0.3, 5), 30)` |
| 4.2 3-model comparison | Baseline / HP-Only / Enhanced with attribution breakdown |
| 4.3 Impact type tagging | Additive/Encoding/Cleaning labels with optional additive boost |

### âœ… Phase 5: HP Meta-Model v2 + App Integration
| Feature | What It Does |
|---------|-------------|
| HP Meta-Model v2 | GroupKFold CV in predictor and ranker, ensemble inference, HP consistency constraints |
| App integration (`compute_lgbm_params_meta`) | App loads HP meta-models and uses predictorâ†’scorerâ†’ranker pipeline for HP selection |
| Archetype fallback | If HP meta-models unavailable, app falls back to rule-based archetype system |

### âœ… Critical Fixes (Sessions 10-13)
| Fix | Problem | Solution |
|-----|---------|----------|
| Score normalization | All scores clustered at 50Â±2, gate couldn't learn | Adaptive normalization: z-score + tanh mapping, now spread 0-100 with std=14.4 |
| Feature importance display | All features showed "0" importance | Fixed formatting from `{val:.0f}` to percentage display |
| GroupKFold CV | Dataset leakage inflated metrics | GroupKFold ensures no dataset appears in both train and validation |
| Gate model training | F1=0.000 â€” predicted all-negative | Switched to StratifiedKFold + AUC metric + calibrated threshold (now AUC=0.58) |
| Gate NaN problem | Interaction features NaN for single-column â†’ median-imputed â†’ gate biased toward interactions | Sentinel value (-1) for non-interactions so gate can distinguish "not an interaction" from "moderate interaction" |
| Method priors | System recommended neutral/harmful methods with high confidence | Empirical method performance saved and used to filter/adjust at inference |
| Candidate filtering | `cat_concat` (43.3) and `impute_median` (28.0) consistently recommended | Auto-excluded if mean score < 46 with sufficient sample count |
| Evaluation pipeline | No clear signal whether FE was helping or hurting | Added FE impact verdict, feature importance analysis, warnings when FE degrades performance |

### âœ… SLURM HPC Infrastructure (Sessions 14+)
| Feature | What It Does |
|---------|-------------|
| `generate_task_list.py` | Fetches all supervised tasks from OpenML, filters by size/quality, deduplicates by dataset, outputs task_list.json. Supports configurable limits (min/max rows, cols, cell count). |
| `prefetch_data.py` | Pre-downloads all OpenML datasets into shared cache directory on /work filesystem, preventing redundant downloads across SLURM workers. |
| `run_slurm.sh` | SLURM batch script with 8 CPUs, 32GB RAM, 4hr wall-time. Sets OMP/MKL/OPENBLAS threading to match SLURM allocation. Supports OFFSET variable for batched array submissions. |
| `DataCollector_slurm.py` | Thin SLURM wrapper: reads one task from task_list.json by array index, loads dataset, applies runtime size filters (â‰¥500 rows, â‰¥3 cols), calls DataCollector_3.py, writes per-worker CSV + checkpoint. Checkpoint allows resuming failed/resubmitted jobs. |
| `merge_results.py` | Post-processing: reads all per-worker CSVs + checkpoints, enforces CSV schema alignment, merges into single meta_learning_db.csv, prints status summary (done/failed/skipped counts + failure reasons). |
| OFFSET mechanism | Allows splitting large task lists into batches (e.g., `OFFSET=500 sbatch --array=0-499`), useful when cluster policies limit array sizes. |
| Smart size reduction | `_smart_size_reduction()` in DataCollector_3.py intelligently downsamples oversized datasets (>100M cells) by pruning low-importance columns and subsampling rows, instead of skipping them entirely. |
| Adaptive method tracking | DataCollector tracks per-method success rates during a dataset; disables a method after 10+ columns with 0 improvements â€” saves significant compute. |

### âœ… DataCollector v6 + Downstream Integration (Session 16)
| Feature | What It Does |
|---------|-------------|
| **RÂ² Scoring for Regression** | Switched from MSE/RMSE to RÂ² for regression tasks in `DataCollector_3.py`. Aligning metrics to "higher is better" (like AUC) allows uniform normalization logic across all task types. |
| **Noise Floor Clamping** | Fixed critical bug where near-zero noise floor caused metric explosions (Cohen's d â†’ âˆž). Clamped `null_std` to `1e-4` minimum. |
| **Partner Column Metadata** | Added 16 features (`col_b_*`, `col_c_*`) describing the *other* column(s) in an interaction. Enables meta-models to learn rules like "multiply X by Y if Y is skewed." |
| **Tree-Guided Scores** | Added `tree_pair_score` feature, exposing the decision-tree-derived interaction strength directly to the meta-model. |
| **Downstream Compatibility** | Updated `train_meta_model_3.py` to train on new features. Updated `auto_fe_app_3.py` to populate them at inference time. |
| **Relative Headroom Fix** | Fixed `relative_headroom` calculation in inference app to correctly handle regression scores `(max(1.0 - r2, 0.001))`. |

### âœ… Pipeline Meta-Model System (Session 17) â€” NEW
| Feature | What It Does |
|---------|-------------|
| **PipelineDataCollector.py** | Evaluates entire FE pipelines (combinations of transforms) end-to-end. Generates 10-15 diverse pipelines per dataset from 7 archetypes. Captures 29 dataset metadata + 23 pipeline config features + 8 evaluation results. Reuses MetaDataCollector as transform engine. |
| **PipelineCollector_slurm.py** | SLURM wrapper for pipeline collection: reads task_list.json, processes one dataset with PipelineDataCollector, manages checkpoints. |
| **run_pipeline_slurm.sh** | SLURM batch script for pipeline collection (8 CPUs, 32GB RAM, 5hr wall-time). |
| **merge_pipeline_results.py** | Merges per-worker pipeline CSVs into unified pipeline_meta_learning_db.csv with pipeline-level statistics. |
| **train_pipeline_meta_model.py** | Trains 5 complementary pipeline meta-models: Pipeline Scorer, Pipeline Gate, Pipeline Ranker, FE Sensitivity Predictor, Archetype Recommender. Uses task-type normalization, datasetÃ—pipeline interaction features, within-dataset relevance. |

### âœ… DataCollector v7: AutoFeat + FeatureWiz (Session 19)
| Feature | What It Does |
|---------|-------------|
| **FeatureWiz Integration** | SULOV + recursive XGBoost column pre-ranking. Runs before column selection, identifies columns with independent predictive signal. Blends 20% into `composite_predictive_score` for improved column selection and interaction pool prioritization. Adds `featurewiz_selected` and `featurewiz_importance` per-row CSV fields for meta-model training. |
| **AutoFeat Integration** | Polynomial feature generation + L1 selection to discover non-obvious interaction pairs. Runs after tree-guided pairs, discovers pairs the importance-based heuristic misses. AutoFeat-discovered pairs are evaluated with `interaction_source='autofeat_discovered'` source label. |
| **Optional Dependencies** | Both libraries are optional (`pip install autofeat featurewiz`). When not installed, behavior is identical to v6 with zero changes. Controlled via `use_autofeat`, `use_featurewiz` init flags. |
| **archetype_guidance.py** | Bridging module connecting Pipeline Meta-Model to FE Meta-Model. Converts pipeline archetype probabilities into per-method score boosts. Uses `ArchetypeGuidance` class with `ARCHETYPE_METHOD_WEIGHTS` mapping (7 archetypes Ã— 25 methods). Soft weighting: all candidate types still generated, archetype probs control score boosts scaled by prediction confidence and FE sensitivity. |
| **v7_integration_roadmap.md** | 5-phase downstream update plan covering: (1) data infrastructure, (2) meta-model trainer updates, (3) Streamlit app compatibility, (4) pipeline system (no changes needed), (5) backward CSV compatibility. 11-task checklist with effort estimates. Minimum viable integration: tasks 4+5+7 (~30 min). Full integration: all tasks (~2 hours + SLURM run). |

---

## v7 Integration Roadmap Summary

DataCollector v7 adds two new CSV fields (`featurewiz_selected`, `featurewiz_importance`) and a new `interaction_source` value (`autofeat_discovered`). Downstream files need updates in this order:

**Phase 1 â€" Data Infrastructure:** `merge_results.py` needs to handle the two new schema columns (auto-handled via CSV_SCHEMA import). `DataCollector_slurm.py` needs CLI flags for `--use-autofeat` / `--no-autofeat`. SLURM cluster needs `pip install autofeat featurewiz`.

**Phase 2 â€" Meta-Model Trainer (`train_meta_model_3.py`):** Add `featurewiz_selected` and `featurewiz_importance` to the `extra` feature list in `MetaDataPreparator.prepare()`. Add sentinel value `-1.0` for rows where FeatureWiz wasn't run (same pattern as non-interaction pair features). Optionally encode `interaction_source` as a categorical feature (4 values: `default_interaction`, `tree_guided`, `importance_fallback`, `autofeat_discovered`).

**Phase 3 â€" Streamlit App (`auto_fe_app_3.py`):** Minimum: set both new fields to `-1.0` sentinel in candidate feature assembly (2-line change). Full: run FeatureWiz in the app (~5â€"15s overhead) to populate real values. AutoFeat in the app is deferred (adds too much complexity for now).

**Phase 4 â€" Pipeline System:** No changes needed. Pipeline meta-models use dataset-level features, not column-level fields. `archetype_guidance.py` is already independent.

**Phase 5 â€" Backward Compatibility:** Old v6 CSVs get NaN for new columns during merge, which the sentinel logic converts to `-1.0`. Mixed v6+v7 databases work cleanly.

**Minimum viable integration (tasks 4+5+7):** ~30 minutes. **Full integration (all 11 tasks):** ~2 hours plus SLURM run time.

---

## Plan: What's Next

### Priority 1: Scale Data Collection (Infrastructure Ready â€” Run Needed) â­

**Goal:** The meta-models are data-starved. 30 datasets / 7K rows is not enough for 80+ features. Target: 200+ datasets / 40K+ rows.

**Status:** All SLURM infrastructure is built and tested for all three systems (FE, HP, Pipeline). The pipelines are ready for large-scale runs.

| Action | Status | Expected Impact |
|--------|--------|----------------|
| Generate task list for 2000+ OpenML datasets | âœ… Ready (`generate_task_list.py`) | Covers diverse domains |
| Pre-fetch datasets into shared cache | âœ… Ready (`prefetch_data.py`) | Prevents download bottlenecks |
| Submit FE SLURM array jobs | âœ… Ready (`run_slurm.sh` + `DataCollector_slurm.py`) | Massively parallel FE collection |
| Submit HP SLURM array jobs | âœ… Ready (`run_hp_slurm.sh` + `HPCollector.py`) | Massively parallel HP collection |
| Submit Pipeline SLURM array jobs | âœ… Ready (`run_pipeline_slurm.sh` + `PipelineCollector_slurm.py`) | Massively parallel pipeline collection |
| Merge results into unified CSVs | âœ… Ready (`merge_results.py`, `merge_hp_results.py`, `merge_pipeline_results.py`) | Clean, schema-aligned databases |
| Retrain all meta-models on larger databases | âœ… Ready (`train_meta_model_3.py`, `train_hp_meta_model.py`, `train_pipeline_meta_model.py`) | Better generalization |

### Priority 2: Integrate Pipeline Meta-Model into App ⭐ (Partially Done)

**Goal:** The Pipeline system is fully built (data collection + training) and the bridging module (`archetype_guidance.py`) now connects Pipeline → FE meta-models. Remaining work is app UI integration.

| Action | Status | Expected Impact |
|--------|--------|----------------|
| Archetype guidance bridging module | ✅ Done (`archetype_guidance.py`) | Converts pipeline archetype probs → per-method FE score boosts |
| Load pipeline meta-models in app | ❌ Not started | Enable pipeline-level recommendations |
| FE Sensitivity check ("is FE worth it?") | ✅ Logic ready (in `archetype_guidance.py`) – UI not started | Saves time when FE won't help |
| Archetype recommendation UI | ❌ Not started | Quick pipeline recommendation without full scoring |
| Full pipeline scoring + ranking UI | ❌ Not started | Best pipeline selection from candidates |
| Pipeline vs individual transform comparison | ❌ Not started | Let users choose between approaches |

### Priority 2b: v7 Downstream Integration ⭐

**Goal:** Propagate DataCollector v7 changes (FeatureWiz + AutoFeat) through the downstream pipeline. See [v7 Integration Roadmap Summary](#v7-integration-roadmap-summary) for details.

| Action | Status | Expected Impact |
|--------|--------|----------------|
| Add FeatureWiz fields to meta-model trainer | ❌ Not started (task 4+5) | Meta-models learn from FeatureWiz signal |
| Populate new fields in app (sentinel) | ❌ Not started (task 7) | App compatibility with v7-trained models |
| SLURM infra for v7 (CLI flags + cluster install) | ❌ Not started (tasks 1+2+3) | Enable v7 data collection at scale |
| Run SLURM collection with v7 DataCollector | ❌ Not started (task 9) | Real FeatureWiz/AutoFeat training data |
| Retrain meta-models on v7 data | ❌ Not started (task 10) | Models learn from new features |
| (Optional) Run FeatureWiz in app | ❌ Deferred (task 8) | Real values instead of sentinel at inference |
| (Optional) Encode interaction_source | ❌ Deferred (task 6) | Extra meta-model feature |

### Priority 3: Improve Gate Model Quality

**Goal:** Gate AUC 0.58 is barely useful. Target: 0.65+.

| Action | Expected Impact |
|--------|----------------|
| More training data (Priority 1) | Single biggest factor â€” 30 datasets is too few for GroupKFold |
| Sentinel values for non-interactions (âœ… implemented) | Gate no longer confused by NaNâ†’median for single-column transforms |
| Hyperparameter tuning with Optuna | Better model configuration |
| Consider separate gate models for single-column vs. interaction transforms | Cleaner feature spaces per model |

### Priority 4: Optuna-Tuned Meta-Models

**Goal:** Current meta-model HPs are hand-picked. Automated tuning should help.

| Action | Expected Impact |
|--------|----------------|
| Optuna HPO for gate classifier | Better AUC through optimized tree depth, learning rate, regularization |
| Optuna HPO for score regressor | Lower MAE |
| Optuna HPO for ranker | Better NDCG |

### Priority 5: Advanced Features

| Feature | Description |
|---------|-------------|
| Feature selection step | RFECV or importance-based pruning as preprocessing |
| Separate single-column vs. interaction meta-models | Pro: cleaner features per model. Con: more complexity |
| Noise-floor-based scoring | Use Cohen's d and calibrated deltas (already collected) as alternative training targets |

---

## Methods Tested

| Column Type | Methods |
|-------------|---------|
| Numeric | `log_transform`, `sqrt_transform`, `quantile_binning`, `polynomial_square`, `impute_median`, `missing_indicator` |
| Temporal component | `cyclical_encode`, `impute_median`, `missing_indicator` |
| Categorical | `frequency_encoding`, `target_encoding`, `onehot_encoding`, `hashing_encoding`, `missing_indicator`, `text_stats` |
| Date (datetime) | `date_extract_basic`, `date_cyclical_month`, `date_cyclical_dow`, `date_cyclical_hour`, `date_cyclical_day`, `date_elapsed_days` |
| Interaction (numÃ—num) | `product_interaction`, `division_interaction`, `addition_interaction`, `subtraction_interaction`, `abs_diff_interaction` |
| Interaction (3-way) | `three_way_interaction`, `three_way_addition`, `three_way_ratio`, `three_way_normalized_diff` |
| Interaction (catÃ—num) | `group_mean`, `group_std` |
| Interaction (catÃ—cat) | `cat_concat` |
| Global | `row_stats` (mean, std, sum, min, max, range, + zeros/missing if applicable) |

### Transform Impact Types

| Type | Emoji | Methods | Why It Matters |
|------|-------|---------|----------------|
| **Additive** | ðŸŸ¢ | interactions, row_stats, polynomial_square, cyclical_encode, text_stats, target_encoding | Creates genuinely new information the model can't derive on its own |
| **Encoding** | ðŸ”„ | log, sqrt, frequency_encoding, onehot, hashing | Re-encodes existing info; tree models already approximate most of these |
| **Cleaning** | ðŸ§¹ | impute_median, missing_indicator, quantile_binning | Handles data quality issues |

---

## Session History

| Session | What Was Done |
|---------|--------------|
| 1-4 | Initial system build (see v2 summary) |
| 5 | Row stats, type safety, downloads |
| 6 | Diagnosis of 9 root causes, 5-phase plan |
| 7 | Phase 0 quick fixes (dedup, conditional row-stats, weaker baseline) |
| 8 | Archetype HP system, 3-model comparison, impact type tagging |
| 9 | Gate + Ranker models, adaptive recommendation count |
| 10 | ID column detection fix, gate model threshold tuning |
| 11 | Score normalization fix (std 2.3â†’14.9), data-driven gate threshold |
| 12 | Gate training rewrite (StratifiedKFold, AUC), GroupKFold, Phase 2.1/2.5 features |
| 13 | Method priors, gate NaN fix, candidate filtering, evaluation pipeline improvements |
| 14 | SLURM HPC infrastructure: task list generator, prefetch script, SLURM batch script, per-worker data collector, merge script, OFFSET batching support. DataCollector v5 with noise-floor calibration, tree-guided interactions, stratified column selection, adaptive method disabling, time budgets, and memory management. **FE Meta-Model Trainer v5**: Added Upside Model (quantile regressor for 90th percentile effect size), method family features, missingness indicators, feature interactions, continuous ranker relevance, learned ensemble weights, and richer method priors (effect_p90, effect_p10, effect_range). |
| 15 | **Hyperparameter MetaModel System**: Built complete parallel HP tuning pipeline with HPCollector.py (Latin Hypercube Sampling, 26 configs per dataset, 3Ã—5-fold CV, 30+ dataset meta-features). Implemented train_hp_meta_model.py. Added dataset archetype clustering via K-Means. Integrated with SLURM infrastructure. |
| 16 | **DataCollector v6 & Robustness**: Switched regression scoring from MSE to RÂ² (scale invariant). Fixed critical noise-floor bug (clamped null_std). Added 16 new features capturing partner column metadata (`col_b_*`, `col_c_*`) + `tree_pair_score`. Updated `train_meta_model_3.py` to use these features. Fixed `auto_fe_app_3.py` regression bug and ensured inference-time feature parity. |
| 17 | **HP Meta-Model v2 + App Integration**: Added GroupKFold to DirectHPPredictor and ranker, ensemble inference (scorer + predictor + ranker combined), HP consistency constraints. Integrated HP meta-models into `auto_fe_app_3.py` with `compute_lgbm_params_meta()` function and archetype fallback. |
| 18 | **Pipeline Meta-Model System**: Built complete pipeline-level FE evaluation system. `PipelineDataCollector.py` generates and evaluates 10-15 diverse FE pipelines per dataset from 7 archetypes. `PipelineCollector_slurm.py` + `run_pipeline_slurm.sh` for SLURM-parallel collection. `merge_pipeline_results.py` for result merging. `train_pipeline_meta_model.py` trains 5 complementary models: Pipeline Scorer, Pipeline Gate, Pipeline Ranker, FE Sensitivity Predictor, Archetype Recommender. Added `three_way_normalized_diff` to methods. |
| 19 | **DataCollector v7 (AutoFeat + FeatureWiz)**: Integrated two external libraries for enhanced column ranking and interaction discovery. **FeatureWiz** (SULOV + recursive XGBoost) runs before column selection, ranks columns by independent predictive signal, blends 20% into `composite_predictive_score`, and writes `featurewiz_selected` + `featurewiz_importance` per-row CSV fields. **AutoFeat** (polynomial L1 selection) runs after tree-guided pairs, discovers non-obvious interaction pairs fed into evaluation pipeline with `interaction_source='autofeat_discovered'`. Both optional with graceful fallback. New init flags: `use_autofeat`, `use_featurewiz`, `autofeat_max_gb`, `autofeat_feateng_steps`, `featurewiz_corr_limit`. Built `archetype_guidance.py` bridging module (Pipeline→FE meta-model integration). Created `v7_integration_roadmap.md` with 5-phase downstream update plan and 11-task checklist. |

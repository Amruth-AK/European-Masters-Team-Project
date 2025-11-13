import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# This helper function is duplicated from model_suggestion.py to keep this module
# self-contained and decoupled.
def _infer_model_family(model_name: str) -> str:
    """Heuristic to map AutoGluon model names to a coarse model family."""
    name = model_name.lower()
    if "catboost" in name:
        return "catboost"
    if "lightgbm" in name:
        return "lightgbm"
    if "xgboost" in name:
        return "xgboost"
    if "randomforest" in name or "extratrees" in name or "rf_" in name or "et_" in name:
        return "ensemble_trees"
    if "knn" in name or "kneighbors" in name:
        return "knn"
    if "nn_" in name or "neuralnet" in name:
        return "neural_network"
    if "linear" in name or "lr_" in name:
        return "linear_model"
    return "unknown"


def _get_analysis_metrics(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts key metrics from the analysis results for scoring."""
    metrics = {}
    general_info = analysis_results.get('general_info', {})
    
    # Dataset Shape
    shape = general_info.get('shape', (0, 0))
    metrics['rows'], metrics['cols'] = shape

    # Feature Type Ratios
    data_types = general_info.get('data_types', {})
    if not data_types:
        metrics['cat_ratio'] = 0.0
        metrics['num_ratio'] = 0.0
    else:
        num_count = sum(1 for dtype in data_types.values() if 'int' in dtype or 'float' in dtype)
        cat_count = sum(1 for dtype in data_types.values() if 'object' in dtype or 'category' in dtype)
        total_cols = len(data_types)
        metrics['cat_ratio'] = cat_count / total_cols if total_cols > 0 else 0
        metrics['num_ratio'] = num_count / total_cols if total_cols > 0 else 0

    # High Cardinality Check
    categorical_info = analysis_results.get('categorical_info', {})
    metrics['has_high_cardinality'] = any(
        info.get('unique_values', 0) > 50 for info in categorical_info.values()
    )

    # Missing Values Check
    missing_values = analysis_results.get('missing_values', {})
    metrics['has_missing_values'] = any(
        info.get('missing_percentage', 0) > 5 for info in missing_values.values()
    )

    # Outliers Check
    outlier_info = analysis_results.get('outlier_info', {})
    metrics['has_outliers'] = any(
        info.get('outlier_percentage', 0) > 5 for info in outlier_info.values()
    )
    
    return metrics


def _get_autogluon_ranks(leaderboard: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Gets the rank and validation score for each model family."""
    ranks = {}
    scores = {}
    
    if leaderboard is None or leaderboard.empty:
        return ranks, scores

    # Get the top entry for each family
    leaderboard['model_family'] = leaderboard['model'].apply(_infer_model_family)
    top_models_per_family = leaderboard.loc[leaderboard.groupby('model_family')['score_val'].idxmax()]
    
    # Sort by the best score to determine rank
    sorted_families = top_models_per_family.sort_values('score_val', ascending=False)
    
    for i, (_, row) in enumerate(sorted_families.iterrows()):
        family = row['model_family']
        if family != 'unknown':
            ranks[family] = i + 1
            scores[family] = row['score_val']
            
    return ranks, scores


def score_and_rank_models(analysis_results: Dict[str, Any], autogluon_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scores and ranks model families using a hybrid heuristic and performance-based approach.

    Args:
        analysis_results: The output from DataAnalyzer.
        autogluon_results: The output from run_model_suggestions.

    Returns:
        A sorted list of dictionaries, each containing the model family, score, and justification.
    """
    model_families = ['catboost', 'lightgbm', 'xgboost', 'ensemble_trees', 'linear_model', 'neural_network']
    scores = {
        family: {'score': 0, 'justification': []} for family in model_families
    }

    metrics = _get_analysis_metrics(analysis_results)

    # --- 1. Heuristic Scoring based on Data Characteristics ---

    # Rule: Feature Types
    if metrics['cat_ratio'] > 0.3:
        scores['catboost']['score'] += 25
        scores['catboost']['justification'].append("+25: Strong handling of categorical features (>30% categorical).")
        scores['lightgbm']['score'] += 15
        scores['lightgbm']['justification'].append("+15: Good support for categorical features.")
        scores['xgboost']['score'] += 15
        scores['xgboost']['justification'].append("+15: Good support for categorical features.")
        scores['linear_model']['score'] -= 10
        scores['linear_model']['justification'].append("-10: Requires manual encoding for categorical data.")
        
    if metrics['num_ratio'] > 0.7:
        scores['lightgbm']['score'] += 20
        scores['lightgbm']['justification'].append("+20: Highly efficient with numerical data (>70% numerical).")
        scores['xgboost']['score'] += 20
        scores['xgboost']['justification'].append("+20: Highly efficient with numerical data (>70% numerical).")
        scores['ensemble_trees']['score'] += 10
        scores['ensemble_trees']['justification'].append("+10: Handles numerical data well.")
        scores['linear_model']['score'] += 15
        scores['linear_model']['justification'].append("+15: Well-suited for primarily numerical data.")
        scores['neural_network']['score'] += 20
        scores['neural_network']['justification'].append("+20: Excellent with numerical data, can find complex patterns.")

    if metrics['has_high_cardinality']:
        scores['catboost']['score'] += 20
        scores['catboost']['justification'].append("+20: Best-in-class for high-cardinality categorical features.")
        scores['lightgbm']['score'] += 15
        scores['lightgbm']['justification'].append("+15: Good handling of high-cardinality features.")
        scores['linear_model']['score'] -= 20
        scores['linear_model']['justification'].append("-20: One-hot encoding high-cardinality features can lead to poor performance.")

    # Rule: Dataset Size
    if metrics['rows'] < 10000:
        scores['linear_model']['score'] += 15
        scores['linear_model']['justification'].append("+15: Performs well on smaller datasets (<10k rows).")
        scores['ensemble_trees']['score'] += 10
        scores['ensemble_trees']['justification'].append("+10: Good baseline for smaller datasets.")
        scores['neural_network']['score'] -= 15
        scores['neural_network']['justification'].append("-15: Tends to overfit on smaller datasets.")

    if metrics['rows'] > 100000:
        scores['catboost']['score'] += 20
        scores['catboost']['justification'].append("+20: Scales well to large datasets (>100k rows).")
        scores['lightgbm']['score'] += 20
        scores['lightgbm']['justification'].append("+20: Very fast and memory-efficient on large datasets.")
        scores['xgboost']['score'] += 20
        scores['xgboost']['justification'].append("+20: Scales well to large datasets.")
        scores['neural_network']['score'] += 15
        scores['neural_network']['justification'].append("+15: Benefits from large amounts of data.")

    # Rule: Data Quality
    if metrics['has_missing_values']:
        scores['catboost']['score'] += 20
        scores['catboost']['justification'].append("+20: Natively handles missing values.")
        scores['lightgbm']['score'] += 20
        scores['lightgbm']['justification'].append("+20: Natively handles missing values.")
        scores['xgboost']['score'] += 20
        scores['xgboost']['justification'].append("+20: Natively handles missing values.")
        scores['linear_model']['score'] -= 20
        scores['linear_model']['justification'].append("-20: Requires imputation for missing values.")
        scores['neural_network']['score'] -= 20
        scores['neural_network']['justification'].append("-20: Requires imputation for missing values.")

    if metrics['has_outliers']:
        scores['catboost']['score'] += 15
        scores['catboost']['justification'].append("+15: Tree-based models are robust to outliers.")
        scores['lightgbm']['score'] += 15
        scores['lightgbm']['justification'].append("+15: Tree-based models are robust to outliers.")
        scores['xgboost']['score'] += 15
        scores['xgboost']['justification'].append("+15: Tree-based models are robust to outliers.")
        scores['ensemble_trees']['score'] += 15
        scores['ensemble_trees']['justification'].append("+15: Tree-based models are robust to outliers.")
        scores['linear_model']['score'] -= 20
        scores['linear_model']['justification'].append("-20: Sensitive to outliers, requires handling.")
        scores['neural_network']['score'] -= 15
        scores['neural_network']['justification'].append("-15: Sensitive to outliers.")

    # --- 2. AutoGluon Performance Scoring ---
    leaderboard = autogluon_results.get('leaderboard')
    ranks, ag_scores = _get_autogluon_ranks(leaderboard)
    
    performance_bonuses = {1: 50, 2: 30, 3: 15}
    for family, rank in ranks.items():
        if family in scores and rank in performance_bonuses:
            bonus = performance_bonuses[rank]
            scores[family]['score'] += bonus
            scores[family]['justification'].append(f"+{bonus}: Ranked #{rank} in the initial performance test.")
            
    # --- 3. Finalization and Normalization ---
    final_ranking = []
    for family, data in scores.items():
        entry = {
            'model_family': family,
            'score': data['score'],
            'justification': data['justification'],
            'autogluon_rank': ranks.get(family, 'N/A'),
            'autogluon_score': f"{ag_scores.get(family, 0):.4f}" if ranks.get(family) else "N/A"
        }
        final_ranking.append(entry)

    # Normalize scores to 0-100 range for better presentation
    all_scores = [entry['score'] for entry in final_ranking]
    min_score, max_score = min(all_scores), max(all_scores)
    
    if max_score == min_score: # Avoid division by zero
        for entry in final_ranking:
            entry['normalized_score'] = 50.0
    else:
        for entry in final_ranking:
            normalized = 100 * (entry['score'] - min_score) / (max_score - min_score)
            entry['normalized_score'] = round(normalized)

    # Sort by the final normalized score
    final_ranking.sort(key=lambda x: x['normalized_score'], reverse=True)
    
    return final_ranking
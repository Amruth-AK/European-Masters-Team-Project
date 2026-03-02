"""
Archetype-Guided FE Recommendation Module
==========================================

Bridges the Pipeline Meta-Model and the FE Meta-Model.

Flow:
  1. User uploads dataset → compute ds_meta (already done in app)
  2. Pipeline sensitivity predictor: "Is FE worth trying?" → predicted max delta
  3. Pipeline archetype recommender: "What strategy?" → archetype probabilities
  4. Convert archetype probs → per-method score boosts for FE recommendation
  5. FE meta-model uses these boosts when scoring candidates

Implements SOFT weighting: all candidate types are still generated,
but archetype probabilities control score boosts. No hard filtering.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Optional, Any


# =============================================================================
# ARCHETYPE → METHOD MAPPING
# =============================================================================

# Which FE methods are characteristic of each pipeline archetype.
# Weights (0-1) indicate how strongly a method is associated with the archetype.
# Derived from: what transforms each archetype generator in PipelineDataCollector
# actually produces, plus empirical method priors from FE training.

ARCHETYPE_METHOD_WEIGHTS = {
    'interaction_heavy': {
        # Primary: 2-way interactions
        'product_interaction': 1.0,
        'division_interaction': 1.0,
        'addition_interaction': 1.0,
        'subtraction_interaction': 1.0,
        'abs_diff_interaction': 1.0,
        # Secondary: group-by and 3-way
        'group_mean': 0.8,
        'group_std': 0.7,
        'three_way_interaction': 0.6,
        'three_way_addition': 0.6,
        'three_way_ratio': 0.6,
        'three_way_normalized_diff': 0.6,
        # Row stats often pair well with interactions
        'row_stats': 0.3,
    },
    'encoding_focused': {
        'frequency_encoding': 1.0,
        'target_encoding': 1.0,
        'onehot_encoding': 0.9,
        'hashing_encoding': 0.8,
        'text_stats': 0.7,
        'missing_indicator': 0.5,
    },
    'single_col_heavy': {
        'log_transform': 1.0,
        'sqrt_transform': 1.0,
        'polynomial_square': 0.9,
        'quantile_binning': 0.7,
        'cyclical_encode': 0.6,
        'missing_indicator': 0.5,
        'polynomial_cube': 0.7,       # NEW: strong single-col transform
        'abs_transform': 0.6,         # NEW
        'exp_transform': 0.5,         # NEW
        'reciprocal_transform': 0.5,  # NEW
    },
    'balanced_mix': {
        'product_interaction': 0.5,
        'division_interaction': 0.5,
        'addition_interaction': 0.5,
        'subtraction_interaction': 0.5,
        'group_mean': 0.5,
        'frequency_encoding': 0.5,
        'target_encoding': 0.5,
        'log_transform': 0.4,
        'sqrt_transform': 0.4,
        'polynomial_square': 0.4,
        'row_stats': 0.4,
        'polynomial_cube': 0.3,
        'abs_transform': 0.3,
        'exp_transform': 0.3,
        'reciprocal_transform': 0.3,
    },
    'kitchen_sink': {
        'product_interaction': 0.6,
        'division_interaction': 0.6,
        'addition_interaction': 0.6,
        'group_mean': 0.6,
        'group_std': 0.5,
        'frequency_encoding': 0.6,
        'target_encoding': 0.6,
        'log_transform': 0.5,
        'polynomial_square': 0.5,
        'row_stats': 0.5,
        'missing_indicator': 0.4,
        'polynomial_cube': 0.4,
        'abs_transform': 0.4,
        'exp_transform': 0.3,
        'reciprocal_transform': 0.3,
    },
    'minimal_surgical': {
        # Minimal: only the most proven methods get a small bump
        'group_mean': 0.6,
        'frequency_encoding': 0.5,
        'target_encoding': 0.5,
        'missing_indicator': 0.4,
        'log_transform': 0.3,
    },
    'row_stats_plus': {
        'row_stats': 1.0,
        'product_interaction': 0.4,
        'division_interaction': 0.4,
        'group_mean': 0.4,
        'polynomial_square': 0.3,
    },
}

# All methods the FE system knows about
ALL_FE_METHODS = [
    'log_transform', 'sqrt_transform', 'quantile_binning', 'polynomial_square',
    'polynomial_cube', 'abs_transform', 'exp_transform', 'reciprocal_transform', 
    'impute_median', 'missing_indicator', 'cyclical_encode',
    'frequency_encoding', 'target_encoding', 'onehot_encoding',
    'hashing_encoding', 'text_stats',
    'product_interaction', 'division_interaction', 'addition_interaction',
    'subtraction_interaction', 'abs_diff_interaction',
    'three_way_interaction', 'three_way_addition', 'three_way_ratio',
    'three_way_normalized_diff',
    'group_mean', 'group_std', 'cat_concat',
    'row_stats',
]


# =============================================================================
# CORE: ARCHETYPE GUIDANCE COMPUTER
# =============================================================================

class ArchetypeGuidance:
    """
    Computes archetype-guided score boosts for FE candidates.
    
    Uses:
      - Pipeline archetype recommender (predict_proba → 7 archetype probs)
      - Pipeline sensitivity predictor (predict → expected max delta)
      - Archetype profiles (empirical stats per archetype)
    
    Produces:
      - Per-method score boosts (dict: method_name → float)
      - FE sensitivity estimate (is FE worth trying?)
      - Explanation dict for UI display
    """
    
    def __init__(self, pipeline_preparator, sensitivity_predictor,
                 archetype_recommender, archetype_profiles=None):
        self.preparator = pipeline_preparator
        self.sensitivity = sensitivity_predictor
        self.archetype_rec = archetype_recommender
        self.archetype_profiles = archetype_profiles or {}
        
        # Get archetype class names from the preparator's encoder
        self.archetype_names = list(pipeline_preparator.archetype_encoder.classes_)
    
    def compute_guidance(self, ds_meta: dict, task_type: str,
                         max_boost: float = 3.0,
                         sensitivity_gate: float = 0.001) -> Dict[str, Any]:
        """
        Main entry point: compute archetype-guided boosts for FE candidates.
        
        Args:
            ds_meta: dataset meta-features dict (from FeatureComputer)
            task_type: 'classification' or 'regression'
            max_boost: maximum score boost any method can receive
            sensitivity_gate: if predicted max delta < this, FE barely helps
        
        Returns dict with:
            'method_boosts': {method_name: float} - per-method score boosts
            'archetype_probs': {archetype_name: float} - predicted probabilities  
            'top_archetype': str - highest-probability archetype
            'fe_sensitivity': float - predicted max delta
            'fe_worth_trying': bool - whether FE is predicted to help
            'explanation': dict - human-readable explanation for UI
        """
        # Build feature row for pipeline models
        ds_row = self._build_dataset_row(ds_meta, task_type)
        X_ds = self.preparator.transform_dataset(ds_row)
        
        # --- FE Sensitivity ---
        fe_sensitivity = float(self.sensitivity.predict(X_ds)[0])
        fe_worth_trying = fe_sensitivity > sensitivity_gate
        
        # --- Archetype Probabilities ---
        archetype_probs_raw = self.archetype_rec.predict_proba(X_ds)[0]
        archetype_probs = {
            name: float(prob) 
            for name, prob in zip(self.archetype_names, archetype_probs_raw)
        }
        top_archetype = max(archetype_probs, key=archetype_probs.get)
        top_prob = archetype_probs[top_archetype]
        
        # --- Convert to Method Boosts ---
        method_boosts = self._compute_method_boosts(
            archetype_probs, fe_sensitivity, max_boost)
        
        # --- Build Explanation ---
        explanation = self._build_explanation(
            archetype_probs, top_archetype, top_prob,
            fe_sensitivity, fe_worth_trying, method_boosts)
        
        return {
            'method_boosts': method_boosts,
            'archetype_probs': archetype_probs,
            'top_archetype': top_archetype,
            'top_prob': top_prob,
            'fe_sensitivity': fe_sensitivity,
            'fe_worth_trying': fe_worth_trying,
            'explanation': explanation,
        }
    
    def _build_dataset_row(self, ds_meta: dict, task_type: str) -> pd.DataFrame:
        """Convert ds_meta dict to DataFrame suitable for pipeline preparator."""
        row = dict(ds_meta)
        try:
            row['task_type_encoded'] = int(
                self.preparator.task_type_encoder.transform([task_type])[0])
        except (ValueError, KeyError):
            row['task_type_encoded'] = 0 if task_type == 'classification' else 1
        return pd.DataFrame([row])
    
    def _compute_method_boosts(self, archetype_probs: dict,
                                fe_sensitivity: float,
                                max_boost: float) -> dict:
        """
        Convert archetype probabilities to per-method score boosts.
        
        For each FE method:
          boost = Σ (archetype_prob × method_weight) × scale_factor
        
        scale_factor depends on:
          - Prediction confidence (low entropy = confident = bigger boosts)
          - Dataset FE sensitivity (more sensitive = bigger boosts)
        """
        # Confidence from entropy of archetype distribution
        probs = np.array(list(archetype_probs.values()))
        probs_safe = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs_safe * np.log(probs_safe))
        max_entropy = np.log(len(probs))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Sensitivity scaling: typical values 0.001 - 0.05
        # Map to [0.2, 1.0] so even low-sensitivity datasets get some guidance
        sensitivity_scale = np.clip(fe_sensitivity * 20, 0.2, 1.0)
        
        # Combined scale factor
        scale = max_boost * confidence * sensitivity_scale
        
        # Compute per-method boosts
        method_boosts = {}
        for method in ALL_FE_METHODS:
            boost = 0.0
            for archetype, prob in archetype_probs.items():
                weight = ARCHETYPE_METHOD_WEIGHTS.get(archetype, {}).get(method, 0.0)
                boost += prob * weight
            method_boosts[method] = float(np.clip(boost * scale, -max_boost, max_boost))
        
        return method_boosts
    
    def _build_explanation(self, archetype_probs, top_archetype, top_prob,
                           fe_sensitivity, fe_worth_trying, method_boosts) -> dict:
        """Build human-readable explanation for the UI."""
        sorted_archs = sorted(archetype_probs.items(), key=lambda x: x[1], reverse=True)
        
        sorted_boosts = sorted(method_boosts.items(), key=lambda x: x[1], reverse=True)
        top_boosted = [(m, b) for m, b in sorted_boosts if b > 0.1][:5]
        
        archetype_descriptions = {
            'interaction_heavy': 'Focus on feature interactions (A×B, A/B, group-by)',
            'encoding_focused': 'Focus on categorical encoding strategies',
            'single_col_heavy': 'Focus on single-column transforms (log, sqrt, etc.)',
            'balanced_mix': 'Balanced approach across transform types',
            'kitchen_sink': 'Comprehensive — try many transform types together',
            'minimal_surgical': 'Minimal — only a few high-impact transforms',
            'row_stats_plus': 'Row-level statistics as primary strategy',
        }
        
        archetype_emoji = {
            'interaction_heavy': '🔗',
            'encoding_focused': '🏷️',
            'single_col_heavy': '📊',
            'balanced_mix': '⚖️',
            'kitchen_sink': '🍳',
            'minimal_surgical': '🎯',
            'row_stats_plus': '📈',
        }
        
        profile = self.archetype_profiles.get(top_archetype, {})
        
        return {
            'top_archetype': top_archetype,
            'top_archetype_prob': top_prob,
            'top_archetype_desc': archetype_descriptions.get(top_archetype, ''),
            'top_archetype_emoji': archetype_emoji.get(top_archetype, '🔧'),
            'archetype_ranking': sorted_archs,
            'archetype_descriptions': archetype_descriptions,
            'archetype_emoji': archetype_emoji,
            'fe_sensitivity': fe_sensitivity,
            'fe_worth_trying': fe_worth_trying,
            'top_boosted_methods': top_boosted,
            'archetype_improvement_rate': profile.get('improvement_rate'),
            'archetype_avg_delta': profile.get('avg_delta'),
            'confidence_note': (
                'High confidence'
                if top_prob > 0.4
                else 'Moderate confidence'
                if top_prob > 0.25
                else 'Low confidence — multiple strategies may work'
            ),
        }


# =============================================================================
# LOADING HELPER
# =============================================================================

def load_pipeline_guidance_models(pipeline_model_dir: str) -> Optional[ArchetypeGuidance]:
    """
    Load pipeline meta-models needed for archetype guidance.
    
    Returns ArchetypeGuidance instance, or None if models not available.
    """
    try:
        from Pipeline_MetaModel.train_pipeline_meta_model import (
            PipelineDataPreparator, FESensitivityPredictor, ArchetypeRecommender
        )
    except ImportError:
        print("WARNING: Could not import train_pipeline_meta_model. "
              "Pipeline guidance unavailable.")
        return None
    
    required_files = [
        'pipeline_preparator.pkl',
        'pipeline_sensitivity.pkl',
        'pipeline_archetype_rec.pkl',
    ]
    
    for fname in required_files:
        if not os.path.exists(os.path.join(pipeline_model_dir, fname)):
            print(f"Pipeline guidance: missing {fname}")
            return None
    
    try:
        preparator = PipelineDataPreparator()
        preparator.load(os.path.join(pipeline_model_dir, 'pipeline_preparator.pkl'))
        
        sensitivity = FESensitivityPredictor()
        sensitivity.load(os.path.join(pipeline_model_dir, 'pipeline_sensitivity.pkl'))
        
        archetype_rec = ArchetypeRecommender()
        archetype_rec.load(os.path.join(pipeline_model_dir, 'pipeline_archetype_rec.pkl'))
        
        profiles = {}
        profiles_path = os.path.join(pipeline_model_dir, 'pipeline_archetype_profiles.json')
        if os.path.exists(profiles_path):
            with open(profiles_path, 'r') as f:
                profiles = json.load(f)
        
        guidance = ArchetypeGuidance(
            pipeline_preparator=preparator,
            sensitivity_predictor=sensitivity,
            archetype_recommender=archetype_rec,
            archetype_profiles=profiles,
        )
        print(f"✓ Pipeline archetype guidance loaded "
              f"({len(guidance.archetype_names)} archetypes)")
        return guidance
    
    except Exception as e:
        print(f"WARNING: Failed to load pipeline guidance: {e}")
        return None
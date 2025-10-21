"""
Main Entry Point - Team Project
Integrates all modules: Analysis → Suggestions → Preprocessing

Workflow:
1. Load dataset and specify target variable
2. Analyze dataset (analyze.py)
3. Generate preprocessing suggestions (preprocessing_suggestions.py)
4. Present suggestions to user
5. Execute approved transformations (preprocessing_function.py)
6. Suggest ML models (future)
"""

import sys
from pathlib import Path
import pandas as pd

# Add preprocessing directory to path
preprocessing_dir = Path(__file__).parent / "Preprocessing suggestion"
sys.path.insert(0, str(preprocessing_dir))


def load_dataset(filepath, target_column):
    """
    Load dataset and separate target variable.
    
    Args:
        filepath (str): Path to CSV file
        target_column (str): Name of target variable
    
    Returns:
        tuple: (DataFrame, target_series)
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if target_column not in df.columns:
            print(f"Warning: Target column '{target_column}' not found in dataset")
            return df, None
        
        return df, df[target_column]
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def analyze_dataset(df, target_column):
    """
    Analyze dataset and generate diagnostic report.
    
    TODO: This will be replaced by Adrian & Alisa's analyze.py
    For now, we use mock data for testing.
    
    Args:
        df (DataFrame): Input dataset
        target_column (str): Target variable name
    
    Returns:
        dict: Analysis report
    """
    print("\n" + "="*80)
    print("STEP 1: ANALYZING DATASET")
    print("="*80)
    
    # For now, use mock data
    # TODO: Replace with: from analyze import analyze_dataset
    from mock_analysis_output import get_mock_analysis_report
    
    analysis_report = get_mock_analysis_report()
    
    print("✓ Analysis complete")
    print(f"  - Features analyzed: {len(analysis_report.get('feature_types', {}))}")
    print(f"  - Missing values detected: {len(analysis_report.get('missing_values', {}))}")
    print(f"  - Outliers detected: {sum(1 for v in analysis_report.get('outliers', {}).values() if v.get('has_outliers'))}")
    
    return analysis_report


def generate_preprocessing_suggestions(analysis_report):
    """
    Generate preprocessing suggestions based on analysis.
    
    Args:
        analysis_report (dict): Analysis diagnostics
    
    Returns:
        list: Preprocessing suggestions
    """
    print("\n" + "="*80)
    print("STEP 2: GENERATING PREPROCESSING SUGGESTIONS")
    print("="*80)
    
    from preprocessing_suggestions import generate_suggestions
    
    suggestions = generate_suggestions(analysis_report)
    
    print(f"✓ Generated {len(suggestions)} preprocessing suggestions")
    
    return suggestions


def present_suggestions(suggestions):
    """
    Present suggestions to user in a readable format.
    
    Args:
        suggestions (list): List of suggestion dictionaries
    """
    from preprocessing_suggestions import print_suggestions
    print_suggestions(suggestions)


def execute_preprocessing(df, suggestions, auto_apply=False):
    """
    Execute preprocessing transformations based on user confirmation.
    
    Args:
        df (DataFrame): Input dataset
        suggestions (list): Preprocessing suggestions
        auto_apply (bool): If True, apply all without asking
    
    Returns:
        DataFrame: Preprocessed dataset
    """
    print("\n" + "="*80)
    print("STEP 3: EXECUTING PREPROCESSING")
    print("="*80)
    
    # Import preprocessing functions
    try:
        from preprocessing_function import (
            standard_scaler, minmax_scaler, 
            clip_outliers_iqr, remove_outliers_iqr
        )
        
        # Map function names to actual functions
        function_map = {
            'standard_scaler': standard_scaler,
            'minmax_scaler': minmax_scaler,
            'clip_outliers_iqr': clip_outliers_iqr,
            'remove_outliers_iqr': remove_outliers_iqr,
            # TODO: Add more functions as they're implemented
            # 'impute_median': impute_median,
            # 'impute_mode': impute_mode,
            # 'encode_one_hot': encode_one_hot,
            # etc.
        }
        
    except ImportError as e:
        print(f"Warning: Could not import preprocessing functions: {e}")
        print("Preprocessing execution skipped.")
        return df
    
    df_processed = df.copy()
    applied_count = 0
    skipped_count = 0
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n[{i}/{len(suggestions)}] Feature: {suggestion['feature']}")
        print(f"Issue: {suggestion['issue']}")
        print(f"Suggestion: {suggestion['suggestion']}")
        print(f"Function: {suggestion['function_to_call']}")
        
        # Get user confirmation
        if auto_apply:
            apply = True
            print("→ Auto-applying...")
        else:
            response = input("Apply this transformation? (yes/no/quit): ").lower()
            
            if response == 'quit':
                print("Preprocessing stopped by user.")
                break
            
            apply = response in ['yes', 'y']
        
        if apply:
            func_name = suggestion['function_to_call']
            
            # Check if function is available
            if func_name not in function_map:
                print(f"✗ Function '{func_name}' not yet implemented. Skipping.")
                skipped_count += 1
                continue
            
            try:
                # Execute the transformation
                func = function_map[func_name]
                df_processed = func(df_processed, **suggestion['kwargs'])
                print("✓ Applied successfully")
                applied_count += 1
                
            except Exception as e:
                print(f"✗ Error applying transformation: {e}")
                skipped_count += 1
        else:
            print("○ Skipped")
            skipped_count += 1
    
    print("\n" + "="*80)
    print(f"Preprocessing complete: {applied_count} applied, {skipped_count} skipped")
    print("="*80)
    
    return df_processed


def save_preprocessed_data(df, output_path):
    """Save preprocessed dataset to file."""
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✓ Preprocessed data saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Error saving data: {e}")


def main(dataset_path=None, target_column=None, auto_apply=False, output_path=None):
    """
    Main workflow execution.
    
    Args:
        dataset_path (str): Path to input CSV
        target_column (str): Target variable name
        auto_apply (bool): Auto-apply all suggestions
        output_path (str): Path to save preprocessed data
    """
    print("="*80)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print("="*80)
    
    # For demo/testing purposes, use mock data if no path provided
    if dataset_path is None:
        print("\nNo dataset provided. Running in DEMO MODE with mock analysis.")
        print("Usage: python main.py <dataset_path> <target_column>\n")
        
        # Use mock analysis
        from mock_analysis_output import get_mock_analysis_report
        analysis_report = get_mock_analysis_report()
        
        # Generate suggestions
        suggestions = generate_preprocessing_suggestions(analysis_report)
        
        # Present suggestions
        present_suggestions(suggestions)
        
        print("\n" + "="*80)
        print("DEMO MODE: To run with actual data, provide dataset path:")
        print("  python main.py data/titanic.csv Survived")
        print("="*80)
        
        return
    
    # Step 1: Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    print(f"Target variable: {target_column}")
    
    df, target = load_dataset(dataset_path, target_column)
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Step 2: Analyze dataset
    analysis_report = analyze_dataset(df, target_column)
    
    # Step 3: Generate suggestions
    suggestions = generate_preprocessing_suggestions(analysis_report)
    
    # Step 4: Present suggestions
    present_suggestions(suggestions)
    
    # Step 5: Execute preprocessing
    if not suggestions:
        print("\nNo preprocessing needed!")
        return
    
    print("\n" + "="*80)
    proceed = input("Proceed with preprocessing? (yes/no): ").lower()
    
    if proceed not in ['yes', 'y']:
        print("Preprocessing cancelled.")
        return
    
    df_processed = execute_preprocessing(df, suggestions, auto_apply=auto_apply)
    
    # Step 6: Save results
    if output_path:
        save_preprocessed_data(df_processed, output_path)
    else:
        output_path = dataset_path.replace('.csv', '_preprocessed.csv')
        save_preprocessed_data(df_processed, output_path)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {df_processed.shape}")


if __name__ == "__main__":
    # Command line interface
    if len(sys.argv) == 1:
        # No arguments - run demo mode
        main()
    elif len(sys.argv) >= 3:
        # With arguments
        dataset_path = sys.argv[1]
        target_column = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        
        main(dataset_path, target_column, output_path=output_path)
    else:
        print("Usage: python main.py [dataset_path] [target_column] [output_path]")
        print("   Or: python main.py  (for demo mode)")


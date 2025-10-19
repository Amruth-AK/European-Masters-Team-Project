# Preprocessing_suggestion
- **Input**: A dictionary of diagnostic results from the Analysis Team (the "Input Contract").
- **Process**: Applies a series of `IF-THEN` rules to the diagnostics.
- **Output**: A list of suggestion dictionaries (the "Output Contract").


# Preprocessing_function
- Contains a collection of standalone, well-documented Python functions, each performing a single, specific preprocessing task (e.g., `impute_median`, `encode_one_hot`).

# Main
- Receives the diagnostics.
- Calls the **Rule Engine** to get suggestions.
- Iterates through the suggestions, printing them to the user.
- Prompts the user for a `yes/no` confirmation for each suggestion.
- If confirmed, it calls the appropriate function from the **Function Library** to execute the transformation on the dataset.

# Workflow

```python
# Input
analysis_results = {
    'feature_types': {'Age': 'numerical', 'Sex': 'categorical', 'Cabin': 'categorical'},
    'missing_values_rate': {'Age': 0.19, 'Cabin': 0.77},
    'cardinality': {'Cabin': 147, 'Sex': 2},
    'skewness': {'Age': 0.38, 'Fare': 47.2}
}


# Output
preprocessing_suggestions = [
    {
        'feature': 'Age',
        'issue': 'Missing Values (19%)',
        'suggestion': 'Recommend using median imputation as the data is skewed.',
        'function_to_call': 'impute_median', # CRITICAL: This links suggestion to action
        'kwargs': {'column': 'Age'} # Arguments for the function
    },
    {
        'feature': 'Cabin',
        'issue': 'High Cardinality (147 unique values)',
        'suggestion': 'Recommend using Target Encoding to avoid dimensionality explosion.',
        'function_to_call': 'encode_target',
        'kwargs': {'column': 'Cabin', 'target': 'Survived'}
    }
]
```
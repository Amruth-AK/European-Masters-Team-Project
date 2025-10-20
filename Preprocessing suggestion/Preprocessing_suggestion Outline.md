# Preprocessing_suggestion-3
- **Input**: A dictionary of diagnostic results from the Analysis Team (the "Input Contract").
- **Process**: Applies a series of `IF-THEN` rules to the diagnostics.
- **Output**: A list of suggestion dictionaries (the "Output Contract").




# Preprocessing_function-3
- Contains a collection of standalone, well-documented Python functions, each performing a single, specific preprocessing task (e.g., `impute_median`, `encode_one_hot`).


```python
def impute_median(df,column):
	-----
	return df
	
def encode_one_hot(df,column):
	return df	
	
```

1. Missing Value-**Tecla**
	1. Deletion
	2. Median/Mean/Mode/(KNN-predict) Imputation
2. Duplicate Values-**Telca**
	1. Deletion
3. Categorical Features-**Amruth**
	1. One-hot encoding
	2. Label Encoding
	3. Target Encoding
	4. .....
4. Numerical Features-**Zhiqi**
	1. `standardscaler`
	2. `minmascaler`
5. Outlier
	1. `Clip_outliers_iqr(df, column, whisker_width=1.5)`: The cap method/limit method replaces the outliers of the range with boundary values.
	2. `Remove_outliers_iqr(df, column)`: Directly delete rows containing outliers
6. 

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
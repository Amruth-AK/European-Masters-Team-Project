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

## Numerical Features Preprocessing
### 1. `standard_scaler(df, column)`
- **Standard Scaling (Z-score Normalization)**
- **Formula**: `z = (x - mean) / std`
- **Use Case**: Best for normally distributed data
- **Output Range**: Typically between -3 and +3 (for normal distribution)
- **Handles**: Single column or list of columns

```python

df = standard_scaler(df, 'Age')

df = standard_scaler(df, ['Age', 'Fare'])

```

  
#### **When to use:**
- Data is approximately normally distributed
- Algorithm requires standardized features (e.g., SVM, Neural Networks, PCA)
- Features have different units/scales

  

### 2. `minmax_scaler(df, column, feature_range=(0, 1))`
- **Min-Max Scaling**
- **Formula**: `x_scaled = (x - min) / (max - min) * (max_range - min_range) + min_range`
- **Use Case**: Best when you need bounded values in a specific range
- **Output Range**: Default (0, 1), customizable
- **Handles**: Single column or list of columns

```python

df = minmax_scaler(df, 'Age')

df = minmax_scaler(df, 'Fare', feature_range=(0, 1))

df = minmax_scaler(df, ['Age', 'Fare'], feature_range=(-1, 1))

```

#### **When to use:**
- Need features in a specific range (e.g., [0,1] for neural networks
- Data doesn't have outliers
- Want to preserve the original distribution shape


## Outlier

### 1. `clip_outliers_iqr(df, column, whisker_width=1.5)`
- **Capping Method** - Replaces outliers with boundary values
```
Lower bound = Q1 - whisker_width × IQR
Upper bound = Q3 + whisker_width × IQR
```
  - **Behavior:**
	- Values below lower bound → set to lower bound
	- Values above upper bound → set to upper bound
- **Preserves all rows** in the dataset

```python

# Original values: [10, 12, 11, 100, 9, 200, 11]

df = clip_outliers_iqr(df, 'Fare', whisker_width=1.5)

# Result: [10, 12, 11, upper_bound, 9, upper_bound, 11]

# All rows kept, extreme values capped

```

#### **When to use:**
-  Want to preserve dataset size
- Outliers might contain useful information
- Small datasets where losing rows is costly
- Need to maintain row relationships

### 2. `remove_outliers_iqr(df, column, whisker_width=1.5)`
- **Deletion Method** - Removes rows containing outliers
- **Detection:**
```

Lower bound = Q1 - whisker_width × IQR

Upper bound = Q3 + whisker_width × IQR

```
- **Behavior:**
	- Rows with values < lower bound → deleted
	- Rows with values > upper bound → deleted
- **Reduces dataset size**

```python

# Original: 15 rows

df = remove_outliers_iqr(df, 'Fare', whisker_width=1.5)

# Result: 13 rows (2 outlier rows removed)

```
 

#### **When to use:**

-  Large dataset (can afford to lose rows)
- Outliers are measurement errors
- Want clean statistical properties
- Avoid if dataset is small

  

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




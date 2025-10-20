# Numerical Features Preprocessing - Zhiqi Yang

## Implemented Functions

### 1. `standard_scaler(df, column)`
**Standard Scaling (Z-score Normalization)**

- **Formula**: `z = (x - mean) / std`
- **Use Case**: Best for normally distributed data
- **Output Range**: Typically between -3 and +3 (for normal distribution)
- **Handles**: Single column or list of columns

**Example:**
```python
df = standard_scaler(df, 'Age')
df = standard_scaler(df, ['Age', 'Fare'])
```

**When to use:**
- Data is approximately normally distributed
- Algorithm requires standardized features (e.g., SVM, Neural Networks, PCA)
- Features have different units/scales

---

### 2. `minmax_scaler(df, column, feature_range=(0, 1))`
**Min-Max Scaling**

- **Formula**: `x_scaled = (x - min) / (max - min) * (max_range - min_range) + min_range`
- **Use Case**: Best when you need bounded values in a specific range
- **Output Range**: Default (0, 1), customizable
- **Handles**: Single column or list of columns

**Example:**
```python
df = minmax_scaler(df, 'Age')
df = minmax_scaler(df, 'Fare', feature_range=(0, 1))
df = minmax_scaler(df, ['Age', 'Fare'], feature_range=(-1, 1))
```

**When to use:**
- Need features in a specific range (e.g., [0,1] for neural networks)
- Data doesn't have outliers
- Want to preserve the original distribution shape

---

### 3. `robust_scaler(df, column)` [BONUS]
**Robust Scaling (IQR-based)**

- **Formula**: `x_scaled = (x - median) / IQR`
- **Use Case**: Best when data has outliers
- **Output Range**: Unbounded
- **Handles**: Single column or list of columns

**Example:**
```python
df = robust_scaler(df, 'Fare')
```

**When to use:**
- Data contains outliers
- Want scaling robust to extreme values
- Skewed distributions

---

## Integration with Preprocessing Suggestion Workflow

### Input from Analysis Team
```python
analysis_results = {
    'feature_types': {'Age': 'numerical', 'Fare': 'numerical'},
    'skewness': {'Age': 0.38, 'Fare': 4.72},
    'has_outliers': {'Fare': True}
}
```

### Output Suggestions
```python
preprocessing_suggestions = [
    {
        'feature': 'Age',
        'issue': 'Numerical feature needs scaling',
        'suggestion': 'Apply standard scaling for normally distributed data.',
        'function_to_call': 'standard_scaler',
        'kwargs': {'column': 'Age'}
    },
    {
        'feature': 'Fare',
        'issue': 'High skewness (4.72) and outliers detected',
        'suggestion': 'Apply robust scaling to handle outliers.',
        'function_to_call': 'robust_scaler',
        'kwargs': {'column': 'Fare'}
    }
]
```

---

## Decision Tree for Choosing Scaler

```
Does the feature have outliers?
├── YES → Use robust_scaler
└── NO
    ├── Need bounded range [0,1]? → Use minmax_scaler
    └── Data normally distributed? → Use standard_scaler
```

---

## Features

✅ **Handles NaN values**: All scalers ignore NaN when computing statistics  
✅ **Multiple columns**: Can scale single or multiple columns at once  
✅ **Error handling**: Validates column existence and data types  
✅ **Edge cases**: Handles constant columns (zero variance)  
✅ **Type hints**: Full type annotations for better IDE support  
✅ **Documentation**: Comprehensive docstrings with examples  

---

## Dependencies
```python
import pandas as pd
import numpy as np
```

---

## Testing

To run tests:
```bash
cd "Preprocessing suggestion"
python3 test_numerical_features.py
```

Make sure to install dependencies first:
```bash
pip install pandas numpy
```

---

## Notes for Team

- All functions return a **copy** of the DataFrame (non-destructive)
- Functions are **stateless** - no fitting/transforming separation needed for simplicity
- Compatible with the suggestion workflow output format
- Can be easily integrated into the main preprocessing pipeline

---

**Author**: Zhiqi Yang  
**Date**: October 2025  
**Branch**: zhiqi-yang


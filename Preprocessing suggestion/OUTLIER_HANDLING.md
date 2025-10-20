# Outlier Handling Functions - Zhiqi Yang

## Overview
Two methods for handling outliers using the IQR (Interquartile Range) method.

---

## Functions

### 1. `clip_outliers_iqr(df, column, whisker_width=1.5)`

**Capping Method** - Replaces outliers with boundary values

**Detection:**
```
Lower bound = Q1 - whisker_width × IQR
Upper bound = Q3 + whisker_width × IQR
```

**Behavior:**
- Values below lower bound → set to lower bound
- Values above upper bound → set to upper bound
- **Preserves all rows** in the dataset

**Example:**
```python
# Original values: [10, 12, 11, 100, 9, 200, 11]
df = clip_outliers_iqr(df, 'Fare', whisker_width=1.5)
# Result: [10, 12, 11, upper_bound, 9, upper_bound, 11]
# All rows kept, extreme values capped
```

**When to use:**
- ✅ Want to preserve dataset size
- ✅ Outliers might contain useful information
- ✅ Small datasets where losing rows is costly
- ✅ Need to maintain row relationships

---

### 2. `remove_outliers_iqr(df, column, whisker_width=1.5)`

**Deletion Method** - Removes rows containing outliers

**Detection:**
```
Lower bound = Q1 - whisker_width × IQR
Upper bound = Q3 + whisker_width × IQR
```

**Behavior:**
- Rows with values < lower bound → deleted
- Rows with values > upper bound → deleted
- **Reduces dataset size**

**Example:**
```python
# Original: 15 rows
df = remove_outliers_iqr(df, 'Fare', whisker_width=1.5)
# Result: 13 rows (2 outlier rows removed)
```

**When to use:**
- ✅ Large dataset (can afford to lose rows)
- ✅ Outliers are measurement errors
- ✅ Want clean statistical properties
- ❌ Avoid if dataset is small

---

## Parameters

### `whisker_width` (default=1.5)

Controls outlier sensitivity:

| Value | Sensitivity | Use Case |
|-------|-------------|----------|
| 1.5 | Standard | Default box plot definition |
| 2.0 | Moderate | Less aggressive |
| 3.0 | Conservative | Only extreme outliers |

**Example:**
```python
# Aggressive (catches more outliers)
df = clip_outliers_iqr(df, 'Fare', whisker_width=1.5)

# Conservative (catches fewer outliers)
df = clip_outliers_iqr(df, 'Fare', whisker_width=3.0)
```

---

## Workflow Integration

### Input from Analysis
```python
analysis_results = {
    'has_outliers': {'Fare': True, 'Age': False},
    'outlier_count': {'Fare': 23},
    'outlier_percentage': {'Fare': 0.026}
}
```

### Generated Suggestions
```python
suggestions = [
    {
        'feature': 'Fare',
        'issue': 'Contains 23 outliers (2.6% of data)',
        'suggestion': 'Clip outliers to preserve data while reducing extreme values.',
        'function_to_call': 'clip_outliers_iqr',
        'kwargs': {'column': 'Fare', 'whisker_width': 1.5}
    }
]
```

---

## Decision Tree

```
How many outliers?
├── < 5% of data
│   └── Use clip_outliers_iqr (preserve data)
└── > 5% of data
    ├── Large dataset (>1000 rows)
    │   └── Can use either method
    └── Small dataset (<1000 rows)
        └── Use clip_outliers_iqr (preserve data)
```

---

## Comparison

| Aspect | `clip_outliers_iqr` | `remove_outliers_iqr` |
|--------|---------------------|----------------------|
| **Dataset Size** | Unchanged | Reduced |
| **Outlier Values** | Capped to bounds | Row deleted |
| **Data Loss** | None | Yes |
| **Statistical Impact** | Moderate | High |
| **Use Case** | General purpose | Clean datasets |

---

## Output Information

Both functions print diagnostic information:

```
Column 'Fare': Found 5 outliers (1 below 2.50, 4 above 95.30)
Clipping outliers to range [2.50, 95.30]
```

```
Column 'Fare': Removing 5 rows with outliers (outside range [2.50, 95.30])
Original rows: 891, Remaining rows: 886
```

---

## Features

✅ IQR-based detection (robust method)  
✅ Configurable sensitivity via `whisker_width`  
✅ Detailed logging of operations  
✅ Non-destructive (returns copy)  
✅ Type validation  
✅ Handles NaN values properly  

---

## Code Examples

### Basic Usage
```python
from preprocessing_function import clip_outliers_iqr, remove_outliers_iqr

# Clip method (preserve rows)
df_clipped = clip_outliers_iqr(df, 'Fare')

# Remove method (delete rows)
df_cleaned = remove_outliers_iqr(df, 'Fare')

# Custom sensitivity
df_conservative = clip_outliers_iqr(df, 'Fare', whisker_width=3.0)
```

### Integration with Workflow
```python
# From suggestion dictionary
suggestion = {
    'function_to_call': 'clip_outliers_iqr',
    'kwargs': {'column': 'Fare', 'whisker_width': 1.5}
}

# Apply transformation
df_transformed = eval(suggestion['function_to_call'])(df, **suggestion['kwargs'])
```

---

**Author**: Zhiqi Yang  
**Part of**: Numerical Features & Outlier Handling Module  
**Branch**: zhiqi-yang


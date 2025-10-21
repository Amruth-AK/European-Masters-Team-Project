# Data Contract: Team Integration Document

## Purpose
This document defines the data interfaces between different modules to ensure smooth integration.

---

## Module 1: Analysis (Adrian & Alisa)

### File: `analyze.py`

### Function Signature
```python
def analyze_dataset(df: pd.DataFrame, target_column: str) -> dict:
    """
    Analyze dataset and generate diagnostic report.
    
    Args:
        df: Input DataFrame
        target_column: Name of target variable
    
    Returns:
        dict: Analysis report (see structure below)
    """
```

### Output Contract (PROPOSAL - TO BE CONFIRMED)

```python
{
    # Required Keys
    'feature_types': {
        '<column_name>': 'numerical' | 'categorical',
        ...
    },
    
    # Optional but Recommended Keys
    'missing_values': {
        '<column_name>': {
            'count': int,
            'percentage': float,
            'has_missing': bool
        },
        ...
    },
    
    'outliers': {
        '<column_name>': {
            'has_outliers': bool,
            'count': int,
            'percentage': float,
            'method': 'IQR' | 'Z-score' | ...
        },
        ...
    },
    
    'numerical_stats': {
        '<column_name>': {
            'mean': float,
            'median': float,
            'std': float,
            'min': float,
            'max': float,
            'skewness': float,
            'kurtosis': float
        },
        ...
    },
    
    'categorical_stats': {
        '<column_name>': {
            'cardinality': int,
            'unique_values': list | None,
            'most_frequent': str | int,
            'most_frequent_count': int
        },
        ...
    },
    
    'duplicates': {
        'has_duplicates': bool,
        'count': int,
        'percentage': float
    },
    
    'metadata': {
        'total_rows': int,
        'total_columns': int,
        'memory_usage': str
    }
}
```

### Status
- ⏳ **Pending**: Awaiting confirmation from Adrian & Alisa
- 📝 **Action**: Please review and update this structure

---

## Module 2: Preprocessing Suggestions (Yang, Amruth, Tecla)

### File: `preprocessing_suggestions.py`

### Function Signature
```python
def generate_suggestions(analysis_report: dict) -> list:
    """
    Generate preprocessing suggestions from analysis report.
    
    Args:
        analysis_report: Output from analyze.py
    
    Returns:
        list: Preprocessing suggestions (see structure below)
    """
```

### Input Contract
Receives the `analysis_report` dictionary from Module 1 (structure above).

### Output Contract

```python
[
    {
        'feature': str,              # Column name or "All columns"
        'issue': str,                # Problem description
        'suggestion': str,           # Human-readable recommendation
        'function_to_call': str,     # Exact function name from preprocessing_function.py
        'kwargs': dict,              # Arguments for the function
        'priority': int              # 1=high, 2=medium, 3=medium, 4=low
    },
    ...
]
```

### Example Output
```python
[
    {
        'feature': 'Age',
        'issue': 'Missing values (19% of data)',
        'suggestion': 'Apply median imputation',
        'function_to_call': 'impute_median',
        'kwargs': {'column': 'Age'},
        'priority': 1
    },
    {
        'feature': 'Fare',
        'issue': 'High skewness (4.72)',
        'suggestion': 'Apply standard scaling',
        'function_to_call': 'standard_scaler',
        'kwargs': {'column': 'Fare'},
        'priority': 4
    }
]
```

### Team Responsibilities

#### Yang/Zhiqi
- `_suggest_numerical_scaling()`
- `_suggest_outlier_handling()`

#### Amruth
- `_suggest_categorical_encoding()`

#### Tecla
- `_suggest_missing_values()`
- `_suggest_duplicates()`

### Status
- ✅ **Complete**: Framework implemented
- ⏳ **Pending**: Each team member implements their section

---

## Module 3: Preprocessing Functions (Yang, Amruth, Tecla)

### File: `Preprocessing suggestion/preprocessing_function.py`

### Function Signatures

All functions follow this pattern:
```python
def function_name(df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
    """
    Brief description.
    
    Args:
        df: Input DataFrame
        column: Column to process
        **kwargs: Additional parameters
    
    Returns:
        DataFrame: Transformed copy (non-destructive)
    """
    df = df.copy()
    # ... transformation logic ...
    return df
```

### Implemented Functions (Yang/Zhiqi)

✅ `standard_scaler(df, column)`  
✅ `minmax_scaler(df, column, feature_range=(0,1))`  
✅ `clip_outliers_iqr(df, column, whisker_width=1.5)`  
✅ `remove_outliers_iqr(df, column, whisker_width=1.5)`  

### Pending Functions (Amruth)

⏳ `encode_one_hot(df, column)`  
⏳ `encode_label(df, column)`  
⏳ `encode_target(df, column, target)`  

### Pending Functions (Tecla)

⏳ `impute_median(df, column)`  
⏳ `impute_mean(df, column)`  
⏳ `impute_mode(df, column)`  
⏳ `remove_duplicates(df)`  
⏳ `delete_missing_rows(df, column)`  

---

## Module 4: Main Integration (Everyone)

### File: `main.py`

### Workflow
```
User Input → analyze.py → preprocessing_suggestions.py → User Confirmation → preprocessing_function.py → Output
```

### Status
- ✅ **Complete**: Basic framework ready
- 🧪 **Testing**: Needs integration testing

---

## Critical Dependencies

### Module 2 depends on Module 1
- **Blocker**: Cannot finalize preprocessing_suggestions.py until analyze.py output format is confirmed
- **Action**: Meeting with Adrian & Alisa to confirm analysis_report structure

### Module 4 depends on Modules 1, 2, 3
- **Blocker**: Full integration testing requires all preprocessing functions
- **Workaround**: Use mock data for now (`mock_analysis_output.py`)

---

## Testing Strategy

### Unit Testing
- Each team member tests their own functions independently
- Use sample DataFrames (e.g., Titanic dataset)

### Integration Testing
1. Test Module 1 → Module 2 with mock analysis output
2. Test Module 2 → Module 3 with sample suggestions
3. Test end-to-end pipeline with main.py

### Test Data
- Primary: Titanic dataset (everyone familiar)
- Secondary: Iris, Boston Housing, or other standard datasets

---

## Communication Plan

### When you need to update this contract:
1. Edit this file
2. Commit to your branch
3. Tag relevant team members in chat

### When you implement your functions:
1. Follow the function signature pattern
2. Add docstrings
3. Test with sample data
4. Update status in this document

---

## Mock Data for Development

### File: `mock_analysis_output.py`
- Contains realistic example of analyze.py output
- Use this until Adrian & Alisa provide actual implementation
- Two versions available:
  - `get_mock_analysis_report()` - Full version
  - `get_simplified_mock()` - Quick testing

---

## Questions & Issues

### Open Questions
1. **Adrian & Alisa**: Is the proposed analysis_report structure correct?
2. **Everyone**: Should we prioritize certain preprocessing steps?
3. **Everyone**: How do we handle interdependent transformations?

### Known Issues
- None yet

---

## Meeting Action Items

### For Adrian & Alisa
- [ ] Review proposed `analysis_report` structure
- [ ] Confirm/modify dictionary keys and data types
- [ ] Provide timeline for analyze.py completion
- [ ] (Optional) Share code snippet or example output

### For Yang
- [x] Implement numerical scaling functions
- [x] Implement outlier handling functions
- [ ] Implement rule logic in preprocessing_suggestions.py
- [ ] Test integration with mock data

### For Amruth
- [ ] Implement categorical encoding functions
- [ ] Implement rule logic in preprocessing_suggestions.py
- [ ] Test with sample categorical data

### For Tecla
- [ ] Implement missing value handling functions
- [ ] Implement duplicate handling functions
- [ ] Implement rule logic in preprocessing_suggestions.py
- [ ] Test with data containing missing values

### For Everyone
- [ ] Review this data contract
- [ ] Test main.py with your functions
- [ ] Document any issues or suggestions

---

**Last Updated**: [Add date after team meeting]  
**Next Review**: [Schedule after initial implementation]


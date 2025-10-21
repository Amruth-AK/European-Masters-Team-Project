#inspect dataset metadata

import pandas as pd
import numpy as np


def analyze_structure(df, target):
    features = df.drop(columns=[target])
    info = []

    for col in features.columns:
        dtype = features[col].dtype
        missing_ratio = features[col].isnull().mean()
        unique_values = features[col].nunique()
        cardinality = 'high' if unique_values > 50 else 'low'
        info.append({
            'column': col,
            'dtype': dtype,
            'missing_ratio': missing_ratio,
            'unique_values': unique_values,
            'cardinality': cardinality
        })

    target_type = 'classification' if df[target].nunique() < 20 and df[target].dtype != float else 'regression'

    return pd.DataFrame(info), target_type






#GENERATE PREPROCESSING SUGGESTIONS

def suggest_preprocessing(info_df):
    suggestions = {}

    for _, row in info_df.iterrows():
        col = row['column']
        dtype = row['dtype']
        recs = []

        if row['missing_ratio'] > 0.1:
            recs.append("Consider imputing missing values")

        if np.issubdtype(dtype, np.number):
            if row['unique_values'] < 10:
                recs.append("Check if numeric column represents categories")
            recs.append("Scale numeric features (StandardScaler or MinMaxScaler)")
        else:
            if row['cardinality'] == 'low':
                recs.append("Use OneHotEncoder for categorical encoding")
            else:
                recs.append("Use TargetEncoder or HashingEncoder for high-cardinality features")

        suggestions[col] = recs

    return suggestions




#APPLY PREPROCESSING
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessing_pipeline(df, suggestions):
    numeric_features = df.select_dtypes(include=np.number).columns
    categorical_features = df.select_dtypes(exclude=np.number).columns

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor


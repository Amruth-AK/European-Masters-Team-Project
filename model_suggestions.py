#SUGGEST SUITABLE ML MODELS
def suggest_models(target_type, n_samples, n_features):
    if target_type == 'classification':
        if n_samples < 10000:
            return ["LogisticRegression", "RandomForestClassifier"]
        else:
            return ["LightGBM", "XGBoost"]
    else:
        if n_samples < 10000:
            return ["LinearRegression", "RandomForestRegressor"]
        else:
            return ["LightGBMRegressor", "XGBoostRegressor"]

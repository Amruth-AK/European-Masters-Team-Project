import pandas as pd
import numpy as np
import lightgbm as lgb
# --- FIX 1: Explicit import for the callback ---
from lightgbm import early_stopping
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy.stats import skew
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class MetaDataCollector:
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.meta_data_log = []
        
        # --- PROXY MODEL PARAMETERS ---
        # Constrained depth forces the model to rely on feature engineering
        # rather than brute-forcing the data.
        self.base_params = {
            'n_estimators': 150,      
            'learning_rate': 0.1,     
            'num_leaves': 31,         
            'max_depth': 5,           # <--- CRITICAL: Constrain depth!
            'subsample': 0.8,         
            'colsample_bytree': 0.8,  
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }

    def get_dataset_meta(self, df):
        return {
            'n_rows': df.shape[0],
            'n_cols': df.shape[1],
            'cat_ratio': len(df.select_dtypes(include=['object', 'category', 'bool']).columns) / df.shape[1],
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'row_col_ratio': df.shape[0] / df.shape[1]
        }

    def get_column_meta(self, col):
        is_num = np.issubdtype(col.dtype, np.number)
        res = {
            'dtype': str(col.dtype),
            'null_pct': col.isnull().mean(),
            'unique_count': col.nunique(),
            'unique_ratio': col.nunique() / len(col)
        }
        if is_num:
            clean_col = col.dropna()
            res.update({
                'skewness': skew(clean_col) if len(clean_col) > 0 else 0,
                'std_dev': clean_col.std() if len(clean_col) > 0 else 0,
                'coeff_variation': clean_col.std() / clean_col.mean() if clean_col.mean() != 0 else 0
            })
        return res

    def evaluate_with_intervention(self, X, y, col_to_transform=None, method=None):
        """
        Runs 5-Fold CV with Leakage-Free Transformations and Early Stopping.
        Returns: (mean_score, std_dev_score)
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # --- CONSISTENT CATEGORICAL HANDLING ---
            # Map all categories based on training set to integers
            cat_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns
            for c in cat_cols:
                # Skip the column we are currently engineering (it gets handled below)
                if c == col_to_transform: continue 
                
                # Create mapping from Train
                mapping = {v: k for k, v in enumerate(X_train[c].astype('category').cat.categories)}
                # Apply to Train & Val, fill unknown with -1
                X_train[c] = X_train[c].map(mapping).fillna(-1)
                X_val[c] = X_val[c].map(mapping).fillna(-1)

            # --- APPLY INTERVENTION (Within Fold) ---
            if col_to_transform and method:
                try:
                    if method == 'log_transform':
                        offset = abs(X_train[col_to_transform].min()) + 1 if X_train[col_to_transform].min() <= 0 else 0
                        X_train[col_to_transform] = np.log1p(X_train[col_to_transform] + offset)
                        X_val[col_to_transform] = np.log1p(X_val[col_to_transform] + offset)
                    
                    elif method == 'standard_scaler':
                        scaler = StandardScaler()
                        X_train[col_to_transform] = scaler.fit_transform(X_train[[col_to_transform]])
                        X_val[col_to_transform] = scaler.transform(X_val[[col_to_transform]])
                    
                    elif method == 'yeo_johnson':
                        pt = PowerTransformer(method='yeo-johnson')
                        X_train[col_to_transform] = pt.fit_transform(X_train[[col_to_transform]])
                        X_val[col_to_transform] = pt.transform(X_val[[col_to_transform]])

                    elif method == 'frequency_encoding':
                        freq = X_train[col_to_transform].value_counts(normalize=True)
                        X_train[col_to_transform] = X_train[col_to_transform].map(freq).fillna(0)
                        X_val[col_to_transform] = X_val[col_to_transform].map(freq).fillna(0)
                    
                    elif method == 'target_encoding':
                        # Smoothed Target Encoding
                        global_mean = y_train.mean()
                        agg = y_train.groupby(X_train[col_to_transform]).agg(['count', 'mean'])
                        counts = agg['count']
                        means = agg['mean']
                        m = 10 
                        smooth = (counts * means + m * global_mean) / (counts + m)
                        
                        X_train[col_to_transform] = X_train[col_to_transform].map(smooth).fillna(global_mean)
                        X_val[col_to_transform] = X_val[col_to_transform].map(smooth).fillna(global_mean)
                        
                except Exception as e:
                    return None, None

            # --- FINAL PREP ---
            # Ensure any remaining object columns are converted to codes (fallback)
            for df_tmp in [X_train, X_val]:
                for c in df_tmp.select_dtypes(include=['object', 'category', 'bool']).columns:
                    df_tmp[c] = df_tmp[c].astype('category').cat.codes

            # --- TRAIN & EVALUATE WITH EARLY STOPPING ---
            if self.task_type == 'classification':
                model = lgb.LGBMClassifier(**self.base_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    # --- FIX 1: Correct usage of early_stopping callback ---
                    callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
                )
                probs = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
                scores.append(metrics.roc_auc_score(y_val, probs))

            else:
                model = lgb.LGBMRegressor(**self.base_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='l2',
                    # --- FIX 1: Correct usage of early_stopping callback ---
                    callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
                )
                preds = model.predict(X_val, num_iteration=model.best_iteration_)
                scores.append(metrics.mean_squared_error(y_val, preds))

        return np.mean(scores), np.std(scores)

    def collect(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        ds_meta = self.get_dataset_meta(X)
        print(f"Starting analysis on {ds_meta['n_cols']} columns...")

        base_score, base_std = self.evaluate_with_intervention(X, y)
        print(f"Baseline Score: {base_score:.5f} (std: {base_std:.5f})")

        # Helper to calculate Delta correctly based on task type
        def get_delta(base, new):
            if self.task_type == 'classification':
                return new - base # Higher AUC is better
            return base - new     # Lower MSE is better

        # 3. Systematic Intervention Loop
        for col in X.columns:
            col_meta = self.get_column_meta(X[col])
            is_num = np.issubdtype(X[col].dtype, np.number)
            
            methods = ['log_transform', 'standard_scaler', 'yeo_johnson'] if is_num else ['frequency_encoding', 'target_encoding']

            for method in methods:
                new_score, new_std = self.evaluate_with_intervention(X, y, col_to_transform=col, method=method)
                
                if new_score is not None:
                    # --- FIX 2: Normalized Delta Calculation ---
                    delta = get_delta(base_score, new_score)
                    
                    self.meta_data_log.append({
                        **ds_meta,
                        **col_meta,
                        'column_name': col,
                        'method': method,
                        'delta': delta,
                        'absolute_score': new_score,
                        'is_interaction': False
                    })
                    print(f"  - {col} | {method}: {delta:+.5f}")

        # 4. Interaction Interventions
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) >= 2:
            top_nums = X[num_cols].var().sort_values(ascending=False).head(5).index
            for i, col_a in enumerate(top_nums):
                for col_b in top_nums[i+1:]:
                    X_interact = X.copy()
                    X_interact[f"{col_a}_x_{col_b}"] = X_interact[col_a] * X_interact[col_b]
                    
                    new_score, new_std = self.evaluate_with_intervention(X_interact, y)
                    
                    # --- FIX 2: Normalized Delta Calculation ---
                    delta = get_delta(base_score, new_score)
                    
                    self.meta_data_log.append({
                        **ds_meta,
                        'column_name': f"{col_a}_x_{col_b}",
                        'method': 'product_interaction',
                        'delta': delta,
                        'absolute_score': new_score,
                        'is_interaction': True,
                        'dtype': 'interaction_float'
                    })
                    print(f"  - Interaction | {col_a} * {col_b}: {delta:+.5f}")

        return pd.DataFrame(self.meta_data_log)

# --- EXECUTION ---
if __name__ == "__main__":
    # Load Titanic as a sample dataset
    data = sns.load_dataset('titanic')
    # Pre-processing only to make it runnable (Impute NaNs)
    data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
    data['age'] = data['age'].fillna(data['age'].median())
    data = data.drop(columns=['deck', 'embark_town', 'who', 'alive', 'class'])

    collector = MetaDataCollector(task_type='classification')
    meta_results = collector.collect(data, 'survived')

    print("\nMeta-Data Collection Results:")
    print(meta_results.sort_values(by='delta', ascending=False).head(10))
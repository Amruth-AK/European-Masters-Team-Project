# main.py
import pandas as pd

from preprocessing_suggestions import analyze_structure, suggest_preprocessing, suggest_models

df = pd.read_csv("dataset.csv")
info_df, target_type = analyze_structure(df, target="label")
preproc_suggestions = suggest_preprocessing(info_df)
model_suggestions = suggest_models(target_type, df.shape[0], df.shape[1])

print("=== Preprocessing Suggestions ===")
for col, recs in preproc_suggestions.items():
    print(f"{col}: {', '.join(recs)}")

print("\n=== Model Suggestions ===")
print(model_suggestions)
















#=======================================
#test
#========================================

df = pd.DataFrame({
    "A": [1, 2, np.nan, 4],
    "B": [7, np.nan, 9, 10]
})

imputer = KNNImputer(n_neighbors=2)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(df_imputed)

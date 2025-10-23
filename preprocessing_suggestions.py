import pandas as pd
from preprocessing_function import (
    delete_missing_rows,
    delete_missing_columns,
    impute_mean,
    impute_median,
    impute_mode,
    delete_duplicates
)
#============================================================================
#handles missing values
##============================================================================


def handle_missing_values(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """
    Handles missing values based on user preference and analysis results.
    """
    missing_info = pd.DataFrame.from_dict(results['missing_values'], orient='index')
    total_missing = missing_info['missing_count'].sum()

    if total_missing == 0:
        print("\n No missing values detected in the dataset.")
        return df

    print("\n⚠ Missing values detected in the dataset:")
    print(missing_info)

    choice = input(
        "\nHow would you like to handle missing values?\n"
        "1 - Delete rows with too many missing values\n"
        "2 - Delete columns with too many missing values\n"
        "3 - Impute using mean\n"
        "4 - Impute using median\n"
        "5 - Impute using mode\n"
        "6 - Skip\n"
        "Enter your choice (1-6): "
    )

    if choice == "1":
        threshold = float(input("Enter threshold for row deletion (e.g., 0.5): ") or 0.5)
        df = delete_missing_rows(df, threshold)

    elif choice == "2":
        threshold = float(input("Enter threshold for column deletion (e.g., 0.5): ") or 0.5)
        df = delete_missing_columns(df, threshold)

    elif choice == "3":
        cols = input("Enter column names to impute with mean (comma-separated or 'all'): ")
        columns = df.select_dtypes(include='number').columns.tolist() if cols.lower() == 'all' else [c.strip() for c in cols.split(',')]
        df = impute_mean(df, columns)

    elif choice == "4":
        cols = input("Enter column names to impute with median (comma-separated or 'all'): ")
        columns = df.select_dtypes(include='number').columns.tolist() if cols.lower() == 'all' else [c.strip() for c in cols.split(',')]
        df = impute_median(df, columns)

    elif choice == "5":
        cols = input("Enter column names to impute with mode (comma-separated or 'all'): ")
        columns = df.columns.tolist() if cols.lower() == 'all' else [c.strip() for c in cols.split(',')]
        df = impute_mode(df, columns)

    elif choice == "6":
        print(" Skipping missing value handling.")

    else:
        print("Invalid choice, skipping missing value handling.")

    return df

#============================================================================
#handles duplicates
##============================================================================


def handle_duplicates(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """
    Handles duplicate rows based on user preference and analysis results.
    """
    duplicate_info = results.get('row_duplicate_info', results.get('duplicate_info', {}))
    total_dups = duplicate_info.get('total_duplicates', 0)

    if total_dups == 0:
        print("\n No duplicate rows detected in the dataset.")
        return df

    print(f"\n️ {total_dups} duplicate rows detected "
          f"({duplicate_info.get('duplicate_percentage', 0):.2f}% of dataset).")

    choice = input("Would you like to remove duplicate rows? (y/n): ").strip().lower()
    if choice == "y":
        subset_cols = input("Enter subset of columns for duplicate detection (comma-separated or 'all'): ")
        subset = None if subset_cols.lower() == "all" else [c.strip() for c in subset_cols.split(',')]
        df = delete_duplicates(df, subset)
    else:
        print(" Skipping duplicate removal.")

    return df




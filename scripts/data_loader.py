"""
Data loading utilities for IE421 Olympics Data Science Project.
"""
import pandas as pd
import numpy as np
from scripts.config import HISTORICAL_DATA, PARIS2024_DATA, DISCIPLINE_COLUMNS


def load_historical_data():
    """
    Load the historical Olympic athlete events dataset (1896-2016).

    Returns:
        pd.DataFrame: Historical Olympics data
    """
    df = pd.read_csv(HISTORICAL_DATA)
    print(f"Loaded historical data: {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_paris2024_data():
    """
    Load the Paris 2024 athlete registration dataset.

    Returns:
        pd.DataFrame: Paris 2024 athletes data
    """
    df = pd.read_csv(PARIS2024_DATA)
    print(f"Loaded Paris 2024 data: {len(df):,} rows, {len(df.columns)} columns")
    return df


def filter_summer(df):
    """
    Filter dataset to Summer Olympics only.

    Args:
        df: DataFrame with 'Season' column

    Returns:
        pd.DataFrame: Filtered to Summer Olympics
    """
    if 'Season' not in df.columns:
        print("Warning: 'Season' column not found. Returning original dataframe.")
        return df

    filtered = df[df['Season'] == 'Summer'].copy()
    print(f"Filtered to Summer Olympics: {len(filtered):,} rows")
    return filtered


def clean_biometrics(df, cols=None):
    """
    Remove rows with missing biometric data.

    Args:
        df: DataFrame to clean
        cols: List of columns to check for NA (default: Age, Height, Weight)

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if cols is None:
        cols = ["Age", "Height", "Weight"]

    # Only check columns that exist
    existing_cols = [c for c in cols if c in df.columns]

    if not existing_cols:
        print("Warning: No biometric columns found. Returning original dataframe.")
        return df

    original_len = len(df)
    cleaned = df.dropna(subset=existing_cols).copy()
    removed = original_len - len(cleaned)
    print(f"Cleaned biometrics: removed {removed:,} rows with missing {existing_cols}")
    return cleaned


def detect_discipline_column(df):
    """
    Auto-detect the discipline/sport column in Paris 2024 data.

    Args:
        df: DataFrame to check

    Returns:
        str: Name of the discipline column, or None if not found
    """
    for col in DISCIPLINE_COLUMNS:
        if col in df.columns:
            print(f"Detected discipline column: '{col}'")
            return col

    print(f"Warning: No discipline column found among {DISCIPLINE_COLUMNS}")
    return None


def standardize_country_columns(df):
    """
    Ensure consistent country column naming.
    Safe no-op if columns already correct.

    Args:
        df: DataFrame to standardize

    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    # For historical data, NOC is the standard country code column
    # Just return as-is since NOC is already present
    return df


if __name__ == "__main__":
    # Test loading functions
    print("Testing data loaders...")

    hist = load_historical_data()
    print(f"Historical columns: {list(hist.columns)}")

    paris = load_paris2024_data()
    print(f"Paris 2024 columns: {list(paris.columns)}")

    disc_col = detect_discipline_column(paris)
    if disc_col:
        print(f"Sample disciplines: {paris[disc_col].head()}")

"""
Q2: Medal Prediction using Multiple Linear Regression
- Train: 1960-2012
- Validate: 2016
- Features: delegation_size, prev_medals, rolling_avg_medals
- Focus: Top-20 NOCs by medal count for detailed evaluation
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import VISUALS_DIR, FIGURE_DPI, FIGURE_SIZE, TRAIN_YEARS, VALID_YEAR, SUMMER_ONLY
from scripts.data_loader import load_historical_data, filter_summer


def create_country_year_aggregation(df):
    """
    Aggregate athlete-level data to country-year level.

    Args:
        df: Historical Olympics DataFrame

    Returns:
        pd.DataFrame: Country-year level aggregation
    """
    # Filter to Summer Olympics if configured
    if SUMMER_ONLY:
        df = filter_summer(df)

    # Aggregate by NOC and Year
    agg = df.groupby(['NOC', 'Year']).agg(
        delegation_size=('ID', 'nunique'),  # Number of unique athletes
        total_medals=('Medal', lambda x: x.notna().sum())  # Count non-null medals
    ).reset_index()

    print(f"Created aggregation: {len(agg):,} country-year observations")
    return agg


def create_features(df):
    """
    Create features for the regression model.

    Features:
    - delegation_size: Number of unique athletes
    - prev_medals: Medals from previous Olympics (lag-1)
    - rolling_avg_medals: Rolling mean of last 2 Olympics

    Args:
        df: Country-year aggregated DataFrame

    Returns:
        pd.DataFrame: DataFrame with features added
    """
    df = df.sort_values(['NOC', 'Year']).copy()

    # Create lag features per country
    df['prev_medals'] = df.groupby('NOC')['total_medals'].shift(1)

    # Rolling average of last 2 Olympics (not including current)
    df['rolling_avg_medals'] = df.groupby('NOC')['total_medals'].transform(
        lambda x: x.shift(1).rolling(window=2, min_periods=1).mean()
    )

    # Fill missing values with 0 (for first appearances)
    df['prev_medals'] = df['prev_medals'].fillna(0)
    df['rolling_avg_medals'] = df['rolling_avg_medals'].fillna(0)

    print(f"Features created. Sample:\n{df.head()}")
    return df


def train_model(df, train_start, train_end):
    """
    Train Multiple Linear Regression model.

    Args:
        df: DataFrame with features
        train_start: Start year for training
        train_end: End year for training

    Returns:
        tuple: (model, feature_names, train_metrics)
    """
    # Filter to training period
    train_df = df[(df['Year'] >= train_start) & (df['Year'] <= train_end)].copy()

    # Remove rows with missing features
    train_df = train_df.dropna()

    feature_cols = ['delegation_size', 'prev_medals', 'rolling_avg_medals']
    target_col = 'total_medals'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    print(f"Training on {len(train_df):,} observations ({train_start}-{train_end})")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")

    # Print coefficients
    print("\nModel Coefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"  {feat}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    return model, feature_cols, {'rmse': train_rmse, 'r2': train_r2}


def validate_model(model, df, valid_year, feature_cols):
    """
    Validate model on specified year with Top-20 NOC focus.

    Args:
        model: Trained LinearRegression model
        df: DataFrame with features
        valid_year: Year to validate on
        feature_cols: List of feature column names

    Returns:
        tuple: (validation_df, all_metrics, top20_metrics, top20_nocs)
    """
    # Filter to validation year
    valid_df = df[df['Year'] == valid_year].copy()
    valid_df = valid_df.dropna()

    if len(valid_df) == 0:
        raise ValueError(f"No data found for validation year {valid_year}")

    X_valid = valid_df[feature_cols]
    y_valid = valid_df['total_medals']

    # Predict
    y_pred = model.predict(X_valid)
    valid_df['predicted_medals'] = y_pred

    # Calculate metrics for ALL NOCs
    rmse_all = np.sqrt(mean_squared_error(y_valid, y_pred))
    mae_all = mean_absolute_error(y_valid, y_pred)
    r2_all = r2_score(y_valid, y_pred)

    print(f"\n--- Validation Results: ALL NOCs ({valid_year}) ---")
    print(f"  NOCs evaluated: {len(valid_df)}")
    print(f"  RMSE: {rmse_all:.2f}")
    print(f"  MAE: {mae_all:.2f}")
    print(f"  R²: {r2_all:.4f}")

    # Identify Top-20 NOCs by actual medals in 2016
    top20_df = valid_df.nlargest(20, 'total_medals')
    top20_nocs = top20_df['NOC'].tolist()

    # Calculate metrics for Top-20 NOCs only
    y_valid_top20 = top20_df['total_medals']
    y_pred_top20 = top20_df['predicted_medals']

    rmse_top20 = np.sqrt(mean_squared_error(y_valid_top20, y_pred_top20))
    mae_top20 = mean_absolute_error(y_valid_top20, y_pred_top20)
    r2_top20 = r2_score(y_valid_top20, y_pred_top20)

    print(f"\n--- Validation Results: TOP-20 NOCs ({valid_year}) ---")
    print(f"  Top-20 NOCs: {', '.join(top20_nocs[:10])}...")
    print(f"  RMSE: {rmse_top20:.2f}")
    print(f"  MAE: {mae_top20:.2f}")
    print(f"  R²: {r2_top20:.4f}")

    all_metrics = {'rmse': rmse_all, 'mae': mae_all, 'r2': r2_all, 'n': len(valid_df)}
    top20_metrics = {'rmse': rmse_top20, 'mae': mae_top20, 'r2': r2_top20, 'n': 20}

    return valid_df, all_metrics, top20_metrics, top20_nocs


def plot_prediction_scatter(valid_df, all_metrics, top20_metrics, top20_nocs, output_path):
    """
    Create scatter plot of actual vs predicted medals with Top-20 NOC labels.

    Args:
        valid_df: Validation DataFrame with predictions
        all_metrics: Dictionary of all-NOC validation metrics
        top20_metrics: Dictionary of Top-20 NOC validation metrics
        top20_nocs: List of Top-20 NOC codes
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(14, 10))

    # Separate Top-20 and others
    top20_df = valid_df[valid_df['NOC'].isin(top20_nocs)]
    other_df = valid_df[~valid_df['NOC'].isin(top20_nocs)]

    # Plot other NOCs (smaller, lighter)
    plt.scatter(other_df['predicted_medals'], other_df['total_medals'],
                alpha=0.4, s=50, c='#90CAF9', edgecolors='gray', linewidth=0.3,
                label=f'Other NOCs (n={len(other_df)})')

    # Plot Top-20 NOCs (larger, darker)
    plt.scatter(top20_df['predicted_medals'], top20_df['total_medals'],
                alpha=0.8, s=120, c='#1565C0', edgecolors='black', linewidth=1,
                label=f'Top-20 NOCs (n=20)')

    # Add y=x reference line
    max_val = max(valid_df['total_medals'].max(), valid_df['predicted_medals'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')

    # Label ONLY Top-20 NOCs with their codes
    for _, row in top20_df.iterrows():
        plt.annotate(row['NOC'],
                     (row['predicted_medals'], row['total_medals']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, fontweight='bold', alpha=0.9,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Add metrics box with both ALL and Top-20 metrics
    metrics_text = (
        f"ALL NOCs (n={all_metrics['n']}):\n"
        f"  RMSE: {all_metrics['rmse']:.2f}\n"
        f"  MAE: {all_metrics['mae']:.2f}\n"
        f"  R²: {all_metrics['r2']:.4f}\n\n"
        f"TOP-20 NOCs:\n"
        f"  RMSE: {top20_metrics['rmse']:.2f}\n"
        f"  MAE: {top20_metrics['mae']:.2f}\n"
        f"  R²: {top20_metrics['r2']:.4f}"
    )
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Styling
    plt.xlabel('Predicted Medal Count', fontsize=13, fontweight='bold')
    plt.ylabel('Actual Medal Count', fontsize=13, fontweight='bold')
    plt.title(f'Medal Prediction Accuracy - {VALID_YEAR} Olympics\n'
              f'(Model trained on {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}, Top-20 NOCs highlighted)',
              fontsize=14, fontweight='bold', pad=20)

    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)

    # Set axis limits
    plt.xlim(-5, max_val + 15)
    plt.ylim(-5, max_val + 15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=max(FIGURE_DPI, 200), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run Q2 Medal Prediction Analysis."""
    print("=" * 60)
    print("Q2: Medal Prediction (Multiple Linear Regression)")
    print("    Focus: Top-20 NOCs by medal count")
    print("=" * 60)

    # Ensure visuals directory exists
    os.makedirs(VISUALS_DIR, exist_ok=True)

    # Load and prepare data
    print("\n--- Loading and Preparing Data ---")
    hist_data = load_historical_data()
    country_year = create_country_year_aggregation(hist_data)
    country_year = create_features(country_year)

    # Train model
    print("\n--- Training Model ---")
    model, feature_cols, train_metrics = train_model(
        country_year,
        train_start=TRAIN_YEARS[0],
        train_end=TRAIN_YEARS[1]
    )

    # Validate on 2016 with Top-20 focus
    print("\n--- Validating Model ---")
    valid_df, all_metrics, top20_metrics, top20_nocs = validate_model(
        model, country_year, VALID_YEAR, feature_cols
    )

    # Create visualization with Top-20 labels
    scatter_path = os.path.join(VISUALS_DIR, "q2_prediction_scatter.png")
    plot_prediction_scatter(valid_df, all_metrics, top20_metrics, top20_nocs, scatter_path)

    # Print Top-20 predictions vs actual
    print("\n--- Top-20 Countries: Predicted vs Actual (2016) ---")
    top20_df = valid_df[valid_df['NOC'].isin(top20_nocs)].copy()
    top20_df = top20_df.sort_values('total_medals', ascending=False)
    top20_df['error'] = top20_df['predicted_medals'] - top20_df['total_medals']
    print(top20_df[['NOC', 'total_medals', 'predicted_medals', 'delegation_size', 'error']].to_string(index=False))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_metrics": train_metrics,
        "all_metrics": all_metrics,
        "top20_metrics": top20_metrics,
        "top20_nocs": top20_nocs,
        "validation_df": valid_df
    }


if __name__ == "__main__":
    main()

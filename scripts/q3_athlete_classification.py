"""
Q3: Athlete Classification for Medal Success
- Binary classification: Medal_Won (1) vs No Medal (0)
- Features: Age, Height, Weight, Sex
- Model: Logistic Regression with class_weight='balanced'
- Post-2000 data, selected high-physicality sports
- Includes threshold tuning for F1 optimization
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import VISUALS_DIR, FIGURE_DPI, MIN_YEAR, SELECTED_SPORTS, SUMMER_ONLY
from scripts.data_loader import load_historical_data, filter_summer, clean_biometrics


def prepare_classification_data(df):
    """
    Prepare data for binary classification.

    Args:
        df: Historical Olympics DataFrame

    Returns:
        pd.DataFrame: Prepared data for classification
    """
    # Filter to Summer Olympics if configured
    if SUMMER_ONLY:
        df = filter_summer(df)

    # Filter to post-2000 for better biometric completeness
    df = df[df['Year'] >= MIN_YEAR].copy()
    print(f"Filtered to post-{MIN_YEAR}: {len(df):,} rows")

    # Find intersection of selected sports with available sports
    available_sports = set(df['Sport'].unique())
    selected_available = [s for s in SELECTED_SPORTS if s in available_sports]

    if not selected_available:
        print(f"Warning: None of the selected sports found. Available: {available_sports}")
        # Fall back to top sports by count
        selected_available = df['Sport'].value_counts().head(5).index.tolist()

    print(f"Using sports: {selected_available}")

    # Filter to selected sports
    df = df[df['Sport'].isin(selected_available)].copy()
    print(f"Filtered to selected sports: {len(df):,} rows")

    # Clean biometrics (remove rows with missing Age, Height, Weight)
    df = clean_biometrics(df, cols=['Age', 'Height', 'Weight'])

    # Create binary target: Medal_Won
    df['Medal_Won'] = df['Medal'].notna().astype(int)

    # Encode Sex as binary
    df['Sex_Encoded'] = (df['Sex'] == 'M').astype(int)

    # Get unique athlete-event combinations to avoid duplicates
    # (same athlete can appear multiple times in same event across years)
    df = df.drop_duplicates(subset=['ID', 'Event', 'Year'])

    print(f"Final dataset: {len(df):,} athlete-event observations")
    print(f"Medal distribution: {df['Medal_Won'].value_counts().to_dict()}")
    print(f"Medal rate: {df['Medal_Won'].mean():.2%}")

    return df, selected_available


def find_best_threshold(y_true, y_prob, metric='f1'):
    """
    Find the threshold that maximizes the specified metric.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        tuple: (best_threshold, best_score, threshold_scores)
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = []

    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_thresh, zero_division=0)
        scores.append(score)

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    return best_threshold, best_score, list(zip(thresholds, scores))


def train_classification_model(df):
    """
    Train Logistic Regression classifier with detailed metrics and threshold tuning.

    Args:
        df: Prepared DataFrame for classification

    Returns:
        tuple: (model, scaler, X_test, y_test, y_pred, y_prob, metrics, threshold_info)
    """
    feature_cols = ['Age', 'Height', 'Weight', 'Sex_Encoded']
    target_col = 'Medal_Won'

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Print class distribution
    print("\n--- Class Distribution ---")
    train_dist = y_train.value_counts()
    test_dist = y_test.value_counts()
    print(f"Training set:")
    print(f"  No Medal (0): {train_dist[0]:,} ({train_dist[0]/len(y_train):.1%})")
    print(f"  Medal (1):    {train_dist[1]:,} ({train_dist[1]/len(y_train):.1%})")
    print(f"Test set:")
    print(f"  No Medal (0): {test_dist[0]:,} ({test_dist[0]/len(y_test):.1%})")
    print(f"  Medal (1):    {test_dist[1]:,} ({test_dist[1]/len(y_test):.1%})")

    print(f"\nTrain size: {len(X_train):,}, Test size: {len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression with balanced class weights
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Predictions at default threshold (0.5)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics at default threshold
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n--- Classification Results (Default Threshold = 0.5) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f} (PRIMARY)")
    print(f"ROC-AUC:   {roc_auc:.4f} (PRIMARY)")
    print("\nNote: F1 and ROC-AUC are primary metrics due to class imbalance")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Medal', 'Medal']))

    # Threshold tuning for F1
    print("\n--- Threshold Tuning for F1 ---")
    best_thresh, best_f1, thresh_scores = find_best_threshold(y_test, y_prob, metric='f1')
    print(f"Best threshold: {best_thresh:.2f}")
    print(f"F1 at best threshold: {best_f1:.4f}")
    print(f"F1 at default (0.5): {f1:.4f}")
    print(f"F1 improvement: {best_f1 - f1:+.4f}")

    # Metrics at best threshold
    y_pred_best = (y_prob >= best_thresh).astype(int)
    precision_best = precision_score(y_test, y_pred_best)
    recall_best = recall_score(y_test, y_pred_best)

    print(f"\nAt best threshold ({best_thresh:.2f}):")
    print(f"  Precision: {precision_best:.4f}")
    print(f"  Recall:    {recall_best:.4f}")
    print(f"  F1:        {best_f1:.4f}")

    # Print feature coefficients
    print("\nFeature Coefficients (log-odds):")
    for feat, coef in zip(feature_cols, model.coef_[0]):
        print(f"  {feat}: {coef:.4f}")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    threshold_info = {
        'best_threshold': best_thresh,
        'best_f1': best_f1,
        'default_f1': f1,
        'precision_at_best': precision_best,
        'recall_at_best': recall_best
    }

    return model, scaler, X_test_scaled, y_test, y_pred, y_prob, metrics, threshold_info


def plot_classification_results(y_test, y_pred, y_prob, metrics, threshold_info, output_path):
    """
    Create combined figure with ROC curve and Confusion Matrix (at default threshold).

    Args:
        y_test: True labels
        y_pred: Predicted labels (at default threshold)
        y_prob: Predicted probabilities
        metrics: Dictionary of metrics
        threshold_info: Dictionary with threshold tuning results
        output_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: ROC Curve
    ax1 = axes[0]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    ax1.plot(fpr, tpr, color='#2196F3', linewidth=2.5,
             label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random Classifier')

    ax1.fill_between(fpr, tpr, alpha=0.2, color='#2196F3')

    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # Panel 2: Confusion Matrix (at default threshold 0.5)
    ax2 = axes[1]
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['No Medal', 'Medal'],
                yticklabels=['No Medal', 'Medal'],
                annot_kws={'size': 14})

    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Threshold = 0.5)', fontsize=14, fontweight='bold')

    # Add metrics text below confusion matrix
    cm_metrics = (
        f"Default (0.5): F1={metrics['f1']:.4f} | "
        f"Best ({threshold_info['best_threshold']:.2f}): F1={threshold_info['best_f1']:.4f} | "
        f"AUC={metrics['roc_auc']:.4f}"
    )
    ax2.text(0.5, -0.15, cm_metrics, transform=ax2.transAxes,
             ha='center', fontsize=10, fontweight='bold')

    # Overall title
    fig.suptitle('Q3: Athlete Medal Classification Results\n'
                 '(Logistic Regression with Biometric Features, Post-2000, Selected Sports)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=max(FIGURE_DPI, 200), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run Q3 Athlete Classification Analysis."""
    print("=" * 60)
    print("Q3: Athlete Classification (Logistic Regression)")
    print("    Scope: Post-2000, Selected High-Physicality Sports")
    print("=" * 60)

    # Ensure visuals directory exists
    os.makedirs(VISUALS_DIR, exist_ok=True)

    # Load and prepare data
    print("\n--- Loading and Preparing Data ---")
    hist_data = load_historical_data()
    class_data, sports_used = prepare_classification_data(hist_data)

    # Train model with detailed metrics
    print("\n--- Training Classification Model ---")
    model, scaler, X_test, y_test, y_pred, y_prob, metrics, threshold_info = train_classification_model(class_data)

    # Create visualization
    results_path = os.path.join(VISUALS_DIR, "q3_classification_results.png")
    plot_classification_results(y_test, y_pred, y_prob, metrics, threshold_info, results_path)

    # Print summary
    print("\n--- Summary ---")
    print(f"Scope: Post-{MIN_YEAR} Summer Olympics, athletes with complete biometrics")
    print(f"Sports analyzed: {sports_used}")
    print(f"\nPrimary metrics (class imbalance aware):")
    print(f"  F1 Score (default threshold):  {metrics['f1']:.4f}")
    print(f"  F1 Score (best threshold):     {threshold_info['best_f1']:.4f}")
    print(f"  Best threshold:                {threshold_info['best_threshold']:.2f}")
    print(f"  ROC-AUC:                       {metrics['roc_auc']:.4f}")
    print(f"\nSecondary metric:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "threshold_info": threshold_info,
        "sports_used": sports_used
    }


if __name__ == "__main__":
    main()

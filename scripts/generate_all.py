"""
Master script to run all analyses for IE421 Olympics Data Science Project.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import VISUALS_DIR


def main():
    """Run all analyses and generate visualizations."""
    print("=" * 70)
    print("IE421 Olympics Data Science Project - Generate All")
    print("Team: Data and The City")
    print("=" * 70)

    # Ensure visuals directory exists
    os.makedirs(VISUALS_DIR, exist_ok=True)

    results = {}

    # Q1: Gender Parity Analysis
    print("\n" + "=" * 70)
    print("Running Q1: Gender Parity Analysis...")
    print("=" * 70)
    try:
        from scripts.q1_gender_parity import main as q1_main
        results['q1'] = q1_main()
        print("\n[OK] Q1 completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Q1 failed: {e}")
        results['q1'] = None

    # Q2: Medal Prediction
    print("\n" + "=" * 70)
    print("Running Q2: Medal Prediction (Regression) - Top-20 NOC Focus...")
    print("=" * 70)
    try:
        from scripts.q2_medal_prediction import main as q2_main
        results['q2'] = q2_main()
        print("\n[OK] Q2 completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Q2 failed: {e}")
        results['q2'] = None

    # Q3: Athlete Classification
    print("\n" + "=" * 70)
    print("Running Q3: Athlete Classification - Post-2000, Selected Sports...")
    print("=" * 70)
    try:
        from scripts.q3_athlete_classification import main as q3_main
        results['q3'] = q3_main()
        print("\n[OK] Q3 completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Q3 failed: {e}")
        results['q3'] = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # List generated visuals
    print("\nGenerated Visualizations:")
    expected_visuals = [
        "q1_gender_timeline.png",
        "q1_paris2024_parity.png",
        "q2_prediction_scatter.png",
        "q3_classification_results.png"
    ]

    for vis in expected_visuals:
        path = os.path.join(VISUALS_DIR, vis)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"  {status} {vis}")

    # Print key metrics
    print("\n" + "-" * 40)
    print("KEY METRICS")
    print("-" * 40)

    # Q1 Metrics
    if results.get('q1'):
        print("\nQ1 - Gender Parity:")
        print(f"  Final female ratio (2016): {results['q1'].get('final_female_ratio_2016', 'N/A'):.2%}")
        print(f"  Paris 2024 female ratio:   {results['q1'].get('paris2024_overall_female_ratio', 'N/A'):.2%}")

    # Q2 Metrics - Now with Top-20 focus
    if results.get('q2'):
        print("\nQ2 - Medal Prediction (2016 Validation):")
        all_m = results['q2'].get('all_metrics', {})
        top20_m = results['q2'].get('top20_metrics', {})
        print(f"  ALL NOCs (n={all_m.get('n', '?')}):")
        print(f"    RMSE: {all_m.get('rmse', 'N/A'):.2f}")
        print(f"    MAE:  {all_m.get('mae', 'N/A'):.2f}")
        print(f"    R²:   {all_m.get('r2', 'N/A'):.4f}")
        print(f"  TOP-20 NOCs:")
        print(f"    RMSE: {top20_m.get('rmse', 'N/A'):.2f}")
        print(f"    MAE:  {top20_m.get('mae', 'N/A'):.2f}")
        print(f"    R²:   {top20_m.get('r2', 'N/A'):.4f}")

    # Q3 Metrics - Now with threshold tuning
    if results.get('q3'):
        print("\nQ3 - Athlete Classification:")
        cm = results['q3'].get('metrics', {})
        ti = results['q3'].get('threshold_info', {})
        print(f"  Sports: {results['q3'].get('sports_used', 'N/A')}")
        print(f"  At default threshold (0.5):")
        print(f"    F1 Score: {cm.get('f1', 'N/A'):.4f}")
        print(f"    ROC-AUC:  {cm.get('roc_auc', 'N/A'):.4f}")
        print(f"    Accuracy: {cm.get('accuracy', 'N/A'):.4f}")
        print(f"  Threshold tuning:")
        print(f"    Best threshold: {ti.get('best_threshold', 'N/A'):.2f}")
        print(f"    F1 at best:     {ti.get('best_f1', 'N/A'):.4f}")

    print("\n" + "=" * 70)
    print("All analyses complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()

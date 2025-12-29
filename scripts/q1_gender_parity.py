"""
Q1: Gender Parity Analysis
- Historical trend: 1896-2016 (Summer Olympics)
- Paris 2024 snapshot: Discipline-level parity
"""
import os
import sys
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import VISUALS_DIR, FIGURE_DPI, FIGURE_SIZE, SUMMER_ONLY
from scripts.data_loader import (
    load_historical_data,
    load_paris2024_data,
    filter_summer,
    detect_discipline_column
)


def analyze_historical_gender_trend(df):
    """
    Analyze gender participation trend from 1896-2016.

    Args:
        df: Historical Olympics DataFrame

    Returns:
        pd.DataFrame: Yearly gender statistics
    """
    # Filter to Summer Olympics if configured
    if SUMMER_ONLY:
        df = filter_summer(df)

    # Get unique athletes per year (avoid counting same athlete multiple times per games)
    unique_athletes = df.drop_duplicates(subset=['ID', 'Year'])

    # Group by Year and Sex
    gender_counts = unique_athletes.groupby(['Year', 'Sex']).size().unstack(fill_value=0)

    # Ensure both columns exist
    if 'F' not in gender_counts.columns:
        gender_counts['F'] = 0
    if 'M' not in gender_counts.columns:
        gender_counts['M'] = 0

    # Calculate statistics
    gender_counts['Total'] = gender_counts['F'] + gender_counts['M']
    gender_counts['Female_Ratio'] = gender_counts['F'] / gender_counts['Total']
    gender_counts['Male_Ratio'] = gender_counts['M'] / gender_counts['Total']
    gender_counts['Gap_From_50'] = abs(gender_counts['Female_Ratio'] - 0.5)

    gender_counts = gender_counts.reset_index()
    print(f"Analyzed {len(gender_counts)} Olympic years")
    print(f"Female ratio range: {gender_counts['Female_Ratio'].min():.2%} - {gender_counts['Female_Ratio'].max():.2%}")

    return gender_counts


def plot_historical_gender_timeline(gender_data, output_path):
    """
    Create line chart showing gender participation over time.

    Args:
        gender_data: DataFrame with yearly gender statistics
        output_path: Path to save the visualization
    """
    plt.figure(figsize=FIGURE_SIZE)

    # Plot female ratio over time
    plt.plot(gender_data['Year'], gender_data['Female_Ratio'],
             color='#E91E63', linewidth=2.5, marker='o', markersize=4,
             label='Female Participation Ratio')

    # Add 50% parity line
    plt.axhline(y=0.5, color='#4CAF50', linestyle='--', linewidth=2,
                label='50% Gender Parity')

    # Fill area between curve and parity line
    plt.fill_between(gender_data['Year'], gender_data['Female_Ratio'], 0.5,
                     alpha=0.2, color='#E91E63')

    # Styling
    plt.xlabel('Olympic Year', fontsize=12, fontweight='bold')
    plt.ylabel('Female Participation Ratio', fontsize=12, fontweight='bold')
    plt.title('Evolution of Gender Parity in Summer Olympics (1896-2016)',
              fontsize=14, fontweight='bold', pad=20)

    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Set y-axis limits
    plt.ylim(0, 0.55)

    # Annotate key milestones
    first_women = gender_data[gender_data['F'] > 0].iloc[0]
    plt.annotate(f'First women: {int(first_women["Year"])}',
                 xy=(first_women['Year'], first_women['Female_Ratio']),
                 xytext=(first_women['Year'] + 15, first_women['Female_Ratio'] + 0.08),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    # Latest year annotation
    latest = gender_data.iloc[-1]
    plt.annotate(f'{int(latest["Year"])}: {latest["Female_Ratio"]:.1%}',
                 xy=(latest['Year'], latest['Female_Ratio']),
                 xytext=(latest['Year'] - 20, latest['Female_Ratio'] + 0.05),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_paris2024_parity(df):
    """
    Analyze gender parity by discipline for Paris 2024.

    Args:
        df: Paris 2024 athletes DataFrame

    Returns:
        pd.DataFrame: Discipline-level gender statistics
    """
    # Detect discipline column
    disc_col = detect_discipline_column(df)

    if disc_col is None:
        raise ValueError("Could not detect discipline column in Paris 2024 data")

    # Handle case where discipline is stored as list string
    if df[disc_col].dtype == object and df[disc_col].str.startswith('[').any():
        # Parse the list and take first discipline
        def parse_discipline(x):
            if pd.isna(x):
                return None
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed[0]
                return str(x)
            except (ValueError, SyntaxError):
                return str(x)

        df = df.copy()
        df['discipline_parsed'] = df[disc_col].apply(parse_discipline)
        disc_col = 'discipline_parsed'

    # Detect gender column
    gender_col = None
    for col in ['gender', 'Gender', 'Sex', 'sex']:
        if col in df.columns:
            gender_col = col
            break

    if gender_col is None:
        raise ValueError("Could not detect gender column in Paris 2024 data")

    print(f"Using gender column: '{gender_col}'")

    # Group by discipline and gender
    discipline_gender = df.groupby([disc_col, gender_col]).size().unstack(fill_value=0)

    # Normalize gender values (handle Male/Female vs M/F)
    if 'Male' in discipline_gender.columns:
        discipline_gender = discipline_gender.rename(columns={'Male': 'M', 'Female': 'F'})

    # Ensure both columns exist
    if 'F' not in discipline_gender.columns:
        discipline_gender['F'] = 0
    if 'M' not in discipline_gender.columns:
        discipline_gender['M'] = 0

    # Calculate statistics
    discipline_gender['Total'] = discipline_gender['F'] + discipline_gender['M']
    discipline_gender['Female_Pct'] = discipline_gender['F'] / discipline_gender['Total'] * 100
    discipline_gender['Deviation_From_50'] = discipline_gender['Female_Pct'] - 50

    discipline_gender = discipline_gender.reset_index()
    discipline_gender = discipline_gender.rename(columns={disc_col: 'Discipline'})

    # Sort by absolute deviation
    discipline_gender['Abs_Deviation'] = abs(discipline_gender['Deviation_From_50'])
    discipline_gender = discipline_gender.sort_values('Abs_Deviation', ascending=False)

    print(f"Analyzed {len(discipline_gender)} disciplines for Paris 2024")

    return discipline_gender


def plot_paris2024_parity(discipline_data, output_path, top_n=20):
    """
    Create horizontal bar chart showing discipline-level parity at Paris 2024.

    Args:
        discipline_data: DataFrame with discipline-level gender statistics
        output_path: Path to save the visualization
        top_n: Number of disciplines to show
    """
    # Take top N disciplines by participation
    plot_data = discipline_data.nlargest(top_n, 'Total').copy()
    plot_data = plot_data.sort_values('Female_Pct')

    plt.figure(figsize=(12, 10))

    # Create color based on deviation from 50%
    colors = ['#E91E63' if x < 50 else '#2196F3' for x in plot_data['Female_Pct']]

    # Create horizontal bar chart
    bars = plt.barh(plot_data['Discipline'], plot_data['Female_Pct'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add 50% reference line
    plt.axvline(x=50, color='#4CAF50', linestyle='--', linewidth=2,
                label='50% Parity')

    # Add value labels
    for i, (pct, dev) in enumerate(zip(plot_data['Female_Pct'], plot_data['Deviation_From_50'])):
        label = f'{pct:.1f}%'
        x_pos = pct + 1 if pct < 90 else pct - 5
        plt.text(x_pos, i, label, va='center', fontsize=9)

    # Styling
    plt.xlabel('Female Participation (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Discipline', fontsize=12, fontweight='bold')
    plt.title('Gender Parity by Discipline - Paris 2024 Olympics',
              fontsize=14, fontweight='bold', pad=20)

    plt.xlim(0, 105)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='x')

    # Add subtitle
    plt.figtext(0.5, 0.02,
                'Pink bars: Male-dominated | Blue bars: Female-dominated | Green line: 50% parity',
                ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run Q1 Gender Parity Analysis."""
    print("=" * 60)
    print("Q1: Gender Parity Analysis")
    print("=" * 60)

    # Ensure visuals directory exists
    os.makedirs(VISUALS_DIR, exist_ok=True)

    # Part 1: Historical trend (1896-2016)
    print("\n--- Part 1: Historical Gender Trend (1896-2016) ---")
    hist_data = load_historical_data()
    gender_trend = analyze_historical_gender_trend(hist_data)

    timeline_path = os.path.join(VISUALS_DIR, "q1_gender_timeline.png")
    plot_historical_gender_timeline(gender_trend, timeline_path)

    # Part 2: Paris 2024 snapshot
    print("\n--- Part 2: Paris 2024 Discipline Parity ---")
    paris_data = load_paris2024_data()
    discipline_parity = analyze_paris2024_parity(paris_data)

    parity_path = os.path.join(VISUALS_DIR, "q1_paris2024_parity.png")
    plot_paris2024_parity(discipline_parity, parity_path)

    # Print summary metrics
    print("\n--- Summary Metrics ---")
    print(f"Latest historical female ratio (2016): {gender_trend.iloc[-1]['Female_Ratio']:.2%}")

    # Top 5 disciplines farthest from parity
    print("\nTop 5 disciplines farthest from 50% parity (Paris 2024):")
    for _, row in discipline_parity.head(5).iterrows():
        direction = "male-dominated" if row['Female_Pct'] < 50 else "female-dominated"
        print(f"  - {row['Discipline']}: {row['Female_Pct']:.1f}% female ({direction})")

    # Overall Paris 2024 ratio
    total_f = discipline_parity['F'].sum()
    total_m = discipline_parity['M'].sum()
    overall_ratio = total_f / (total_f + total_m)
    print(f"\nOverall Paris 2024 female ratio: {overall_ratio:.2%}")

    return {
        "historical_trend": gender_trend,
        "paris2024_parity": discipline_parity,
        "final_female_ratio_2016": gender_trend.iloc[-1]['Female_Ratio'],
        "paris2024_overall_female_ratio": overall_ratio
    }


if __name__ == "__main__":
    main()

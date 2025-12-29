# Olympics Data Science Project

## Data and The City - IE 421 Term Project

Istanbul Bilgi University | Fall 2024-2025

---

## Team Members

| Name | Student ID |
|------|------------|
| Mehmet Fatih Cetinkaya | 121205031 |
| Mehmet Tolga Cakan | 121203029 |
| Demet Irem Yilmaz | 121203095 |
| Sila Kahya | 123203018 |
| Eylul Balci | 122203020 |

---

## Research Questions

### Q1: Gender Parity Analysis (Descriptive)
**Question:** To what extent has the gender participation gap converged towards parity from 1896 to 2016, and what is the deviation from 50/50 in Paris 2024 disciplines?

- **Data:** Historical trend (1896-2016) + Paris 2024 discipline-level snapshot
- **Method:** Descriptive statistics and visualization
- **Output:** 2 visualizations

### Q2: Medal Prediction (Regression)
**Question:** How accurately can a Multiple Linear Regression model predict total medal counts?

- **Model:** Multiple Linear Regression
- **Features:** delegation_size, prev_medals, rolling_avg_medals
- **Training:** 1960-2012
- **Validation:** 2016 Olympics
- **Scope:** Focus on Top-20 NOCs by medal count for detailed evaluation
- **Metrics:** RMSE, MAE, R² reported for both all NOCs and Top-20 subset
- **Output:** 1 visualization + model metrics

### Q3: Athlete Classification (Classification)
**Question:** Do biometric features predict medal success in high-physicality sports?

- **Model:** Logistic Regression (class_weight='balanced')
- **Features:** Age, Height, Weight, Sex
- **Scope:** Post-2000 Summer Olympics, athletes with complete biometrics only
- **Sports:** Athletics, Swimming, Wrestling, Boxing, Weightlifting (high-physicality)
- **Metrics:** F1 and ROC-AUC (primary), Accuracy (secondary)
- **Threshold Tuning:** Scan 0.05-0.95 to find optimal F1 threshold
- **Output:** 1 visualization (ROC + Confusion Matrix)

---

## Project Structure

```
IE421_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── athlete_events.csv    # Historical data (1896-2016)
│   │   └── athletes.csv          # Paris 2024 data
│   └── processed/
├── scripts/
│   ├── config.py                 # Configuration settings
│   ├── data_loader.py            # Data loading utilities
│   ├── q1_gender_parity.py       # Q1 analysis
│   ├── q2_medal_prediction.py    # Q2 regression model
│   ├── q3_athlete_classification.py  # Q3 classification model
│   └── generate_all.py           # Master runner script
├── visuals/
│   ├── q1_gender_timeline.png
│   ├── q1_paris2024_parity.png
│   ├── q2_prediction_scatter.png
│   └── q3_classification_results.png
└── docs/
    ├── index.html                # GitHub Pages home
    ├── requirements.html         # Analysis & results
    └── style.css                 # Website styling
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Analyses
```bash
python scripts/generate_all.py
```

This will:
- Execute Q1 Gender Parity Analysis
- Execute Q2 Medal Prediction
- Execute Q3 Athlete Classification
- Generate all 4 visualizations in `visuals/`

### 3. Run Individual Scripts
```bash
python scripts/q1_gender_parity.py
python scripts/q2_medal_prediction.py
python scripts/q3_athlete_classification.py
```

---

## Data Sources

1. **120 Years of Olympic History: Athletes and Results**
   - Source: [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
   - Records: 271,116 athlete-event records (1896-2016)

2. **Paris 2024 Olympic Summer Games**
   - Source: [Kaggle](https://www.kaggle.com/datasets/piterfm/paris-2024-olympic-summer-games)
   - Records: 11,113 athlete registrations

---

## Website

GitHub Pages: https://bilgi-ie-421.github.io/ie421-2025-2026-1-termproject-data-and-the-city/

---

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- scikit-learn >= 1.1.0

---

## License

This project is for educational purposes as part of IE 421 Data Science course at Istanbul Bilgi University.

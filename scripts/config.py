"""
Configuration settings for IE421 Olympics Data Science Project.
"""
import os

# Get the project root directory (parent of scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_RAW = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
VISUALS_DIR = os.path.join(ROOT_DIR, "visuals")

# Data files
HISTORICAL_DATA = os.path.join(DATA_RAW, "athlete_events.csv")
PARIS2024_DATA = os.path.join(DATA_RAW, "athletes.csv")

# Q1: Gender Parity Analysis
SUMMER_ONLY = True  # Filter to Summer Olympics only

# Q2: Medal Prediction (Regression)
TRAIN_YEARS = (1960, 2012)  # Training period: 1960-2012
VALID_YEAR = 2016  # Validation year

# Q3: Athlete Classification
MIN_YEAR = 2000  # Only use data from 2000 onwards (better biometric completeness)
SELECTED_SPORTS = [
    "Athletics",
    "Swimming",
    "Wrestling",
    "Boxing",
    "Weightlifting"
]

# Visualization settings
FIGURE_DPI = 200
FIGURE_SIZE = (12, 8)

# Possible discipline column names in Paris 2024 data
DISCIPLINE_COLUMNS = ["disciplines", "discipline", "Discipline", "sport", "Sport"]

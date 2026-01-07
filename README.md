# UFC Predictor 

Project inspired by Green Code
 
Video https://www.youtube.com/watch?v=LkJpNLIaeVk&t

Data from https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset

This project explores **machine-learning-based fight outcome prediction** in mixed martial arts (MMA), using historical UFC fight data enriched with a custom **Elo rating system**. The goal is to model relative fighter strength and stylistic advantages while avoiding reliance on betting odds.

The system predicts whether the **Red** or **Blue** corner fighter will win a bout and provides interpretable signals explaining *why* the model favors one fighter over the other.

---

## Project Overview

The pipeline is composed of two main components:

1. **Elo Rating Engine (`elo.py`)**
2. **Supervised Learning Model (`main.py`)**

Together, they transform raw UFC fight records into a structured prediction system that combines long-term fighter skill (Elo) with fight-specific performance and physical attributes.

---

## Elo Rating System

Each fighter is assigned an Elo rating representing their estimated competitive strength.

Key design choices:
- All fighters start at a **base Elo of 1500**
- Ratings are updated **chronologically** to preserve temporal realism
- A higher-than-standard **K-factor (32)** is used to reflect the inherent volatility of MMA

The K-factor is adjusted dynamically:
- **Title fights** carry more weight
- **KO/TKO and submission finishes** are treated as more decisive than decisions

For every fight, the system records:
- `RedElo`
- `BlueElo`
- `EloDifference` (Red − Blue)

These values represent **pre-fight ratings**, preventing post-fight leakage.

---

## Modeling Approach

The predictive model is framed as a **binary classification problem**:
- `1` → Red corner wins
- `0` → Blue corner wins

### Feature Categories

The model draws from multiple feature groups:
- **Performance metrics** (wins, losses, streaks, striking, grappling)
- **Physical attributes** (height, reach, age, weight)
- **Fight context** (title bout, scheduled rounds)
- **Differential features** (Red − Blue comparisons)
- **Ranking indicators**
- **Elo ratings**

Betting odds are intentionally excluded to ensure the model relies solely on **observable competitive factors**.

Missing values are handled using **K-nearest-neighbors imputation**, which better preserves feature relationships than mean or median filling.

---

## Model Architecture

- Gradient-boosted decision trees via **XGBoost**
- Wrapped in a preprocessing pipeline with standardization
- Evaluated using accuracy, cross-validation, confusion matrix, and ROC-AUC
- Feature importance analysis is used to inspect model behavior

---

## Interpretability

Beyond win probabilities, the project includes an explanation layer that identifies **key advantages** for each fighter. These are derived from large differentials in areas such as:
- Reach or height
- Age
- Win streaks
- Finishing ability
- Elo rating gaps

This provides human-readable insight into *why* the model favors one fighter.

---

## Outputs

The project produces:
- A trained prediction model
- Per-fighter Elo rankings
- A fight-level dataset enriched with Elo features
- Visual diagnostics (ROC curve, confusion matrix, feature importances, correlations)

---

## Motivation & Future Work

This project was built to explore:
- How traditional rating systems (Elo) interact with modern ML models
- Feature-driven fight prediction without market bias
- Interpretability in sports analytics

Planned improvements include:
- Time-based train/test splits to reduce temporal leakage
- Automated data refresh and retraining
- Probability calibration
- More granular Elo adjustments by weight class

Wish List Improvements:
- New more in depth fighter data collected such as punch volume and fighter styles using some sort of CV pipeline.
---

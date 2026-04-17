# FIFA World Cup 2026 — Match Outcome Predictor ⚽

A machine learning project that predicts international football match outcomes (Win / Draw / Loss) using historical match data. Built around the FIFA World Cup 2026 context, with a deployed Streamlit app where you can pick any two nations and get win probabilities.

---

## Live App

🔗 **[⚽ FIFA World Cup 2026 — Match Outcome Predictor
](https://swayam0804-fifa-2026-match-outcome-predictor-app-ap0blr.streamlit.app/)**

---

## What It Does

- Trains on 10,000+ international matches from 2010 to 2025
- Computes team-specific features: recent form, goals per game, head-to-head record
- Compares Logistic Regression vs Random Forest via 5-fold cross-validation
- Tunes the best model with GridSearchCV
- Deploys as an interactive Streamlit app — pick two teams, get outcome probabilities + H2H history + recent form

---

## Features Used

| Feature                    | Description                                    |
|----------------------------|------------------------------------------------|
| Win rate (last 10 matches) | Recent form for each team                      |
| Goals scored per game      | Attacking strength                             |
| Goals conceded per game    | Defensive strength                             |
| Head-to-head win rate      | Historical matchup record                      |
| Form differential          | Team 1 win rate minus Team 2 win rate          |
| Tournament importance      | World Cup > Continental > Qualifier > Friendly |
| Neutral ground flag        | Whether the match is on neutral territory      |

---

## Model Results

| Model               | CV Accuracy |
|---------------------|-------------|
| Logistic Regression | ~0.62       |
| Random Forest       | ~0.60       |

*Football prediction is inherently hard due to randomness. 55–60% accuracy is consistent with published sports prediction literature. The value here is in calibrated probability estimates, not just the single predicted outcome.*

---

## Project Structure

```
fifa-wc-2026-predictor/
├── train_model.py        ← run this first
├── app.py                ← streamlit app
├── requirements.txt
├── all_matches.csv           ← download from Kaggle (link below)
├── artifacts/            ← auto-created after training
│   ├── model.pkl
│   ├── label_encoder.pkl
│   ├── feature_cols.pkl
│   ├── all_teams.pkl
│   ├── features_df.csv
│   └── results_clean.csv
│   └── scaler.pkl
└── plots/                ← 6 EDA and evaluation plots
```

---

## Setup

**1. Get the dataset**
Download `all_matches.csv` from Kaggle:
[All International Football Results](https://www.kaggle.com/datasets/patateriedata/all-international-football-results)

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train_model.py
```
Takes around 5–10 minutes. Creates `artifacts/` and `plots/` folders automatically.

**4. Run locally**
```bash
streamlit run app.py
```

**5. Deploy free on Streamlit Cloud**
Push everything including `artifacts/` to GitHub → go to [share.streamlit.io](https://share.streamlit.io) → connect repo → set main file to `app.py` → Deploy.

---

## Tech Stack

| Tool                 | Use                                    |
|----------------------|----------------------------------------|
| Pandas / NumPy       | Data loading and feature engineering   |
| Matplotlib / Seaborn | EDA and evaluation plots               |
| Scikit-learn         | Models, GridSearchCV, cross-validation |
| Joblib               | Saving model artifacts                 |
| Streamlit            | Interactive app and deployment         |

---

## Data Source

[All International Football Results](https://www.kaggle.com/datasets/patateriedata/all-international-football-results)
~50,000 international match results with scores, tournament type, and neutral ground flag.

# FIFA World Cup 2026 Predictor ⚽

A machine learning project that predicts international football match outcomes for the 2026 FIFA World Cup. Built with a Random Forest classifier trained on 10,000+ historical international matches, deployed as a multi-tab Streamlit app.

---

## Live App

🔗 **[FIFA World Cup 2026 Predictor
](https://swayam0804-fifa-2026-match-outcome-predictor-app-ap0blr.streamlit.app)**

---

## What It Does

The app has **3 tabs:**

### 🏟️ Groups
- Displays all 48 qualified teams across Groups A–L (official December 2025 draw)
- Marks host nations 🏠 and World Cup debutants ⭐
- Explains the new 2026 format — Round of 32, wild card spots, 104 matches

### ⚔️ Match Predictor
- Pick any two of the 48 qualified teams
- Get Win / Draw / Loss probabilities powered by the ML model
- Shows recent form (last 10 matches), goals per game, and head-to-head history

### 🏆 Tournament Simulator
- Full stage-by-stage manual bracket simulation
- **Group Stage** → pick top 2 from each group + 8 wild card 3rd-place teams
- **Round of 32** → 16 matches, pick each winner
- **Round of 16** → 8 matches
- **Quarter Finals** → 4 matches
- **Semi Finals** → 2 matches + 3rd place match
- **Final** → pick the World Cup Champion
- Model win probabilities shown alongside every matchup to guide your picks

---

## Official 2026 Group Draw

| Group  | Teams                                                  |
|--------|--------------------------------------------------------|
| A      | Mexico 🏠, South Africa, South Korea, Czechia          |
| B      | Canada 🏠, Bosnia and Herzegovina, Qatar, Switzerland  |
| C      | Brazil, Morocco, Haiti, Scotland                       |
| D      | USA 🏠, Paraguay, Australia, Turkiye                   |
| E      | Germany, Curacao ⭐, Ivory Coast, Ecuador               |
| F      | Netherlands, Japan, Sweden, Tunisia                    |
| G      | Belgium, Egypt, Iran, New Zealand                      |
| H      | Spain, Cape Verde ⭐, Saudi Arabia, Uruguay             |
| I      | France, Senegal, Norway, Iraq                          |
| J      | Argentina, Algeria, Austria, Jordan ⭐                  |
| K      | Portugal, DR Congo, Uzbekistan ⭐, Colombia             |
| L      | England, Croatia, Ghana, Panama                        |

🏠 Host nation · ⭐ World Cup debut

---

## Model

**Algorithm:** Random Forest Classifier — tuned via GridSearchCV, 5-fold cross-validation

**Comparison done:** Logistic Regression vs Random Forest via 5-fold CV (Random Forest won)

**Training data:** 10,000+ international matches (2010–2024)

**Features used:**

| Feature                    | Description                                    |
|----------------------------|------------------------------------------------|
| Win rate (last 10 matches) | Recent form for each team                      |
| Goals scored per game      | Attacking strength                             |
| Goals conceded per game    | Defensive strength                             |
| Head-to-head win rate      | Historical matchup record                      |
| Form differential          | Team 1 win rate minus Team 2 win rate          |
| Tournament importance      | World Cup > Continental > Qualifier > Friendly |
| Neutral ground flag        | All World Cup matches are on neutral ground    |

**CV Accuracy:** ~57% — football is inherently unpredictable. The value is in calibrated probability estimates across outcomes, not just the predicted winner.

---

## Project Structure

```
fifa-wc-2026-predictor/
├── train_model.py        ← run this first to train and save the model
├── app.py                ← 3-tab streamlit app
├── requirements.txt
├── results.csv           ← download from Kaggle (link below)
├── artifacts/            ← auto-created after training
│   ├── model.pkl
│   ├── label_encoder.pkl
│   ├── feature_cols.pkl
│   ├── all_teams.pkl
│   ├── features_df.csv
│   └── results_clean.csv
└── plots/                ← 6 EDA and evaluation plots saved here
```

---

## Setup

**1. Get the dataset**
Download `all_matches.csv` from Kaggle:
[International football results 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train_model.py
```
Takes ~5–10 minutes. Creates `artifacts/` and `plots/` automatically.

**4. Run locally**
```bash
streamlit run app.py
```

**5. Deploy free on Streamlit Cloud**
Push everything including `artifacts/` to GitHub → go to [share.streamlit.io](https://share.streamlit.io) → connect repo → set main file to `app.py` → Deploy.

After any update, just `git push` and Streamlit Cloud auto-redeploys within 1–2 minutes.

---

## Tech Stack

| Tool                 | Use                                                                |
|----------------------|--------------------------------------------------------------------|
| Pandas / NumPy       | Data loading and feature engineering                               |
| Matplotlib / Seaborn | EDA and evaluation plots                                           |
| Scikit-learn         | Logistic Regression, Random Forest, GridSearchCV, cross-validation |
| Joblib               | Saving and loading model artifacts                                 |
| Streamlit            | 3-tab interactive app and free cloud deployment                    |

---

## Data Source

[All International Football Results](https://www.kaggle.com/datasets/patateriedata/all-international-football-results)
~50,000 international match results including scores, teams, tournament type, and neutral ground flag. Updated through 31 March 2026.

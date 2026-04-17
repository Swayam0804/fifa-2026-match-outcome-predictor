#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# In[98]:


warnings.filterwarnings("ignore")

if not os.path.exists("plots"):
    os.makedirs("plots")
if not os.path.exists("artifacts"):
    os.makedirs("artifacts")

# Load the data

# In[38]:


df = pd.read_csv("all_matches.csv")

print("shape:", df.shape)
print(df.head())

# In[39]:


print("Columns:\n", df.columns.tolist())

# # Basic Cleaning

# Converting date column to datetime

# In[40]:


df["date"] = pd.to_datetime(df["date"])

# Only keeping matches from 1990 onwards - older data is less relevant for modern football

# In[41]:


df = df[df["date"] >= "1990-01-01"].reset_index(drop=True)

# In[42]:


print(f"Matches after 1990: {len(df)}")

# Adding the "outcome" column - this is our target from the home team's perspective

# In[43]:


def get_outcome(row):
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    else:
        return "Draw"


# In[44]:


df["outcome"] = df.apply(get_outcome, axis=1)

# In[45]:


print("outcome distribution:")
print(df["outcome"].value_counts())

# # EDA Plots

# Outcome distribution

# In[46]:


plt.figure(figsize=(7, 4))
colors = ["#5B8DD9", "#F4A85D", "#E06D6D"]
df["outcome"].value_counts().plot(kind="bar", color=colors, edgecolor="white")
plt.title("Match Outcome Distribution (1990-present)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/Match_outcome_distribution_(1990-present).png", dpi=150)
plt.show()

# Goal Distribution

# In[47]:


plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["home_score"], bins=10, color="#5B8DD9", kde=True)
plt.title("Home Goals Distribution")
plt.xlabel("Goals")

plt.subplot(1, 2, 2)
sns.histplot(df["away_score"], bins=10, color="#E06D6D", kde=True)
plt.title("Away Goals Distribution")
plt.xlabel("Goals")

plt.tight_layout()
plt.savefig("plots/Home_Goals_Distribution.png", dpi=150)
plt.show()

# Tournament Type Breakdown

# In[48]:


plt.figure(figsize=(10, 5))
top_tournaments = df["tournament"].value_counts().head(10)
top_tournaments.plot(kind="barh", color="#5BC08D", edgecolor="white")
plt.title("Top 10 Tournament Types in Dataset")
plt.xlabel("Number of Matches")
plt.tight_layout()
plt.savefig("plots/Tournament_types.png", dpi=150)
plt.show()

print("EDA plots saved")


# # Feature Engineering

# Sorting by date so we can compute rolling stats correctly

# In[49]:


df = df.sort_values("date").reset_index(drop=True)

# Function to compute a team's recent form (last N matches)
# which returns win rate, goals scored per game, goals conceded per game

# In[50]:


def compute_team_stats(team_name, before_date, all_matches, n=10):
    team_matches = all_matches[
        ((all_matches["home_team"] == team_name) | (all_matches["away_team"] == team_name)) &
        (all_matches["date"] < before_date)].tail(n)

    if len(team_matches) == 0:
        return 0.5, 1.0, 1.0  # default values if no history

    wins = 0
    goals_scored = 0
    goals_conceded = 0

    for _, row in team_matches.iterrows():
        if row["home_team"] == team_name:
            goals_scored += row["home_score"]
            goals_conceded += row["away_score"]
            if row["home_score"] > row["away_score"]:
                wins += 1
        else:
            goals_scored += row["away_score"]
            goals_conceded += row["home_score"]
            if row["away_score"] > row["home_score"]:
                wins += 1

    n_matches = len(team_matches)
    return wins / n_matches, goals_scored / n_matches, goals_conceded / n_matches

# Function to compute head to head record between two teams

# In[51]:


def compute_h2h(home_team, away_team, before_date, all_matches, n=10):
    h2h = all_matches[
        (
            ((all_matches["home_team"] == home_team) & (all_matches["away_team"] == away_team)) |
            ((all_matches["home_team"] == away_team) & (all_matches["away_team"] == home_team))
        ) &
        (all_matches["date"] < before_date)
    ].tail(n)

    if len(h2h) == 0:
        return 0.5  # no history, assume equal

    home_wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team and row["home_score"] > row["away_score"]:
            home_wins += 1
        elif row["away_team"] == home_team and row["away_score"] > row["home_score"]:
            home_wins += 1

    return home_wins / len(h2h)

# Tournament importance encoding world cup and continental tournaments matter more than friendlies

# In[52]:


def encode_tournament(tournament):
    if "FIFA World Cup" in tournament:
        return 3
    elif any(t in tournament for t in ["UEFA Euro", "Copa America", "Africa Cup", "Asian Cup"]):
        return 2
    elif "qualification" in tournament.lower() or "qualifier" in tournament.lower():
        return 1
    else:
        return 0  # friendlies and others


print("\nbuilding features... this takes a few minutes")

# Build features for each match using a sample for speed - using matches from 2010 onwards for training this still gives us 10,000+ matches which is plenty

# In[53]:


df_train = df[df["date"] >= "2010-01-01"].reset_index(drop=True)

# In[54]:


print(f"matches used for feature building: {len(df_train)}")

# In[55]:


rows = []
for idx, row in df_train.iterrows():
    if idx % 1000 == 0:
        print(f"  processing match {idx}/{len(df_train)}...")

    home_wr, home_gf, home_ga = compute_team_stats(row["home_team"], row["date"], df)
    away_wr, away_gf, away_ga = compute_team_stats(row["away_team"], row["date"], df)
    h2h_rate = compute_h2h(row["home_team"], row["away_team"], row["date"], df)
    tournament_imp = encode_tournament(row["tournament"])
    is_neutral = 1 if row["neutral"] else 0

    rows.append({
        "home_win_rate":        home_wr,
        "away_win_rate":        away_wr,
        "home_goals_per_game":  home_gf,
        "away_goals_per_game":  away_gf,
        "home_conceded_per_game": home_ga,
        "away_conceded_per_game": away_ga,
        "h2h_home_win_rate":    h2h_rate,
        "form_diff":            home_wr - away_wr,  # simple diff feature
        "goals_diff":           home_gf - away_gf,
        "tournament_importance": tournament_imp,
        "is_neutral":           is_neutral,
        "outcome":              row["outcome"],
        "home_team":            row["home_team"],
        "away_team":            row["away_team"],
        "date":                 row["date"]
    })

# In[57]:


features_df = pd.DataFrame(rows)
print(f"Feature dataset shape: {features_df.shape}")

# In[58]:


print(features_df.head())

# # Prepare for modelling

# In[59]:


feature_cols = [
    "home_win_rate", "away_win_rate",
    "home_goals_per_game", "away_goals_per_game",
    "home_conceded_per_game", "away_conceded_per_game",
    "h2h_home_win_rate", "form_diff", "goals_diff",
    "tournament_importance", "is_neutral"
]

# In[60]:


X = features_df[feature_cols].values
y = features_df["outcome"].values

# Encode labels

# In[67]:


le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

# # Model comparison

# In[68]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Comparing models with 5-fold CV...")

# Logistic Regression

# In[65]:


lr = LogisticRegression(max_iter=1000, random_state=42)
lr_scores = cross_val_score(lr, X_scaled, y_encoded, cv=kf, scoring="accuracy")

# In[69]:


print(f"Logistic Regression: accuracy = {round(lr_scores.mean(), 4)} +/- {round(lr_scores.std(), 4)}")

# Random Forest (Doesn't need scaling)

# In[70]:


rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_scores = cross_val_score(rf, X, y_encoded, cv=kf, scoring="accuracy")

# In[73]:


print(f"Random Forest: \nAccuracy = {round(rf_scores.mean(), 4)} +/- {round(rf_scores.std(), 4)}")

# Plot Comparison

# In[77]:


plt.figure(figsize=(7, 4))
model_names = ["Logistic Regression", "Random Forest"]
accuracies = [lr_scores.mean(), rf_scores.mean()]
stds = [lr_scores.std(), rf_scores.std()]
bars = plt.bar(model_names, accuracies, yerr=stds, capsize=5,
               color=["#5B8DD9", "#5BC08D"], edgecolor="white")

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.002,
             str(round(acc, 4)), ha="center", va="bottom", fontsize=11)

plt.title("Model Comparison - 5 Fold CV Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/Model_comparison.png", dpi=150)
plt.show()

# Picking the best model

# In[78]:


if rf_scores.mean() >= lr_scores.mean():
    best_model_name = "Random Forest"
    print("Best model: Random Forest")
else:
    best_model_name = "Logistic Regression"
    print("Best model: Logistic Regression")

# # Tune Random Forest

# In[80]:


print("Tuning Random Forest with GridSearchCV...")

# In[81]:


param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_base, param_grid, cv=kf,
                           scoring="accuracy", n_jobs=-1, verbose=1)
grid_search.fit(X, y_encoded)

# In[82]:


print(f"Best params: {grid_search.best_params_}")
print(f"Best CV accuracy: {round(grid_search.best_score_, 4)}")

tuned_rf = grid_search.best_estimator_

# # Evaluation on full data

# In[83]:


tuned_rf.fit(X, y_encoded)
y_pred = tuned_rf.predict(X)

# In[85]:


print("Classification report:")
print(classification_report(y_encoded, y_pred, target_names=le.classes_))

# Confusion Matrix

# In[86]:


fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix - Tuned Random Forest")
plt.tight_layout()
plt.savefig("plots/Confusion_matrix.png", dpi=150)
plt.show()

# Feature Importance

# In[87]:


importances = pd.Series(tuned_rf.feature_importances_, index=feature_cols).sort_values()

# In[88]:


plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#5B8DD9")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/Feature_importance.png", dpi=150)
plt.show()

# In[89]:


print("Top 3 most important features:")
print(importances.sort_values(ascending=False).head(3))

# # Save everything for the streamlit app

# Getting all unique teams in the dataset for the app dropdown

# In[90]:


all_teams = sorted(list(set(df["home_team"].unique().tolist() + df["away_team"].unique().tolist())))

joblib.dump(tuned_rf, "artifacts/model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(le, "artifacts/label_encoder.pkl")
joblib.dump(feature_cols, "artifacts/feature_cols.pkl")
joblib.dump(all_teams, "artifacts/all_teams.pkl")
features_df.to_csv("artifacts/features_df.csv", index=False)

# Also saving the full historical results for H2H lookup in the app

# In[92]:


df.to_csv("artifacts/results_clean.csv", index=False)

print("Artifacts saved:")
print("  artifacts/model.pkl")
print("  artifacts/scaler.pkl")
print("  artifacts/label_encoder.pkl")
print("  artifacts/feature_cols.pkl")
print("  artifacts/all_teams.pkl")
print("  artifacts/features_df.csv")
print("  artifacts/results_clean.csv")

# # Summary

# In[96]:


print("Summary:\n")
print(f"Training matches used: {len(features_df)}")
print(f"Best model: {best_model_name}")
print(f"Best CV Accuracy: {round(grid_search.best_score_, 4)}")
print(f"Total teams in dataset: {len(all_teams)}")
print("Plots saved to plots/")


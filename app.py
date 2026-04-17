import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FIFA 2026 Match Predictor",
    page_icon="⚽",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model        = joblib.load("artifacts/model.pkl")
    le           = joblib.load("artifacts/label_encoder.pkl")
    feature_cols = joblib.load("artifacts/feature_cols.pkl")
    all_teams    = joblib.load("artifacts/all_teams.pkl")
    return model, le, feature_cols, all_teams

@st.cache_data
def load_data():
    return pd.read_csv("artifacts/results_clean.csv", parse_dates=["date"])

model, le, feature_cols, all_teams = load_artifacts()
df = load_data()


# ── helpers ────────────────────────────────────────────────────────────────────

def get_recent_form(team, data, n=10):
    ref = pd.Timestamp("2024-12-31")
    matches = data[
        ((data["home_team"] == team) | (data["away_team"] == team)) &
        (data["date"] < ref)
    ].tail(n)
    if len(matches) == 0:
        return 0.5, 1.0, 1.0
    wins, gf, ga = 0, 0, 0
    for _, row in matches.iterrows():
        if row["home_team"] == team:
            gf += row["home_score"]
            ga += row["away_score"]
            if row["home_score"] > row["away_score"]:
                wins += 1
        else:
            gf += row["away_score"]
            ga += row["home_score"]
            if row["away_score"] > row["home_score"]:
                wins += 1
    n_m = len(matches)
    return wins / n_m, gf / n_m, ga / n_m

def get_h2h(t1, t2, data, n=10):
    return data[
        ((data["home_team"] == t1) & (data["away_team"] == t2)) |
        ((data["home_team"] == t2) & (data["away_team"] == t1))
    ].tail(n)

def h2h_win_rate(t1, t2, data, n=10):
    h = get_h2h(t1, t2, data, n)
    if len(h) == 0:
        return 0.5
    wins = 0
    for _, row in h.iterrows():
        if row["home_team"] == t1 and row["home_score"] > row["away_score"]:
            wins += 1
        elif row["away_team"] == t1 and row["away_score"] > row["home_score"]:
            wins += 1
    return wins / len(h)

def encode_tournament(t):
    if "FIFA World Cup" in t:
        return 3
    if any(x in t for x in ["UEFA Euro", "Copa America", "Africa Cup", "Asian Cup"]):
        return 2
    if "qualif" in t.lower():
        return 1
    return 0

def predict(t1, t2, tournament, neutral, data):
    wr1, gf1, ga1 = get_recent_form(t1, data)
    wr2, gf2, ga2 = get_recent_form(t2, data)
    h2h = h2h_win_rate(t1, t2, data)
    feats = np.array([[
        wr1, wr2,
        gf1, gf2,
        ga1, ga2,
        h2h,
        wr1 - wr2,
        gf1 - gf2,
        encode_tournament(tournament),
        1 if neutral else 0
    ]])
    probs     = model.predict_proba(feats)[0]
    classes   = le.classes_
    result    = {c: round(p * 100, 1) for c, p in zip(classes, probs)}
    predicted = classes[np.argmax(probs)]
    return result, predicted, wr1, wr2, gf1, gf2


# ── ui ─────────────────────────────────────────────────────────────────────────

st.title("⚽ FIFA World Cup 2026 — Match Outcome Predictor")
st.markdown(
    "Pick two international teams, set the match context, and get "
    "**Win / Draw / Loss** probabilities based on historical form and head-to-head data."
)
st.divider()

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    t1_default = all_teams.index("Portugal") if "Portugal" in all_teams else 0
    team1 = st.selectbox("🏳️ Team 1", all_teams, index=t1_default)

with col2:
    st.markdown(
        "<br><br><p style='text-align:center;font-size:26px;font-weight:bold'>VS</p>",
        unsafe_allow_html=True
    )

with col3:
    t2_default = all_teams.index("Argentina") if "Argentina" in all_teams else 1
    team2 = st.selectbox("🏳️ Team 2", all_teams, index=t2_default)

col4, col5 = st.columns(2)
with col4:
    tournament = st.selectbox("Tournament Type", [
        "FIFA World Cup",
        "UEFA Euro qualification",
        "Copa America",
        "Friendly",
        "Other qualification"
    ])
with col5:
    is_neutral = st.checkbox("Neutral Ground?", value=True)

st.divider()

if team1 == team2:
    st.warning("Please select two different teams.")
else:
    if st.button("⚽ Predict Outcome", use_container_width=True):

        result, predicted, wr1, wr2, gf1, gf2 = predict(
            team1, team2, tournament, is_neutral, df
        )

        hw = result.get("Home Win", 0)
        d  = result.get("Draw", 0)
        aw = result.get("Away Win", 0)

        st.subheader(f"Prediction: {team1} vs {team2}")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"🏆 {team1} Win", f"{hw}%")
        c2.metric("🤝 Draw", f"{d}%")
        c3.metric(f"🏆 {team2} Win", f"{aw}%")

        if predicted == "Home Win":
            st.success(f"✅ Most likely: **{team1} wins** ({hw}%)")
        elif predicted == "Away Win":
            st.success(f"✅ Most likely: **{team2} wins** ({aw}%)")
        else:
            st.info(f"🤝 Most likely: **Draw** ({d}%)")

        # probability bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        labels = [f"{team1} Win", "Draw", f"{team2} Win"]
        probs  = [hw, d, aw]
        colors = ["#5B8DD9", "#F4A85D", "#E06D6D"]
        bars = ax.barh(labels, probs, color=colors, height=0.5, edgecolor="white")
        for bar, p in zip(bars, probs):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{p}%", va="center", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 110)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Match Outcome Probabilities")
        ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

        # recent form
        st.subheader("📊 Recent Form — Last 10 Matches")
        f1, f2 = st.columns(2)
        with f1:
            st.markdown(f"**{team1}**")
            st.metric("Win Rate", f"{round(wr1 * 100, 1)}%")
            st.metric("Goals per Game", round(gf1, 2))
        with f2:
            st.markdown(f"**{team2}**")
            st.metric("Win Rate", f"{round(wr2 * 100, 1)}%")
            st.metric("Goals per Game", round(gf2, 2))

        st.divider()

        # h2h table
        st.subheader(f"🔁 Head-to-Head: {team1} vs {team2}")
        h2h_df = get_h2h(team1, team2, df)

        if len(h2h_df) == 0:
            st.info("No head-to-head matches found in the dataset.")
        else:
            display = h2h_df[[
                "date", "home_team", "home_score", "away_score", "away_team", "tournament"
            ]].copy()
            display["date"]  = display["date"].dt.strftime("%Y-%m-%d")
            display["Score"] = (
                display["home_score"].astype(int).astype(str) + " – " +
                display["away_score"].astype(int).astype(str)
            )
            display = display[["date", "home_team", "Score", "away_team", "tournament"]]
            display.columns = ["Date", "Home Team", "Score", "Away Team", "Tournament"]
            st.dataframe(
                display.sort_values("Date", ascending=False).reset_index(drop=True),
                use_container_width=True
            )

        with st.expander("🌍 FIFA World Cup 2026 — Quick Facts"):
            st.markdown("""
            - **Hosts:** USA, Canada, Mexico
            - **Teams:** 48 (expanded format)
            - **Total matches:** 104
            - **Dates:** June 11 – July 19, 2026

            *Predictions use historical data up to 2024 and do not factor in
            current injuries, squad selections, or live form.*
            """)

with st.expander("🤖 About the Model"):
    st.markdown("""
    **Model:** Random Forest Classifier — tuned via GridSearchCV, 5-fold cross-validation

    **Training data:** 10,000+ international matches (2010–2024)

    **Features:** Recent win rate, goals per game, goals conceded per game,
    head-to-head win rate, form differential, tournament importance, neutral ground flag

    **Target:** Match outcome from Team 1's perspective — Win, Draw, or Loss

    **GitHub:** [github.com/Swayam0804](https://github.com/Swayam0804)
    """)
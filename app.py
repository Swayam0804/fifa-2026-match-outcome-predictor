import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide"
)

# Official 48 qualified teams grouped A-L 

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkiye"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ALL_48 = sorted([team for teams in GROUPS.values() for team in teams])

# Name mapping for teams that might differ in the dataset
NAME_MAP = {
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Turkiye": "Turkey",
    "Ivory Coast": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Cape Verde": "Cape Verde",
    "USA": "United States",
}

def dataset_name(team):
    return NAME_MAP.get(team, team)


# Load artifacts

@st.cache_resource
def load_artifacts():
    model        = joblib.load("artifacts/model.pkl")
    le           = joblib.load("artifacts/label_encoder.pkl")
    feature_cols = joblib.load("artifacts/feature_cols.pkl")
    return model, le, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv("artifacts/results_clean.csv", parse_dates=["date"])

model, le, feature_cols = load_artifacts()
df = load_data()


# Helper functions

def get_recent_form(team, data, n=10):
    t = dataset_name(team)
    ref = pd.Timestamp("2024-12-31")
    matches = data[
        ((data["home_team"] == t) | (data["away_team"] == t)) &
        (data["date"] < ref)
    ].tail(n)
    if len(matches) == 0:
        return 0.5, 1.0, 1.0
    wins, gf, ga = 0, 0, 0
    for _, row in matches.iterrows():
        if row["home_team"] == t:
            gf += row["home_score"]; ga += row["away_score"]
            if row["home_score"] > row["away_score"]: wins += 1
        else:
            gf += row["away_score"]; ga += row["home_score"]
            if row["away_score"] > row["home_score"]: wins += 1
    n_m = len(matches)
    return wins / n_m, gf / n_m, ga / n_m

def get_h2h(t1, t2, data, n=10):
    d1, d2 = dataset_name(t1), dataset_name(t2)
    return data[
        ((data["home_team"] == d1) & (data["away_team"] == d2)) |
        ((data["home_team"] == d2) & (data["away_team"] == d1))
    ].tail(n)

def h2h_win_rate(t1, t2, data, n=10):
    d1 = dataset_name(t1)
    h  = get_h2h(t1, t2, data, n)
    if len(h) == 0: return 0.5
    wins = 0
    for _, row in h.iterrows():
        if row["home_team"] == d1 and row["home_score"] > row["away_score"]: wins += 1
        elif row["away_team"] == d1 and row["away_score"] > row["home_score"]: wins += 1
    return wins / len(h)

def encode_tournament(t):
    if "FIFA World Cup" in t: return 3
    if any(x in t for x in ["UEFA Euro", "Copa America", "Africa Cup", "Asian Cup"]): return 2
    if "qualif" in t.lower(): return 1
    return 0

def predict_match(t1, t2, data, tournament="FIFA World Cup", neutral=True):
    wr1, gf1, ga1 = get_recent_form(t1, data)
    wr2, gf2, ga2 = get_recent_form(t2, data)
    h2h = h2h_win_rate(t1, t2, data)
    feats = np.array([[
        wr1, wr2, gf1, gf2, ga1, ga2, h2h,
        wr1 - wr2, gf1 - gf2,
        encode_tournament(tournament), 1 if neutral else 0
    ]])
    probs     = model.predict_proba(feats)[0]
    classes   = le.classes_
    result    = {c: round(p * 100, 1) for c, p in zip(classes, probs)}
    predicted = classes[np.argmax(probs)]
    return result, predicted, wr1, wr2, gf1, gf2


# UI tabs

st.title("⚽ FIFA World Cup 2026 Predictor")
st.caption("48 qualified teams · Groups A–L · June 11 – July 19, 2026 · USA, Canada, Mexico")

tab1, tab2, tab3 = st.tabs(["🏟️ Groups", "⚔️ Match Predictor", "🏆 Tournament Simulator"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GROUP STAGE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("FIFA World Cup 2026 — Group Stage Draw")
    st.markdown("All 48 qualified teams across 12 groups. Top 2 from each group + 8 best 3rd-place teams advance to Round of 32.")
    st.divider()

    cols = st.columns(3)
    group_letters = list(GROUPS.keys())

    for i, letter in enumerate(group_letters):
        with cols[i % 3]:
            teams = GROUPS[letter]
            st.markdown(f"**Group {letter}**")
            for team in teams:
                host_tag = " 🏠" if team in ["Mexico", "Canada", "USA"] else ""
                debut_tag = " ⭐" if team in ["Cape Verde", "Curacao", "Jordan", "Uzbekistan"] else ""
                st.markdown(f"• {team}{host_tag}{debut_tag}")
            st.markdown("")

    st.divider()
    st.caption("🏠 Host nation  ·  ⭐ World Cup debut")

    st.markdown("""
    **2026 Format — Key Points**
    - 12 groups of 4 teams each
    - Top 2 from each group → Round of 32 (24 teams)
    - 8 best 3rd-place teams → Round of 32 (8 teams)
    - Total 32 teams in Round of 32 → Round of 16 → QF → SF → Final
    - 104 total matches
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("⚔️ Match Outcome Predictor")
    st.markdown("Pick any two of the 48 qualified teams to get Win / Draw / Loss probabilities.")
    st.divider()

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        t1_idx = ALL_48.index("Portugal") if "Portugal" in ALL_48 else 0
        team1  = st.selectbox("🏳️ Team 1", ALL_48, index=t1_idx, key="pred_t1")
    with col2:
        st.markdown("<br><br><p style='text-align:center;font-size:26px;font-weight:bold'>VS</p>",
                    unsafe_allow_html=True)
    with col3:
        t2_idx = ALL_48.index("Argentina") if "Argentina" in ALL_48 else 1
        team2  = st.selectbox("🏳️ Team 2", ALL_48, index=t2_idx, key="pred_t2")

    col4, col5 = st.columns(2)
    with col4:
        tournament = st.selectbox("Tournament Stage", [
            "FIFA World Cup", "FIFA World Cup qualification"
        ], key="pred_tour")
    with col5:
        is_neutral = st.checkbox("Neutral Ground?", value=True, key="pred_neutral")

    st.divider()

    if team1 == team2:
        st.warning("Select two different teams.")
    else:
        if st.button("⚽ Predict", use_container_width=True, key="pred_btn"):
            result, predicted, wr1, wr2, gf1, gf2 = predict_match(
                team1, team2, df, tournament, is_neutral
            )
            hw = result.get("Home Win", 0)
            d  = result.get("Draw", 0)
            aw = result.get("Away Win", 0)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"🏆 {team1} Win", f"{hw}%")
            c2.metric("🤝 Draw", f"{d}%")
            c3.metric(f"🏆 {team2} Win", f"{aw}%")

            if predicted == "Home Win":
                st.success(f"✅ Model predicts: **{team1} wins** ({hw}%)")
            elif predicted == "Away Win":
                st.success(f"✅ Model predicts: **{team2} wins** ({aw}%)")
            else:
                st.info(f"🤝 Model predicts: **Draw** ({d}%)")

            # bar chart
            fig, ax = plt.subplots(figsize=(8, 3))
            labels = [f"{team1} Win", "Draw", f"{team2} Win"]
            probs  = [hw, d, aw]
            colors = ["#5B8DD9", "#F4A85D", "#E06D6D"]
            bars = ax.barh(labels, probs, color=colors, height=0.5, edgecolor="white")
            for bar, p in zip(bars, probs):
                ax.text(bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{p}%", va="center", fontsize=12, fontweight="bold")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)")
            ax.set_title(f"{team1} vs {team2} — Outcome Probabilities")
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
                st.metric("Win Rate", f"{round(wr1*100,1)}%")
                st.metric("Goals per Game", round(gf1, 2))
            with f2:
                st.markdown(f"**{team2}**")
                st.metric("Win Rate", f"{round(wr2*100,1)}%")
                st.metric("Goals per Game", round(gf2, 2))

            st.divider()

            # h2h
            st.subheader(f"🔁 Head-to-Head: {team1} vs {team2}")
            h2h_df = get_h2h(team1, team2, df)
            if len(h2h_df) == 0:
                st.info("No head-to-head matches found in dataset.")
            else:
                disp = h2h_df[["date","home_team","home_score","away_score","away_team","tournament"]].copy()
                disp["date"]  = disp["date"].dt.strftime("%Y-%m-%d")
                disp["Score"] = disp["home_score"].astype(int).astype(str) + " – " + disp["away_score"].astype(int).astype(str)
                disp = disp[["date","home_team","Score","away_team","tournament"]]
                disp.columns = ["Date","Home","Score","Away","Tournament"]
                st.dataframe(disp.sort_values("Date", ascending=False).reset_index(drop=True),
                             use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOURNAMENT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🏆 Tournament Simulator")
    st.markdown(
        "Pick winners at each stage manually. "
        "Model probabilities shown alongside each matchup to guide your picks."
    )
    st.divider()

    # initialise session state for simulator
    if "stage" not in st.session_state:
        st.session_state.stage = "group"
    if "group_winners" not in st.session_state:
        st.session_state.group_winners = {}   # letter -> [1st, 2nd]
    if "r32_teams" not in st.session_state:
        st.session_state.r32_teams = []
    if "r32_winners" not in st.session_state:
        st.session_state.r32_winners = []
    if "r16_winners" not in st.session_state:
        st.session_state.r16_winners = []
    if "qf_winners" not in st.session_state:
        st.session_state.qf_winners = []
    if "sf_winners" not in st.session_state:
        st.session_state.sf_winners = []
    if "third_place" not in st.session_state:
        st.session_state.third_place = None
    if "champion" not in st.session_state:
        st.session_state.champion = None

    def show_matchup_probs(t1, t2):
        """Show model win probs for a matchup inline."""
        try:
            result, _, _, _, _, _ = predict_match(t1, t2, df)
            hw = result.get("Home Win", 0)
            d  = result.get("Draw", 0)
            aw = result.get("Away Win", 0)
            st.caption(f"📊 Model: {t1} {hw}% · Draw {d}% · {t2} {aw}%")
        except Exception:
            pass

    def reset_simulator():
        for key in ["stage","group_winners","r32_teams","r32_winners",
                    "r16_winners","qf_winners","sf_winners","third_place","champion"]:
            if key in st.session_state:
                del st.session_state[key]

    if st.button("🔄 Reset Simulator", key="reset_btn"):
        reset_simulator()
        st.rerun()

    # ── STAGE: GROUP ──────────────────────────────────────────────────────────
    if st.session_state.stage == "group":
        st.markdown("### Stage 1 — Group Stage")
        st.markdown("Pick the **top 2 teams** that advance from each group. Then pick **8 wild card 3rd-place teams**.")

        group_selections = {}

        for letter, teams in GROUPS.items():
            with st.expander(f"Group {letter}: {' · '.join(teams)}", expanded=True):
                st.markdown(f"**Select top 2 who advance from Group {letter}:**")

                # show model probabilities for each pair in the group
                col_a, col_b = st.columns(2)
                with col_a:
                    first  = st.selectbox(f"1st place", teams,
                                          key=f"g{letter}_1st")
                with col_b:
                    remaining = [t for t in teams if t != first]
                    second = st.selectbox(f"2nd place", remaining,
                                          key=f"g{letter}_2nd")

                # show top matchup probabilities
                show_matchup_probs(first, second)

                group_selections[letter] = [first, second]

        st.divider()
        st.markdown("### Wild Cards — 8 Best 3rd-Place Teams")
        st.markdown("Pick 8 teams from the 12 possible 3rd-place finishers who also advance to Round of 32.")

        all_thirds = [
            teams[[t for t in teams if t not in group_selections.get(letter, [])][0]
                  if len([t for t in teams if t not in group_selections.get(letter, [])]) > 0
                  else 0]
            if False else
            [t for t in GROUPS[letter] if t not in group_selections.get(letter, [])]
            for letter in GROUPS
        ]
        # flatten and get actual 3rd place candidates
        third_candidates = []
        for letter in GROUPS:
            adv = group_selections.get(letter, [])
            thirds_in_group = [t for t in GROUPS[letter] if t not in adv]
            third_candidates.extend(thirds_in_group)

        wildcards = st.multiselect(
            "Select exactly 8 wild card teams:",
            options=third_candidates,
            max_selections=8,
            key="wildcards"
        )

        if st.button("✅ Confirm Group Stage & Advance to Round of 32",
                     use_container_width=True, key="confirm_groups"):
            if len(wildcards) != 8:
                st.error("Please select exactly 8 wild card teams.")
            else:
                # build r32 teams list: 24 group qualifiers + 8 wildcards
                r32 = []
                for letter in GROUPS:
                    r32.extend(group_selections[letter])
                r32.extend(wildcards)
                st.session_state.group_winners = group_selections
                st.session_state.r32_teams = r32
                st.session_state.stage = "r32"
                st.rerun()

    # STAGE: ROUND OF 32 
    elif st.session_state.stage == "r32":
        st.markdown("### Stage 2 — Round of 32")
        st.markdown(f"32 teams. Pick the winner of each match. Losers are eliminated.")

        teams_32 = st.session_state.r32_teams
        # pair them up sequentially
        matches_r32 = [(teams_32[i], teams_32[i+1]) for i in range(0, 32, 2)]
        winners_r32 = []

        for i, (t1, t2) in enumerate(matches_r32):
            st.markdown(f"**Match {i+1}:** {t1} vs {t2}")
            show_matchup_probs(t1, t2)
            winner = st.radio(f"Winner of Match {i+1}:", [t1, t2],
                              horizontal=True, key=f"r32_m{i}")
            winners_r32.append(winner)
            st.markdown("---")

        if st.button("✅ Confirm Round of 32 & Advance to Round of 16",
                     use_container_width=True, key="confirm_r32"):
            st.session_state.r32_winners = winners_r32
            st.session_state.stage = "r16"
            st.rerun()

    # STAGE: ROUND OF 16 
    elif st.session_state.stage == "r16":
        st.markdown("### Stage 3 — Round of 16")
        st.markdown("16 teams remaining. Pick the winner of each match.")

        teams_16 = st.session_state.r32_winners
        matches_r16 = [(teams_16[i], teams_16[i+1]) for i in range(0, 16, 2)]
        winners_r16 = []

        for i, (t1, t2) in enumerate(matches_r16):
            st.markdown(f"**Match {i+1}:** {t1} vs {t2}")
            show_matchup_probs(t1, t2)
            winner = st.radio(f"Winner of Match {i+1}:", [t1, t2],
                              horizontal=True, key=f"r16_m{i}")
            winners_r16.append(winner)
            st.markdown("---")

        if st.button("✅ Confirm Round of 16 & Advance to Quarter Finals",
                     use_container_width=True, key="confirm_r16"):
            st.session_state.r16_winners = winners_r16
            st.session_state.stage = "qf"
            st.rerun()

    # STAGE: QUARTER FINALS 
    elif st.session_state.stage == "qf":
        st.markdown("### Stage 4 — Quarter Finals")
        st.markdown("8 teams remaining. Pick the winner of each match.")

        teams_qf = st.session_state.r16_winners
        matches_qf = [(teams_qf[i], teams_qf[i+1]) for i in range(0, 8, 2)]
        winners_qf = []

        for i, (t1, t2) in enumerate(matches_qf):
            st.markdown(f"**Quarter Final {i+1}:** {t1} vs {t2}")
            show_matchup_probs(t1, t2)
            winner = st.radio(f"Winner of QF {i+1}:", [t1, t2],
                              horizontal=True, key=f"qf_m{i}")
            winners_qf.append(winner)
            st.markdown("---")

        if st.button("✅ Confirm Quarter Finals & Advance to Semi Finals",
                     use_container_width=True, key="confirm_qf"):
            st.session_state.qf_winners = winners_qf
            st.session_state.stage = "sf"
            st.rerun()

    # STAGE: SEMI FINALS
    elif st.session_state.stage == "sf":
        st.markdown("### Stage 5 — Semi Finals")
        st.markdown("4 teams remaining. The two losers play in the 3rd place match.")

        teams_sf = st.session_state.qf_winners
        sf1_t1, sf1_t2 = teams_sf[0], teams_sf[1]
        sf2_t1, sf2_t2 = teams_sf[2], teams_sf[3]

        st.markdown(f"**Semi Final 1:** {sf1_t1} vs {sf1_t2}")
        show_matchup_probs(sf1_t1, sf1_t2)
        sf1_winner = st.radio("Winner of SF1:", [sf1_t1, sf1_t2],
                              horizontal=True, key="sf1")
        sf1_loser = sf1_t2 if sf1_winner == sf1_t1 else sf1_t1

        st.markdown("---")
        st.markdown(f"**Semi Final 2:** {sf2_t1} vs {sf2_t2}")
        show_matchup_probs(sf2_t1, sf2_t2)
        sf2_winner = st.radio("Winner of SF2:", [sf2_t1, sf2_t2],
                              horizontal=True, key="sf2")
        sf2_loser = sf2_t2 if sf2_winner == sf2_t1 else sf2_t1

        if st.button("✅ Confirm Semi Finals & Go to Final",
                     use_container_width=True, key="confirm_sf"):
            st.session_state.sf_winners  = [sf1_winner, sf2_winner]
            st.session_state.sf_losers   = [sf1_loser, sf2_loser]
            st.session_state.stage = "final"
            st.rerun()

    # STAGE: FINAL
    elif st.session_state.stage == "final":
        sf_winners = st.session_state.sf_winners
        sf_losers  = st.session_state.get("sf_losers", [])

        # 3rd place match
        if len(sf_losers) == 2:
            st.markdown("### 🥉 3rd Place Match")
            st.markdown(f"**{sf_losers[0]} vs {sf_losers[1]}**")
            show_matchup_probs(sf_losers[0], sf_losers[1])
            third = st.radio("3rd Place Winner:",
                             [sf_losers[0], sf_losers[1]],
                             horizontal=True, key="third_place_pick")
            st.markdown("---")

        # final
        st.markdown("### 🏆 The Final")
        st.markdown(f"**{sf_winners[0]} vs {sf_winners[1]}**")
        show_matchup_probs(sf_winners[0], sf_winners[1])
        champion = st.radio("World Cup Champion 🏆:",
                            [sf_winners[0], sf_winners[1]],
                            horizontal=True, key="champion_pick")

        if st.button("🏆 Confirm & See Results", use_container_width=True, key="confirm_final"):
            st.session_state.champion    = champion
            st.session_state.third_place = third if len(sf_losers) == 2 else "N/A"
            st.session_state.stage = "results"
            st.rerun()

    # ── STAGE: RESULTS ────────────────────────────────────────────────────────
    elif st.session_state.stage == "results":
        st.balloons()
        st.markdown("## 🏆 Your FIFA World Cup 2026 Prediction")
        st.divider()

        sf_losers  = st.session_state.get("sf_losers", [])
        sf_winners = st.session_state.sf_winners

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🥇 Champion", st.session_state.champion)
        with col2:
            runner_up = [t for t in sf_winners if t != st.session_state.champion][0]
            st.metric("🥈 Runner-Up", runner_up)
        with col3:
            st.metric("🥉 3rd Place", st.session_state.third_place)
        with col4:
            fourth = [t for t in sf_losers if t != st.session_state.third_place][0] if sf_losers else "N/A"
            st.metric("4th Place", fourth)

        st.divider()
        st.markdown("### Full Bracket Summary")

        # group winners
        st.markdown("**Group Stage Advances:**")
        for letter, winners in st.session_state.group_winners.items():
            st.markdown(f"Group {letter}: {winners[0]} (1st) · {winners[1]} (2nd)")

        st.divider()
        st.markdown("**Knockout Stage Path:**")
        st.markdown(f"Round of 32 → Round of 16 → Quarter Finals → Semi Finals → **{st.session_state.champion} 🏆**")

        st.divider()

        if st.button("🔄 Start Over", use_container_width=True, key="start_over"):
            reset_simulator()
            st.rerun()

    # about section
    with st.expander("🤖 About the Prediction Model"):
        st.markdown("""
        **Model:** Random Forest Classifier — tuned via GridSearchCV, 5-fold cross-validation

        **Training data:** 10,000+ international matches (2010–2024)

        **Features:** Recent win rate, goals per game, goals conceded per game,
        head-to-head win rate, form differential, tournament importance, neutral ground flag

        **Note:** Predictions are based on historical data only.
        They don't account for current injuries, squad selections, or momentum.
        Use them as a guide alongside your own football knowledge.

        **GitHub:** [github.com/Swayam0804](https://github.com/Swayam0804)
        """)

"""
pages/3_Match.py
================

Match page:
- Select a competition, season, stage, and matchweek (single or range).
- Choose a specific match by label.
- Show a concise Home vs Away summary (xG, shots, on target, passes, cards).

Relies on:
- data_utils.load_matches(), data_utils.load_events()
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_utils import load_matches, load_events


st.title("Match")

matches_df = load_matches()
if matches_df.empty:
    st.error("No matches loaded.")
    st.stop()

# 1) Competition selector
competitions = sorted(matches_df["competition"].dropna().unique().tolist()) if "competition" in matches_df.columns else []
if not competitions:
    st.warning("No competitions detected.")
    st.stop()

selected_competition = st.selectbox("Competition:", competitions, key="match_comp_select")

# 2) Season selector (filtered by competition)
if "season" in matches_df.columns:
    seasons = (
        matches_df.loc[matches_df["competition"] == selected_competition, "season"]
        .dropna().unique().tolist()
    )
    seasons = sorted(seasons)
else:
    seasons = []

if not seasons:
    st.warning("No seasons detected for this competition.")
    st.stop()

selected_season = st.selectbox("Season:", seasons, key="match_season_select")

# 3) Stage selector
if "competition_stage" in matches_df.columns:
    stages = (
        matches_df.loc[
            (matches_df["competition"] == selected_competition)
            & (matches_df["season"] == selected_season),
            "competition_stage",
        ].dropna().unique().tolist()
    )
    stages = sorted(stages)
else:
    stages = []

if not stages:
    st.warning("No stages detected for this competition/season.")
    st.stop()

selected_stage = st.selectbox("Competition stage:", stages, key="match_stage_select")

# 4) Matchweek (single or range)
if "match_week" in matches_df.columns:
    mw_series = matches_df.loc[
        (matches_df["competition"] == selected_competition)
        & (matches_df["season"] == selected_season)
        & (matches_df["competition_stage"] == selected_stage),
        "match_week",
    ].dropna()

    # Prefer numeric sorting, fallback otherwise
    try:
        mw_values = pd.to_numeric(mw_series, errors="coerce").dropna().astype(int).unique().tolist()
    except Exception:
        mw_values = mw_series.unique().tolist()

    match_weeks = sorted(mw_values)
else:
    match_weeks = []

if not match_weeks:
    st.warning("No match weeks detected for this selection.")
    st.stop()

selected_match_week = None
start_match_week = None
end_match_week = None

if len(match_weeks) == 1:
    selected_match_week = st.selectbox("Matchweek:", match_weeks, key="match_week_select")
else:
    start_match_week, end_match_week = st.select_slider(
        "Matchweek (range):",
        options=match_weeks,
        value=(match_weeks[0], match_weeks[-1]),
        key="match_week_slider",
    )
    st.caption(f"Selected range: {start_match_week} → {end_match_week}")

# 5) Filter matches by the chosen filters
base_filter = (
    (matches_df["competition"] == selected_competition)
    & (matches_df["season"] == selected_season)
    & (matches_df["competition_stage"] == selected_stage)
)

if selected_match_week is not None:
    filtered_matches = matches_df[base_filter & (matches_df["match_week"] == selected_match_week)]
else:
    filtered_matches = matches_df[
        base_filter
        & (matches_df["match_week"] >= start_match_week)
        & (matches_df["match_week"] <= end_match_week)
    ]

# 6) Build the selection list from label
if "label" in filtered_matches.columns:
    label_options = sorted(filtered_matches["label"].dropna().unique().tolist())
else:
    label_options = []

if not label_options:
    st.warning("No matches found for these filters.")
    st.stop()

selected_label = st.selectbox("Match:", label_options, key="match_label_select")
selected_match = filtered_matches[filtered_matches["label"] == selected_label].iloc[0]

# ---------------------------------------------------------------------------
# Match summary
# ---------------------------------------------------------------------------
match_id = int(selected_match["match_id"])
home_team = selected_match["home_team"]
away_team = selected_match["away_team"]

with st.spinner("Loading events..."):
    events = load_events(match_id)

# Split by team
ev_home = events[events["team"] == home_team]
ev_away = events[events["team"] == away_team]

# Column presence checks
has_type = "type" in events.columns
has_outcome = "shot_outcome" in events.columns
has_xg = "shot_statsbomb_xg" in events.columns
has_card = "card_type" in events.columns
has_pass_type = "pass_type" in events.columns

# Outcomes considered as "saved" (i.e., on-target shots that did not become goals)
SAVED_OUTCOMES = ["Saved", "Saved to Post", "Saved Off Target"]

def kpis_for_team(ev_team: pd.DataFrame, ev_opp: pd.DataFrame) -> dict:
    """
    Compute basic KPIs for a team given its events and the opponent's events.

    Definitions
    ----------
    - shots:       Number of shots by the team.
    - goals:       Number of goals scored by the team.
    - on_target:   Team's shots on target = goals + shots saved by the opponent.
                   (computed from the team's shots only)
    - saved:       Saves made by the team = opponent's shots with a "Saved" outcome.
                   (computed from the opponent's shots)
    - xg:          Sum of StatsBomb xG for the team's shots.
    - passes:      Number of passes by the team.
    - corners:     Number of corners taken by the team.
    - yellow/red:  Count of team's yellow/red cards.

    Parameters
    ----------
    ev_team : pd.DataFrame
        Events for the team of interest (single match).
    ev_opp : pd.DataFrame
        Events for the opponent (same match).

    Returns
    -------
    dict
        {
          "shots": int,
          "goals": int,
          "on_target": int,
          "saved": int,          # saves made by this team (opponent shots saved)
          "xg": float,
          "corners": int,
          "passes": int,
          "yellow": int,
          "red": int
        }

    Notes
    -----
    - We perform defensive column checks so the function won’t crash if a column
      is missing in the Open Data for a specific match/competition.
    - If you display these values, consider labeling "saved" as "Saves"
      in the UI to avoid ambiguity.
    """
    # Defensive column checks (based on StatsBomb Open Data columns)
    has_type       = "type" in ev_team.columns
    has_outcome_t  = "shot_outcome" in ev_team.columns
    has_outcome_o  = "shot_outcome" in ev_opp.columns
    has_xg         = "shot_statsbomb_xg" in ev_team.columns
    has_pass_type  = "pass_type" in ev_team.columns
    has_card       = "card_type" in ev_team.columns

    # --- Offense (team) ---
    shots = ev_team[ev_team["type"] == "Shot"].shape[0] if has_type else 0
    goals = ev_team[ev_team["shot_outcome"] == "Goal"].shape[0] if has_outcome_t else 0
    saved_against_team = (
        ev_team[ev_team["shot_outcome"].isin(SAVED_OUTCOMES)].shape[0] if has_outcome_t else 0
    )
    on_target = goals + saved_against_team  # team's shots on target

    xg = float(ev_team["shot_statsbomb_xg"].sum()) if has_xg else 0.0
    passes = ev_team[ev_team["type"] == "Pass"].shape[0] if has_type else 0
    corners = ev_team[ev_team["pass_type"] == "Corner"].shape[0] if has_pass_type else 0
    yellow = ev_team[ev_team["card_type"] == "Yellow"].shape[0] if has_card else 0
    red    = ev_team[ev_team["card_type"] == "Red"].shape[0] if has_card else 0

    # --- Defense (team) ---
    # Saves made by the team = opponent shots with a saved outcome
    saved = (
        ev_opp[ev_opp["shot_outcome"].isin(SAVED_OUTCOMES)].shape[0] if has_outcome_o else 0
    )

    return dict(
        shots=shots,
        goals=goals,
        on_target=on_target,
        saved=saved,
        xg=xg,
        corners=corners,
        passes=passes,
        yellow=yellow,
        red=red,
    )

home = kpis_for_team(ev_home, ev_away)
away = kpis_for_team(ev_away, ev_home)


# Scoreline row
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([7, 5, 7])

with c1:
    st.markdown(
        f"<div style='text-align:left; font-size:40px; font-weight:bold;'>{home_team}</div>",
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"""
        <div style='display:flex; justify-content:center; align-items:center; font-size:60px; font-weight:bold;'>
            <span>{home['goals']}</span>
            <span style='margin:0 20px;'>-</span>
            <span>{away['goals']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f"<div style='text-align:right; font-size:40px; font-weight:bold;'>{away_team}</div>",
        unsafe_allow_html=True
    )

tab_summary, tab_shots, tab_passes, tab_lineup = st.tabs(["Summary", "Shots", "Passes", "Line-up"])

# Shared CSS for stat rows
st.markdown(
    """
    <style>
    .stat-row{
        --val-font-size: 28px; --val-font-weight: 700; --val-color: #fff;
        --label-font-size: 28px; --label-font-weight: 600; --label-color: #fff;
        --bar-bg: #444; --bar-height: 10px; --bar-radius: 6px;
        --bar-home: #77dd77; --bar-away: #aec6cf;
    }
    .stat-row{margin:8px 0 18px;}
    .stat-row .row-top{display:grid; grid-template-columns:1fr auto 1fr; column-gap:12px; align-items:center;}
    .stat-row .val{font-size:var(--val-font-size); font-weight:var(--val-font-weight); color:var(--val-color);
                   line-height:1; font-variant-numeric:tabular-nums; margin:0;}
    .stat-row .home{text-align:left;} .stat-row .away{text-align:right;}
    .stat-row .label{font-size:var(--label-font-size); font-weight:var(--label-font-weight); color:var(--label-color); white-space:nowrap;}
    .stat-row .row-bars{display:grid; grid-template-columns:1fr 1fr; column-gap:12px; margin-top:6px;}
    .stat-row .bar-left, .stat-row .bar-right{background:var(--bar-bg); height:var(--bar-height); border-radius:var(--bar-radius); overflow:hidden;}
    .stat-row .bar-left{display:flex; justify-content:flex-end;} .stat-row .bar-right{display:flex; justify-content:flex-start;}
    .stat-row .fill-left{background:var(--bar-home); height:var(--bar-height); border-radius:var(--bar-radius);}
    .stat-row .fill-right{background:var(--bar-away); height:var(--bar-height); border-radius:var(--bar-radius);}
    </style>
    """,
    unsafe_allow_html=True
)

def stat_row(home_value, label, away_value, show_bar=True):
    """Render one KPI row with optional left/right bars."""
    def _fmt(v):
        try:
            f = float(v)
            return f"{f:.2f}" if abs(f - int(f)) > 1e-9 else str(int(f))
        except Exception:
            return str(v)

    total = (home_value or 0) + (away_value or 0)
    total = total if total > 0 else 1
    home_pct = (home_value / total) * 100 if total else 0
    away_pct = (away_value / total) * 100 if total else 0

    st.markdown(
        f"""
        <div class="stat-row">
          <div class="row-top">
            <div class="val home">{_fmt(home_value)}</div>
            <div class="label">{label}</div>
            <div class="val away">{_fmt(away_value)}</div>
          </div>
          {f'''
          <div class="row-bars">
            <div class="bar-left"><div class="fill-left" style="width:{home_pct:.6f}%;"></div></div>
            <div class="bar-right"><div class="fill-right" style="width:{away_pct:.6f}%;"></div></div>
          </div>
          ''' if show_bar else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

with tab_summary:
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # KPI rows
    st.markdown("<hr style='border:none; border-top:1px solid #666; margin:14px 0;'>", unsafe_allow_html=True)
    stat_row(home["xg"], "Expected goals", away["xg"])

    st.markdown("<hr style='border:none; border-top:1px solid #666; margin:14px 0;'>", unsafe_allow_html=True)
    stat_row(home["shots"], "Shots", away["shots"])

    st.markdown("<hr style='border:none; border-top:1px solid #666; margin:14px 0;'>", unsafe_allow_html=True)
    stat_row(home["on_target"], "Shots on target", away["on_target"])

    st.markdown("<hr style='border:none; border-top:1px solid #666; margin:14px 0;'>", unsafe_allow_html=True)
    stat_row(home["saved"], "Saved", away["saved"])

    st.markdown("<hr style='border:none; border-top:1px solid #666; margin:14px 0;'>", unsafe_allow_html=True)
    stat_row(home["passes"], "Passes", away["passes"])

    st.markdown("<hr style='border:none; border-top:1px solid #666; margin:14px 0;'>", unsafe_allow_html=True)
    stat_row(home["corners"], "Corners", away["corners"])

    st.markdown("<hr style='border:none; border-top:1px solid #666; margin:14px 0;'>", unsafe_allow_html=True)
    stat_row(home["yellow"], "Yellow cards", away["yellow"])

    st.markdown("<hr style='border:none; border-top:1px solid #666; margin:14px 0;'>", unsafe_allow_html=True)
    stat_row(home["red"], "Red cards", away["red"])

with tab_shots:
    st.info("Add shot maps or shot breakdowns here.")

with tab_passes:
    st.info("Add passing networks or pass maps here.")

with tab_lineup:
    st.info("Add lineups, formations, and substitutions here.")

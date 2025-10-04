"""
pages/2_Season.py
=================

Season page:
- Select a competition & season.
- Display a **ranking table** (for leagues) or a **cup bracket** (for cups).

Relies on:
- data_utils.load_matches() for data.
- viz_utils.plot_ranking_table() and viz_utils.cup_tournament() for visuals.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from data_utils import load_matches
from viz_utils import plot_ranking_table, cup_tournament


st.title("Season")

matches_df = load_matches()
if matches_df.empty:
    st.error("No matches loaded.")
    st.stop()

# 1) Competition selector
competitions = sorted(matches_df["competition"].dropna().unique().tolist()) if "competition" in matches_df.columns else []
if not competitions:
    st.warning("No competitions detected.")
    st.stop()

selected_competition = st.selectbox("Competition:", competitions, key="season_comp_select")

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

selected_season = st.selectbox("Season:", seasons, key="season_season_select")

# Filter dataset for this competition+season
df_season = matches_df[
    (matches_df["competition"] == selected_competition) &
    (matches_df["season"] == selected_season)
]

tab_ranking, tab_stats = st.tabs(["Ranking / Bracket", "Stats"])

with tab_ranking:
    # Determine League vs Cup via metadata added in data_utils
    is_cup = False
    if "competition_type" in df_season.columns:
        val = df_season["competition_type"].iloc[0]
        is_cup = (isinstance(val, str) and val == "Cup")

    if is_cup:
        fig = cup_tournament(df_season)
        st.plotly_chart(fig, use_container_width=True, key="cup_tournament_fig")
    else:
        fig = plot_ranking_table(df_season)
        st.plotly_chart(fig, use_container_width=True, key="league_ranking_fig")

with tab_stats:
    st.info("Add season-level KPIs here (e.g., total goals, xG distribution, top scorers).")

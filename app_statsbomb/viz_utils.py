"""
Visualization helpers for the StatsBomb Streamlit application.

This module defines functions to construct interactive Plotly
figures from StatsBomb event data. Plotly provides built‑in
interactivity such as tooltips, panning and zooming which make
these visualisations ideal for web applications.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots





def plot_ranking_table(data: pd.DataFrame, actor: Optional[str] = None) -> go.Figure:
    """
    Plot a standings-like table built with Plotly scatter traces.

    Parameters
    ----------
    ranking : pd.DataFrame
        Must contain columns: ["rank", "team", "J", "W", "D", "L", "diff_str", "buts", "pts"].
        `rank` is expected to be 1-based (1, 2, 3, ...).
    actor : Optional[str]
        Team name to highlight in the ranking column (optional).

    Returns
    -------
    go.Figure
        The Plotly figure object.
    """

    data = data[[
        "home_team", "away_team", "home_score", "away_score", "match_week", "competition_stage"
    ]]

    # Conditions
    home_win = data["home_score"] > data["away_score"]
    draw = data["home_score"] == data["away_score"]
    away_win = data["home_score"] < data["away_score"]

    # Résultats pour home
    data["home_result"] = np.select(
        [home_win, draw, away_win],
        ["W", "D", "L"]
    )

    # Résultats pour away
    data["away_result"] = np.select(
        [home_win, draw, away_win],
        ["L", "D", "W"]
    )

    # Stats à domicile
    ranking_home = (
        data.groupby('home_team', as_index=False)
        .agg(
            GF=('home_score', 'sum'),
            GA=('away_score', 'sum'),
            W=('home_result', lambda s: (s == 'W').sum()),
            D=('home_result', lambda s: (s == 'D').sum()),
            L=('home_result', lambda s: (s == 'L').sum()),
            J=('home_result', 'count')  # ← ajoute le nombre de matchs
        )
        .rename(columns={'home_team': 'team'})
    )
    ranking_home["info"] = "home"

    # Stats à l'extérieur
    ranking_away = (
        data.groupby('away_team', as_index=False)
        .agg(
            GF=('away_score', 'sum'),
            GA=('home_score', 'sum'),
            W=('away_result', lambda s: (s == 'W').sum()),
            D=('away_result', lambda s: (s == 'D').sum()),
            L=('away_result', lambda s: (s == 'L').sum()),
            J=('home_result', 'count')  # ← ajoute le nombre de matchs
        )
        .rename(columns={'away_team': 'team'})
    )
    ranking_away["info"] = "away"

    # Concat
    ranking = pd.concat([ranking_home, ranking_away], ignore_index=True)

    ranking_resume = (
        ranking.groupby('team', as_index=False)
        .agg(
            GF=('GF', 'sum'),
            GA=('GA', 'sum'),
            W=('W', 'sum'),
            D=('D', 'sum'),
            L=('L', 'sum'),
            J=('J', 'sum'),
        )
    )

    ranking_resume["pts"] = ranking_resume["W"]*3 + ranking_resume["D"]*1

    ranking_resume["diff"] = ranking_resume["GF"] - ranking_resume["GA"]

    ranking_resume["buts"] = ranking_resume["GF"].astype(str) + ":" + ranking_resume["GA"].astype(str)

    ranking_resume = ranking_resume.sort_values(
        by=["pts", "diff", "GF"], 
        ascending=[False, False, False]
    ).reset_index(drop=True)


    ranking_resume = ranking_resume[[
        "team", "W", "D", "L", "buts", "diff", "pts", "J"
    ]]

    ranking_resume["diff_str"] = np.where(
        ranking_resume["diff"] > 0,
        "+" + ranking_resume["diff"].astype(str),  # ajoute le "+" si > 0
        ranking_resume["diff"].astype(str)         # sinon garde la valeur
    )

    ranking_resume.insert(0, "rank", np.arange(1, len(ranking_resume) + 1))




    # --- Validation ----------------------------------------------------------
    required_cols = {"rank", "team", "J", "W", "D", "L", "diff_str", "buts", "pts"}
    missing = required_cols.difference(ranking_resume.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure proper ordering by rank (if not already sorted)
    ranking = ranking_resume.sort_values("rank", kind="stable").reset_index(drop=True)

    # --- Layout constants ----------------------------------------------------
    HEADER_Y = (len(ranking) * 4) + 6
    BASE_Y = (len(ranking) * 4) + 4
    STEP = 4

    # Positions Y (1,2,3,...) -> 42, 40, 38, ...
    y_vals = BASE_Y - ranking["rank"] * STEP

    # --- Helpers -------------------------------------------------------------
    def add_header(fig: go.Figure, x: float, label: str, pos: str = "middle center") -> None:
        fig.add_trace(go.Scatter(
            x=[x], y=[HEADER_Y],
            text=[label],
            mode="text",
            textposition=pos,
            textfont=dict(size=20, color="grey"),
            showlegend=False,
            hoverinfo="skip",
            hovertemplate=None,
        ))

    def add_column(
        fig: go.Figure,
        x: float,
        texts,
        y,
        pos: str = "middle center",
        *,
        mode: str = "text",
        text_size: int = 20,
        text_color: str = "white",
        marker: Optional[dict] = None,
    ) -> None:
        fig.add_trace(go.Scatter(
            x=[x] * len(texts),
            y=y,
            text=texts,
            mode=mode,
            textposition=pos,
            textfont=dict(size=text_size, color=text_color),
            marker=marker or {},
            showlegend=False,
            hoverinfo="skip",
            hovertemplate=None,
        ))

    # --- Figure --------------------------------------------------------------
    fig = go.Figure()

    # Rank column ("#") with optional highlight
    add_header(fig, x=2, label="#")

    # Build marker to highlight `actor` if provided
    marker_colors = ["white"] * len(ranking)
    marker_sizes = [30] * len(ranking)
    if actor is not None:
        # highlight the row where team == actor (case-sensitive by default)
        match_idx = ranking.index[ranking["team"] == actor].tolist()
        for i in match_idx:
            marker_colors[i] = "#FFD54F"  # amber
            marker_sizes[i] = 18

    add_column(
        fig, x=2, texts=ranking["rank"], y=y_vals,
        mode="markers+text",
        text_size=16,
        text_color="black",
        marker=dict(size=marker_sizes, color=marker_colors),
    )

    # Team column
    add_header(fig, x=8, label="Team", pos="middle right")
    add_column(fig, x=8, texts=ranking["team"], y=y_vals, pos="middle right")

    # Stat columns
    stats_columns = [
        (52, "J",     "J"),
        (57, "W",     "W"),
        (62, "D",     "D"),
        (67, "L",     "L"),
        (73, "DIFF",  "diff_str"),
        (81, "Goals", "buts"),
        (88, "PTS",   "pts"),
    ]
    for x, header, col in stats_columns:
        add_header(fig, x=x, label=header)
        add_column(fig, x=x, texts=ranking[col], y=y_vals)

    # Axes & layout
    fig.update_xaxes(range=[0, 90], visible=False)
    fig.update_yaxes(range=[0, HEADER_Y+4], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=(HEADER_Y+4)*1000/90, width=1000,
        margin=dict(l=10, r=10, t=50, b=20),
        showlegend=False,
    )

    return fig


def cup_tournament(cup: pd.DataFrame) -> go.Figure:
    from statsbombpy import sb

    # Phases à élimination directe
    cup = cup[cup["competition_stage"] != "Group Stage"].copy()

    # Libellés (optionnels)
    cup["label"] = (
        cup["home_team"] + " " + cup["home_score"].astype(str)
        + "<br>" + cup["away_team"] + " " + cup["away_score"].astype(str)
    )
    cup["teams_text"]  = cup["home_team"] + "<br>" + cup["away_team"]
    cup["scores_text"] = cup["home_score"].astype(str) + "<br>" + cup["away_score"].astype(str)

    # =========================
    # Pénalties (tirs au but)
    # =========================
    data_cup = []
    for match_id in cup['match_id']:
        df = sb.events(match_id=match_id)
        data_cup.append(df)

    df_cup = pd.concat(data_cup, ignore_index=True)

    penalty_cup = df_cup[
        (df_cup['shot_type'] == 'Penalty') &
        (df_cup['period'] == 5) &
        (df_cup['shot_outcome'] == "Goal")
    ]

    resume_penalty_cup = (
        penalty_cup
        .groupby(['team', 'match_id'])
        .size()
        .reset_index(name='count')
    )

    # Merge pénos
    cup = cup.merge(
        resume_penalty_cup.rename(columns={'team': 'away_team', 'count': 'away_team_penalty'}),
        how='left', on=['match_id', 'away_team']
    )
    cup = cup.merge(
        resume_penalty_cup.rename(columns={'team': 'home_team', 'count': 'home_team_penalty'}),
        how='left', on=['match_id', 'home_team']
    )

    cup['home_team_penalty'] = cup['home_team_penalty'].fillna(0).astype(int)
    cup['away_team_penalty'] = cup['away_team_penalty'].fillna(0).astype(int)

    cup['penalties'] = np.where(
        (cup['home_team_penalty'] > 0) | (cup['away_team_penalty'] > 0), "Yes", "No"
    )

    cup['home_team_result'] = np.where(
        cup['penalties'] == "Yes",
        cup['home_score'].astype(str) + " (" + cup['home_team_penalty'].astype(str) + ")",
        cup['home_score'].astype(str)
    )
    cup['away_team_result'] = np.where(
        cup['penalties'] == "Yes",
        cup['away_score'].astype(str) + " (" + cup['away_team_penalty'].astype(str) + ")",
        cup['away_score'].astype(str)
    )

    # Couleurs vainqueur/perdant
    cup["home_goals"] = cup['home_score'] + cup['home_team_penalty']
    cup["away_goals"] = cup['away_score'] + cup['away_team_penalty']
    cup['home_result'] = np.where(cup['home_goals'] > cup['away_goals'], 'white', 'grey')
    cup['away_result'] = np.where(cup['home_goals'] > cup['away_goals'], 'grey', 'white')

    # =========================
    # Sous-ensembles par tour
    # =========================
    F    = cup[cup["competition_stage"] == "Final"].reset_index(drop=True)
    SF   = cup[cup["competition_stage"] == "Semi-finals"].reset_index(drop=True)
    QF   = cup[cup["competition_stage"] == "Quarter-finals"].reset_index(drop=True)
    R16  = cup[cup["competition_stage"] == "Round of 16"].reset_index(drop=True)

    has_F, has_SF, has_QF, has_R16 = (not F.empty), (not SF.empty), (not QF.empty), (not R16.empty)

    # =========================
    # Helpers robustes
    # =========================
    def safe_match_for_team(df_stage: pd.DataFrame, team_name: str) -> pd.DataFrame:
        """Retourne le match du stage où joue team_name (ou DF vide)."""
        if df_stage.empty or team_name is None or pd.isna(team_name):
            return df_stage.iloc[0:0].copy()
        m = df_stage[(df_stage["home_team"] == team_name) | (df_stage["away_team"] == team_name)]
        return m.head(1).reset_index(drop=True)

    def r16_pair_for_qf(qf_df: pd.DataFrame):
        """Pour un match de quart, retrouve les 2 matchs de 8e (R16) des deux équipes du quart."""
        if qf_df.empty or not has_R16:
            return (R16.iloc[0:0], R16.iloc[0:0])
        t_home = qf_df["home_team"].iloc[0]
        t_away = qf_df["away_team"].iloc[0]
        r1 = safe_match_for_team(R16, t_home)
        r2 = safe_match_for_team(R16, t_away)
        return (r1, r2)

    # =========================
    # Reconstitution de l'arbre (ancrages)
    # =========================
    final = F.head(1).reset_index(drop=True)

    # Demies (ancrées sur la finale si dispo)
    if has_SF and has_F:
        semi_final_1 = safe_match_for_team(SF, final["home_team"].iloc[0])
        semi_final_2 = safe_match_for_team(SF, final["away_team"].iloc[0])
    else:
        semi_final_1 = SF.head(1).reset_index(drop=True) if has_SF else SF.iloc[0:0]
        semi_final_2 = SF.iloc[1:2].reset_index(drop=True) if len(SF) > 1 else SF.iloc[0:0]

    # Quarts (ancrés sur les équipes des demies)
    if has_QF:
        if not semi_final_1.empty:
            semi_final_11 = safe_match_for_team(QF, semi_final_1["home_team"].iloc[0])
            semi_final_12 = safe_match_for_team(QF, semi_final_1["away_team"].iloc[0])
        else:
            semi_final_11 = QF.iloc[0:0]; semi_final_12 = QF.iloc[0:0]
        if not semi_final_2.empty:
            semi_final_21 = safe_match_for_team(QF, semi_final_2["home_team"].iloc[0])
            semi_final_22 = safe_match_for_team(QF, semi_final_2["away_team"].iloc[0])
        else:
            semi_final_21 = QF.iloc[0:0]; semi_final_22 = QF.iloc[0:0]
    else:
        semi_final_11 = QF.iloc[0:0]; semi_final_12 = QF.iloc[0:0]
        semi_final_21 = QF.iloc[0:0]; semi_final_22 = QF.iloc[0:0]

    # 8es (ancrés sur les équipes des quarts)
    semi_final_111, semi_final_112 = r16_pair_for_qf(semi_final_11)
    semi_final_121, semi_final_122 = r16_pair_for_qf(semi_final_12)
    semi_final_211, semi_final_212 = r16_pair_for_qf(semi_final_21)
    semi_final_221, semi_final_222 = r16_pair_for_qf(semi_final_22)

    # =========================
    # Figure & primitives de dessin
    # =========================
    fig = go.Figure()

    def add_connection(x0, y1, y2):
        mid_y = (y1 + y2) / 2
        path = (
            f"M {x0+34} {y1+4.25} "
            f"L {x0+35.5} {y1+4.25} "
            f"L {x0+35.5} {y2+4.25} "
            f"L {x0+34} {y2+4.25} "
            f"M {x0+35.5} {mid_y+4.25} "
            f"L {x0+37} {mid_y+4.25}"
        )
    
        # 1) Halo gris (plus épais)
        fig.add_shape(
            type="path",
            path=path,
            line=dict(color="rgba(150,150,150,0.9)", width=7),
            layer="above",   # ajouté avant -> restera dessous du trait noir
        )
    
        # 2) Trait noir (plus fin) par-dessus
        fig.add_shape(
            type="path",
            path=path,
            line=dict(color="black", width=4),
            layer="above",
        )

    def add_rect(x0, y0, df):
        if df is None or df.empty:
            return
        x1=x0+34
        y1=y0+8.5
        r = min(1.5, (x1 - x0) / 2, (y1 - y0) / 2)
        path = (
            f"M {x0+r},{y0} L {x1-r},{y0} Q {x1},{y0} {x1},{y0+r} "
            f"L {x1},{y1-r} Q {x1},{y1} {x1-r},{y1} L {x0+r},{y1} "
            f"Q {x0},{y1} {x0},{y1-r} L {x0},{y0+r} Q {x0},{y0} {x0+r},{y0} Z"
        )
        fig.add_shape(type="path", path=path, fillcolor="black", line=dict(color="grey", width=1), layer="below")

        fig.add_trace(go.Scatter(
            x=[x0+2], y=[y0+2.25], text=df["home_team"],
            mode="text", textposition="middle right",
            textfont=dict(size=15, color=df["home_result"]),
            showlegend=False, hoverinfo="skip", hovertemplate=None
        ))
        fig.add_trace(go.Scatter(
            x=[x0+2], y=[y0+5.75], text=df["away_team"],
            mode="text", textposition="middle right",
            textfont=dict(size=15, color=df["away_result"]),
            showlegend=False, hoverinfo="skip", hovertemplate=None
        ))
        fig.add_trace(go.Scatter(
            x=[x0+32], y=[y0+2.25], text=df["home_team_result"],
            mode="text", textposition="middle left",
            textfont=dict(size=15, color=df["home_result"]),
            showlegend=False, hoverinfo="skip", hovertemplate=None
        ))
        fig.add_trace(go.Scatter(
            x=[x0+32], y=[y0+5.75], text=df["away_team_result"],
            mode="text", textposition="middle left",
            textfont=dict(size=15, color=df["away_result"]),
            showlegend=False, hoverinfo="skip", hovertemplate=None
        ))

    def try_add_conn(x, y1, y2, cond=True):
        if cond:
            add_connection(x, y1, y2)

    # =========================
    # Layouts EXACTS demandés
    # =========================
    def draw_sf_and_final_only():
        # SF
        add_rect(1, 1,  semi_final_1)
        add_rect(1, 11, semi_final_2)
        try_add_conn(1, 1, 11, cond=(not semi_final_1.empty or not semi_final_2.empty))
        # Final
        add_rect(38, 6, final)

    def draw_qf_sf_final():
        # QF (haut)
        add_rect(1, 1,  semi_final_11)
        add_rect(1, 11, semi_final_12)
        try_add_conn(1, 1, 11, cond=(not semi_final_11.empty or not semi_final_12.empty))
        # QF (bas)
        add_rect(1, 21, semi_final_21)
        add_rect(1, 31, semi_final_22)
        try_add_conn(1, 21, 31, cond=(not semi_final_21.empty or not semi_final_22.empty))
        # SF
        add_rect(38, 6,  semi_final_1)
        add_rect(38, 26, semi_final_2)
        try_add_conn(38, 6, 26, cond=(not semi_final_1.empty or not semi_final_2.empty))
        # Final
        add_rect(75, 16, final)

    def draw_r16_qf_sf_final():
        # R16 (8es)
        add_rect(1,  1, semi_final_111); add_rect(1, 11, semi_final_112); try_add_conn(1, 1, 11, cond=(not semi_final_111.empty or not semi_final_112.empty))
        add_rect(1, 21, semi_final_121); add_rect(1, 31, semi_final_122); try_add_conn(1, 21, 31, cond=(not semi_final_121.empty or not semi_final_122.empty))
        add_rect(1, 41, semi_final_211); add_rect(1, 51, semi_final_212); try_add_conn(1, 41, 51, cond=(not semi_final_211.empty or not semi_final_212.empty))
        add_rect(1, 61, semi_final_221); add_rect(1, 71, semi_final_222); try_add_conn(1, 61, 71, cond=(not semi_final_221.empty or not semi_final_222.empty))
        # QF
        add_rect(38,  6, semi_final_11); add_rect(38, 26, semi_final_12); try_add_conn(38, 6, 26, cond=(not semi_final_11.empty or not semi_final_12.empty))
        add_rect(38, 46, semi_final_21); add_rect(38, 66, semi_final_22); try_add_conn(38, 46, 66, cond=(not semi_final_21.empty or not semi_final_22.empty))
        # SF
        add_rect(75, 16, semi_final_1);  add_rect(75, 56, semi_final_2); try_add_conn(75, 16, 56, cond=(not semi_final_1.empty or not semi_final_2.empty))
        # Final
        add_rect(112, 36, final)

    def draw_final_only():
        add_rect(1, 1, final)

    # =========================
    # Routage selon les tours présents
    # =========================
    if has_R16 and has_QF and has_SF and has_F:
        draw_r16_qf_sf_final()
    elif (not has_R16) and has_QF and has_SF and has_F:
        draw_qf_sf_final()
    elif (not has_R16) and (not has_QF) and has_SF and has_F:
        draw_sf_and_final_only()
    elif has_F and (not has_SF) and (not has_QF) and (not has_R16):
        draw_final_only()
    else:
        # Cas non prévus (ex.: R16 sans QF/SF/F) → on dessine prudemment ce qui existe
        if has_R16:
            draw_r16_qf_sf_final()
        elif has_QF:
            draw_qf_sf_final()
        elif has_SF:
            draw_sf_and_final_only()
        elif has_F:
            draw_final_only()

    # =========================
    # Mise en page Plotly
    # =========================

    if has_R16 and has_QF and has_SF and has_F:
        xrange_0=0
        xrange_1=151
        yrange_0=0
        yrange_1=80
        height_0=600
    elif has_QF and has_SF and has_F:
        xrange_0=0
        xrange_1=114
        yrange_0=0
        yrange_1=40
        height_0=300
    elif has_SF and has_F:
        xrange_0=0
        xrange_1=77
        yrange_0=0
        yrange_1=20
        height_0=200
    elif has_F:
        xrange_0=0
        xrange_1=40
        yrange_0=0
        yrange_1=10
        height_0=150
    else:
        # fallback si autre configuration
        xrange_0=0
        xrange_1=151
        yrange_0=0
        yrange_1=70
        height_0=600



    fig.update_xaxes(range=[xrange_0, xrange_1], visible=False)
    fig.update_yaxes(range=[yrange_0, yrange_1], visible=False)
    fig.update_layout(
        height=height_0, width=(xrange_1-xrange_0)*height_0/(yrange_1-yrange_0),
        margin=dict(l=10, r=10, t=50, b=20),
        showlegend=False,
        #plot_bgcolor="white",
    )

    return fig


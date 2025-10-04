"""
Data loading utilities for the StatsBomb Streamlit application.

These functions wrap common data retrieval tasks using the
`statsbombpy` library and apply Streamlit's caching decorators to
avoid redundant downloads. The aim is to minimise the time spent
fetching data from the StatsBomb API during interactive sessions.

Usage examples
--------------

>>> from statsbomb_app.data_utils import load_matches, load_events
>>> matches = load_matches()
>>> events = load_events(match_id=3795228)

The returned DataFrames include additional columns to make them
more convenient for downstream visualisation routines. In
particular, `load_matches` adds a human‑readable `label` column and
`load_events` expands the nested `location` lists into explicit
`x` and `y` coordinates.
"""

from __future__ import annotations

import re
import unicodedata
import pandas as pd
import streamlit as st
from statsbombpy import sb


# ---------- Table de référence & normalisation ------------------------------

@st.cache_data(persist="disk")
def competition_meta() -> pd.DataFrame:
    """Table de référence competition → country, club, sexe."""
    data = [
        {"competition": "Germany - 1. Bundesliga",                           "country": "Germany",                   "team_type": "club",       "sexe": "M", "competition_type": "League"   },
        {"competition": "Europe - Champions League",                         "country": "Europe",                    "team_type": "club",       "sexe": "M", "competition_type": "Cup"      },
        {"competition": "Spain - Copa del Rey",                              "country": "Spain",                     "team_type": "club",       "sexe": "M", "competition_type": "Cup"      },
        {"competition": "International - FIFA U20 World Cup",                "country": "International",             "team_type": "country",    "sexe": "M", "competition_type": "Cup"      },
        {"competition": "International - FIFA World Cup",                    "country": "International",             "team_type": "country",    "sexe": "M", "competition_type": "Cup"      },
        {"competition": "Spain - La Liga",                                   "country": "Spain",                     "team_type": "club",       "sexe": "M", "competition_type": "League"   },
        {"competition": "France - Ligue 1",                                  "country": "France",                    "team_type": "club",       "sexe": "M", "competition_type": "League"   },
        {"competition": "United States of America - Major League Soccer",    "country": "United States of America",  "team_type": "club",       "sexe": "M", "competition_type": "League"   },
        {"competition": "North and Central America - North American League", "country": "North and Central America", "team_type": "club",       "sexe": "M", "competition_type": "League"   },
        {"competition": "United States of America - NWSL",                   "country": "United States of America",  "team_type": "club",       "sexe": "W", "competition_type": "League"   },
        {"competition": "Italy - Serie A",                                   "country": "Italy",                     "team_type": "club",       "sexe": "M", "competition_type": "Cup"      },
        {"competition": "Europe - UEFA Europa League",                       "country": "Europe",                    "team_type": "club",       "sexe": "M", "competition_type": "Cup"      },
        {"competition": "International - Women's World Cup",                 "country": "International",             "team_type": "country",    "sexe": "W", "competition_type": "Cup"      },
        {"competition": "Africa - African Cup of Nations",                   "country": "Africa",                    "team_type": "country",    "sexe": "M", "competition_type": "Cup"      },
        {"competition": "South America - Copa America",                      "country": "South America",             "team_type": "country",    "sexe": "M", "competition_type": "Cup"      },
        {"competition": "England - FA Women's Super League",                 "country": "England",                   "team_type": "club",       "sexe": "W", "competition_type": "League"   },
        {"competition": "India - Indian Super league",                       "country": "India",                     "team_type": "club",       "sexe": "M", "competition_type": "League"   },
        {"competition": "Argentina - Liga Profesional",                      "country": "Argentina",                 "team_type": "club",       "sexe": "M", "competition_type": "League"   },
        {"competition": "England - Premier League",                          "country": "England",                   "team_type": "club",       "sexe": "M", "competition_type": "League"   },
        {"competition": "Europe - UEFA Euro",                                "country": "Europe",                    "team_type": "country",    "sexe": "M", "competition_type": "Cup"      },
        {"competition": "Europe - UEFA Women's Euro",                        "country": "Europe",                    "team_type": "country",    "sexe": "W", "competition_type": "Cup"      },
    ]
    df = pd.DataFrame(data)
    df["competition"] = df["competition"].astype(str).str.strip()
    return df


def _normalize_competition(s: str) -> str:
    """Normalise la clé de jointure: casse, accents, apostrophes, espaces."""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'")       # apostrophe typographique → simple
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)    # espaces multiples → 1
    return s


# ---------- Chargement & enrichissement -------------------------------------

@st.cache_data(persist="disk")
def load_matches() -> pd.DataFrame:
    """Load all matches available in the StatsBomb open data and enrich with meta info."""
    competitions = sb.competitions()
    matches_list = []

    # Iterate over competitions and seasons to fetch matches per season
    for _, row in competitions.iterrows():
        comp_id = row["competition_id"]
        season_id = row["season_id"]
        try:
            season_matches = sb.matches(competition_id=comp_id, season_id=season_id)
            matches_list.append(season_matches)
        except Exception as exc:
            st.warning(
                f"Impossible de charger les matches pour competition_id={comp_id}, "
                f"season_id={season_id}: {exc}"
            )
            continue

    # Concatenate all match DataFrames
    matches = pd.concat(matches_list, ignore_index=True)

    # Build a descriptive label for each match for user selection
    matches["label"] = (
        " (" + matches["competition"] + " " + matches["season"] + " J-"
        + matches["match_week"].astype(str).str.zfill(2) + ") "
        + matches["home_team"] + " vs " + matches["away_team"]
    )

    # ---- Enrichissement direct avant le return -----------------------------

    # Prépare la table de référence et les clés normalisées
    meta = competition_meta().copy()
    meta["_comp_key"] = meta["competition"].map(_normalize_competition)

    matches["_comp_key"] = matches["competition"].map(_normalize_competition)

    # Jointure m:1 sur la clé normalisée (évite la duplication de `competition`)
    matches = matches.merge(
        meta.drop(columns=["competition"]),
        on="_comp_key",
        how="left",
        validate="m:1",
        suffixes=("", "_meta"),
    ).drop(columns="_comp_key")

    # Feedback Streamlit si des compétitions n'ont pas matché
    missing = matches["country"].isna().sum()
    if missing:
        st.info(f"{missing} lignes sans correspondance de compétition dans la table de référence.")

    return matches


@st.cache_data(persist="disk")
def load_events(match_id_2: int) -> pd.DataFrame:
    """Load event data for a specific match.

    Parameters
    ----------
    match_id : int
        Unique identifier of the match to retrieve events for.

    Returns
    -------
    pd.DataFrame
        DataFrame of events with extracted `x` and `y` coordinates
        where available.
    """

    import numpy as np

    # Fetch the events for the given match ID
    events = sb.events(match_id=match_id_2)
    # Extract coordinates from nested location lists, if present
    if "location" in events.columns:

        events["x"] = events["location"].apply(lambda v: v[0] if isinstance(v, list) else np.nan)
        events["y"] = events["location"].apply(lambda v: v[1] if isinstance(v, list) else np.nan)

        events["end_x"] = events["shot_end_location"].apply(
            lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan
        )
        events["end_y"] = events["shot_end_location"].apply(
            lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan
        )
        events["end_z"] = events["shot_end_location"].apply(
            lambda v: v[2] if isinstance(v, list) and len(v) > 2 else np.nan
        )

    if "bad_behaviour_card" in events.columns:
        conditions = [
            events["bad_behaviour_card"].str.contains("Yellow", na=False),
            events["bad_behaviour_card"].str.contains("Red", na=False)
        ]
        choices = ["Yellow", "Red"]
    
        events["card_type"] = np.select(conditions, choices, default=np.nan)

    return events
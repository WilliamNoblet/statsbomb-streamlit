"""
Streamlit entry point for the StatsBomb interactive dashboard.

This application allows users to browse matches from the open
StatsBomb dataset, select a match of interest, and view an
interactive shot map. The project is modular, with data
handling and visualisation separated into helper modules for
clarity and maintainability.

Run this app locally with:

    streamlit run OneDrive/Desktop/statsbomb-streamlit/app_statsbomb/app.py

Ensure that the dependencies listed in requirements.txt are
installed in your environment. When deployed on Streamlit Cloud
or another hosting platform, the cache settings defined in
data_utils.py will persist the downloaded data across sessions.
"""

import streamlit as st
from data_utils import load_matches


st.set_page_config(page_title="StatsBomb Interactive Analysis", layout="wide")
st.title("StatsBomb Interactive Match Analysis")

matches_df = load_matches()
if matches_df.empty:
    st.error("No matches loaded.")
    st.stop()

st.write("Use the sidebar to open **Season** or **Match** (auto-discovered from `pages/`).")










# streamlit_app.py
import streamlit as st
import pandas as pd
from onir import find_undervalued_stocks

st.set_page_config(page_title="Value Investing Dashboard", layout="wide")

st.title("Value Investing Dashboard")

# Sidebar controls
exchange = st.sidebar.selectbox("Exchange", ["nyse", "allNasdaq"])
div_only = st.sidebar.checkbox("Dividend stocks only")
min_fcf = st.sidebar.number_input("Min FCF Ratio (%)", 0.0, 100.0, 10.0, 1.0)
min_gap = st.sidebar.slider("Min Undervaluation (%)", 0, 90, 30, 5)

# Cache data for 1 hour to avoid hitting API too much
@st.cache_data(ttl=3600)
def get_data(ex: str) -> pd.DataFrame:
    return find_undervalued_stocks(ex)

colA, colB = st.columns([1,1])
with colA:
    if st.button("Refresh now"):
        st.cache_data.clear()

df = get_data(exchange).copy()

# Filters
if not df.empty:
    if div_only:
        df = df[df["Dividend Stock"] == True]
    df = df[df["FCF Ratio %"] >= float(min_fcf)]
    df = df[df["Undervaluation %"] >= float(min_gap)]

    # Sort by undervaluation desc
    df = df.sort_values("Undervaluation %", ascending=False)

st.metric("Candidates", len(df))
st.dataframe(df, use_container_width=True)

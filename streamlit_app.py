# streamlit_app.py
import streamlit as st
import pandas as pd
from streamlit.components.v1 import html as render_html
from onir import find_undervalued_stocks
import plotly.graph_objects as go

st.set_page_config(page_title="Value Investing Dashboard", layout="wide")

# ------------------------------
# CACHING: compute only on demand
# ------------------------------
@st.cache_data(ttl=None, show_spinner=False)
def get_data(ex: str) -> pd.DataFrame:
    return find_undervalued_stocks(ex)

# ------------------------------
# Sidebar controls (list view)
# ------------------------------
exchange = st.sidebar.selectbox("Exchange", ["nyse", "allNasdaq"])
div_only = st.sidebar.checkbox("Dividend stocks only")
# min_fcf = st.sidebar.number_input("Min FCF Ratio (%)", 0.0, 100.0, 10.0, 1.0)
# min_gap = st.sidebar.slider("Min Undervaluation (%)", 0, 90, 30, 5)

colA, colB = st.columns([1, 1])
with colA:
    if st.button("Refresh now", help="Recompute from APIs and update the cache"):
        st.cache_data.clear()
        st.rerun()

# ------------------------------------
# Router: list vs details using params
# ------------------------------------
qp = st.query_params  # dict-like
view = qp.get("view", "list")
symbol_param = qp.get("symbol")

# ------------------------------
# DETAILS VIEW (embedded HTML/JS)
# ------------------------------
# DETAIL_HTML = r"""
# ... (removed giant HTML/JS blob)
# """

def render_details(symbol: str):
    st.title(f"{symbol} • Intrinsic Value")
    if st.button("← Back to list"):
        st.query_params["view"] = "list"
        st.query_params.pop("symbol", None)
        st.rerun()

    from onir import (
        fetch_eps_data, fetch_price_data, fetch_current_price, fetch_ratios,
        fetch_dividend_data, calculate_annualized_return, calculate_intrinsic_value
    )
    import numpy as np
    import pandas as pd

    eps_data = fetch_eps_data(symbol)
    price_data = fetch_price_data(symbol)
    current_price = fetch_current_price(symbol)
    ratios = fetch_ratios(symbol)
    dividend_data = fetch_dividend_data(symbol)
    annualized_return = calculate_annualized_return(symbol, current_price)
    info = calculate_intrinsic_value(symbol)

    # --- EPS and P/E history for tables and chart ---
    eps_years, eps_vals, pe_vals = [], [], []
    for item in eps_data:
        try:
            y = int(item["date"][:4])
            if 2017 <= y <= 2024 and "eps" in item:
                eps_years.append(y)
                eps = float(item["eps"])
                eps_vals.append(eps)
                y_str = str(y)
                if price_data and y_str in price_data and eps != 0:
                    pe_vals.append(price_data[y_str] / eps)
                else:
                    pe_vals.append(None)
        except Exception:
            continue
    # Only keep last 7 years, most recent last
    if len(eps_years) > 7:
        eps_years = eps_years[-7:]
        eps_vals = eps_vals[-7:]
        pe_vals = pe_vals[-7:]
    eps_df = pd.DataFrame({"Year": eps_years, "EPS": eps_vals})
    pe_df = pd.DataFrame({"Year": eps_years, "P/E": pe_vals})

    # --- Headline Metrics ---
    st.markdown("""
        <div style='display:flex;gap:2.5em;margin-bottom:0.5em;'>
            <div style='flex:1;text-align:center;'>
                <div style='font-size:1.1em;color:#888;'>Current Price</div>
                <div style='font-size:2.2em;font-weight:700;'>${:.2f}</div>
            </div>
            <div style='flex:1;text-align:center;'>
                <div style='font-size:1.1em;color:#888;'>Intrinsic Value</div>
                <div style='font-size:2.2em;font-weight:700;'>${:.2f}</div>
            </div>
            <div style='flex:1;text-align:center;'>
                <div style='font-size:1.1em;color:#888;'>Undervaluation</div>
                <div style='font-size:2.2em;font-weight:700;'>{:.1f}%</div>
            </div>
        </div>
    """.format(info['Current Price'], info['Intrinsic Value'], info['Undervaluation %']), unsafe_allow_html=True)

    # --- Projection Metric ---
    st.markdown("""
        <div style='text-align:center;margin-bottom:1.2em;'>
            <span style='font-size:1.1em;color:#888;'>Annualized Return (3y proj.)</span><br>
            <span style='font-size:1.7em;font-weight:600;'>{:.2f}%</span>
        </div>
    """.format(info['Annualized Return %']), unsafe_allow_html=True)

    # --- All Ratios Table ---
    ratios_table = pd.DataFrame({
        "Metric": [
            "EPS Growth Rate", "PEG Ratio", "P/E (avg)", "P/B Ratio",
            "ROE", "FCF Ratio", "Dividend Yield", "Debt to Equity", "P/S Ratio"
        ],
        "Value": [
            f"{info['EPS Growth Rate']:.2f}%",
            f"{info['PEG Ratio']:.2f}",
            f"{np.nanmean([v for v in pe_vals if v is not None]):.2f}" if pe_vals else "-",
            f"{info['P/B Ratio']:.2f}",
            f"{info['ROE %']:.2f}%",
            f"{info['FCF Ratio %']:.2f}%",
            f"{info['Dividend Yield']:.2f}%",
            f"{info['Debt to Equity']:.2f}%",
            f"{info['P/S Ratio']:.2f}"
        ]
    })
    st.markdown("<div style='margin-bottom:0.5em;'></div>", unsafe_allow_html=True)
    st.dataframe(ratios_table.set_index("Metric"), use_container_width=True, height=320)

    # --- EPS Growth Chart and Tables ---
    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eps_years, y=eps_vals, mode="lines+markers", name="EPS"))
        fig.update_layout(
            title="EPS Growth (2017–2024)",
            xaxis_title="Year", yaxis_title="EPS",
            margin=dict(l=20, r=20, t=30, b=20),
            height=220,
            font=dict(size=13)
        )
        st.plotly_chart(fig, use_container_width=True)
    with table_col:
        st.markdown("<b>EPS (last 7y)</b>", unsafe_allow_html=True)
        st.table(eps_df.set_index("Year"))
        st.markdown("<b>P/E (last 7y)</b>", unsafe_allow_html=True)
        st.table(pe_df.set_index("Year"))

# ------------------------------
# LIST VIEW
# ------------------------------
def render_list():
    st.title("Value Investing Dashboard")

    df = get_data(exchange).copy()
    if df.empty:
        st.info("No data yet. Click **Refresh now** to fetch.")
        return

    # Filters
    if div_only:
        df = df[df["Dividend Stock"] == True]
    # df = df[df["FCF Ratio %"] >= float(min_fcf)]
    # df = df[df["Undervaluation %"] >= float(min_gap)]
    df = df.sort_values("Undervaluation %", ascending=False)

    st.metric("Candidates", int(len(df)))
    st.dataframe(df, use_container_width=True)

    # Picker + open details
    if "Symbol" in df.columns and not df.empty:
        sel = st.selectbox("View details for ticker:", options=sorted(df["Symbol"].unique()))
        open_col, _ = st.columns([1, 5])
        with open_col:
            if st.button("Open details"):
                st.query_params["view"] = "detail"
                st.query_params["symbol"] = sel
                st.rerun()

# ------------------------------
# ROUTE
# ------------------------------
if view == "detail" and symbol_param:
    render_details(symbol_param)
else:
    render_list()

# onir.py
import os
import time
import csv
import warnings
from datetime import date

import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

API_KEY = "y0axaPDDB3bSmyfPBpLg45tj4ZJdMjgW"
if not API_KEY:
    # You can also set a default for local testing, but don't commit real keys.
    # API_KEY = "YOUR_DEV_KEY"
    pass

# -----------------------------
# Ticker fetchers
# -----------------------------
def fetch_NYSE_tickers():
    url = f"https://financialmodelingprep.com/api/v3/symbol/NYSE?apikey={API_KEY}"
    try:
        data = requests.get(url, timeout=30).json()
        return [item.get("symbol") for item in data if "symbol" in item]
    except Exception:
        return []

def fetch_all_nasdaq_tickers():
    url = f"https://financialmodelingprep.com/api/v3/symbol/nasdaq?apikey={API_KEY}"
    try:
        data = requests.get(url, timeout=30).json()
        return [item.get("symbol") for item in data if "symbol" in item]
    except Exception:
        return []

# -----------------------------
# Data fetch helpers
# -----------------------------
def fetch_market_cap(symbol):
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}"
    try:
        data = requests.get(url, timeout=30).json()
        return data[0].get("mktCap")
    except Exception:
        return None

def fetch_eps_data(symbol):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={API_KEY}"
    try:
        return requests.get(url, timeout=30).json()
    except Exception:
        return []

def fetch_price_data(symbol):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&from=2017-01-01&to=2024-12-31&apikey={API_KEY}"
    try:
        data = requests.get(url, timeout=45).json()
        if "historical" not in data:
            return None
        yearly_prices = {}
        for item in data["historical"]:
            year = item["date"][:4]
            price = item["close"]
            yearly_prices.setdefault(year, []).append(price)
        # Restrict to 2017–2024 means per your logic
        return {year: float(np.mean(prices))
                for year, prices in yearly_prices.items()
                if "2017" <= year <= "2024"}
    except Exception:
        return None

def fetch_current_price(symbol):
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}"
    try:
        data = requests.get(url, timeout=20).json()
        return float(data[0]["price"])
    except Exception:
        return None

def fetch_dividend_data(symbol):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?apikey={API_KEY}"
    try:
        data = requests.get(url, timeout=30).json()
        # Normalize to a list of entries
        return data.get("historical", []) if isinstance(data, dict) else []
    except Exception:
        return []

def fetch_dividend_yield(symbol, price):
    divs = fetch_dividend_data(symbol)
    if not divs or not price:
        return 0.0
    # Sum 2024 dividends
    dividends_by_year = {}
    for entry in divs:
        try:
            year = int(entry["date"][:4])
            if 2020 <= year <= 2024:
                dividends_by_year[year] = dividends_by_year.get(year, 0.0) + float(entry["dividend"])
        except Exception:
            continue
    annual_dividend_2024 = dividends_by_year.get(2024, 0.0)
    return (annual_dividend_2024 / price) * 100 if price else 0.0

def fetch_ratios(symbol):
    try:
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}"
        income_url  = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=1&apikey={API_KEY}"
        balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=1&apikey={API_KEY}"
        cashflow_url= f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?limit=1&apikey={API_KEY}"

        profile = requests.get(profile_url, timeout=20).json()[0]
        income  = requests.get(income_url,  timeout=20).json()[0]
        balance = requests.get(balance_url, timeout=20).json()[0]
        cash    = requests.get(cashflow_url, timeout=20).json()[0]

        price = float(profile["price"])
        revenue = float(income["revenue"])
        net_income = float(income["netIncome"])
        shares = float(income["weightedAverageShsOut"])
        equity = float(balance["totalStockholdersEquity"])
        fcf = float(cash["freeCashFlow"])
        liabilities = float(balance["totalLiabilities"])
        if equity == 0 or revenue == 0 or shares == 0:
            return None

        d_to_e = liabilities / equity
        book_value_per_share = equity / shares
        pbr = price / book_value_per_share
        roe = (net_income / equity) * 100.0
        market_cap = price * shares
        psr = market_cap / revenue
        fcf_ratio = (fcf / revenue) * 100.0

        return {"psr": psr, "pbr": pbr, "roe": roe, "fcf_ratio": fcf_ratio, "d/e": d_to_e}
    except Exception:
        return None

# -----------------------------
# Analytics
# -----------------------------
def calculate_annualized_return(symbol, current_price):
    """Project 3y ahead using EPS CAGR, same PE, + dividends via avg payout ratio of overlapping years."""
    try:
        eps_data = fetch_eps_data(symbol)
        div_list = fetch_dividend_data(symbol)
        current_price = fetch_current_price(symbol)

        eps_years = {
            int(item["date"][:4]): float(item["eps"])
            for item in eps_data
            if isinstance(item, dict) and "date" in item and "eps" in item and 2017 <= int(item["date"][:4]) <= 2024
        }
        if len(eps_years) < 2:
            return 0.0

        years_sorted = sorted(eps_years)
        eps_values   = [eps_years[y] for y in years_sorted]
        oldest_eps, latest_eps = eps_values[0], eps_values[-1]
        if oldest_eps <= 0 or not latest_eps or not current_price:
            return 0.0

        years_apart = len(eps_values) - 1
        eps_cagr = (latest_eps / oldest_eps) ** (1 / years_apart) - 1

        # Project EPS next 3Y
        eps_y1 = latest_eps * (1 + eps_cagr)
        eps_y2 = eps_y1   * (1 + eps_cagr)
        eps_y3 = eps_y2   * (1 + eps_cagr)

        pe_now = current_price / latest_eps if latest_eps else 0.0
        projected_price = eps_y3 * pe_now if pe_now > 0 else current_price

        # Build DPS map
        dps_by_year = {}
        for entry in div_list:
            try:
                y = int(entry["date"][:4])
                dps_by_year[y] = dps_by_year.get(y, 0.0) + float(entry["dividend"])
            except Exception:
                continue

        # Overlap years for payout ratio
        overlap = sorted(set(years_sorted) & set(dps_by_year))
        overlap = [y for y in overlap if eps_years.get(y, 0) > 0]
        if overlap:
            payout_ratios = [dps_by_year[y] / eps_years[y] for y in overlap[-3:]]  # up to last 3
            avg_payout = sum(payout_ratios) / len(payout_ratios) if payout_ratios else 0.0
            div1 = eps_y1 * avg_payout
            div2 = eps_y2 * avg_payout
            div3 = eps_y3 * avg_payout
        else:
            div1 = div2 = div3 = 0.0

        total_gain = (projected_price - current_price) + div1 + div2 + div3
        if current_price <= 0:
            return 0.0
        ann = (pow(1 + total_gain / current_price, 1 / 3) - 1) * 100.0
        return float(ann)
    except Exception:
        return 0.0

def calculate_intrinsic_value(symbol):
    eps_data   = fetch_eps_data(symbol)
    price_data = fetch_price_data(symbol)
    current    = fetch_current_price(symbol)
    ratios     = fetch_ratios(symbol)
    if not eps_data or not price_data or not current or not ratios:
        return None

    # EPS year->value (2017–2024), sorted ASC by year
    eps_pairs = []
    for item in eps_data:
        try:
            y = int(item["date"][:4])
            if 2017 <= y <= 2024 and "eps" in item:
                eps_pairs.append((y, float(item["eps"])))
        except Exception:
            continue

    if len(eps_pairs) < 7:
        return None

    # Disallow >1 negative EPS year
    if sum(1 for _, eps in eps_pairs if eps < 0) > 1:
        return None

    # IQR filter + eps >= 0.5
    years, eps_vals = zip(*sorted(eps_pairs))  # ascending by year
    eps_vals = list(eps_vals)
    Q1, Q3 = np.percentile(eps_vals, [25, 75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    cleaned = [(y, e) for y, e in zip(years, eps_vals) if (e >= lower and e >= 0.5)]

    if len(cleaned) < 2:
        return None

    # Growth rate
    cleaned.sort(key=lambda x: x[0])  # by year
    oldest_eps = cleaned[0][1]
    latest_eps = cleaned[-1][1]
    if oldest_eps <= 0:
        return None
    yrs_apart = len(cleaned) - 1
    eps_growth_rate = (latest_eps / oldest_eps) ** (1 / yrs_apart) - 1
    if eps_growth_rate <= 0.10:  # require >10% EPS CAGR
        return None

    # P/E per available year (using mean price)
    pe_values = []
    for y, e in cleaned:
        y_str = str(y)
        if y_str in price_data and e != 0:
            pe_values.append(price_data[y_str] / e)
    if not pe_values:
        return None

    # Remove only abnormally high P/Es
    q1p, q3p = np.percentile(pe_values, [25, 75])
    iqrp = q3p - q1p
    upper = q3p + 1.5 * iqrp
    final_pe = [p for p in pe_values if p <= upper]
    if not final_pe:
        return None

    avg_pe = float(np.mean(final_pe))
    peg_ratio = avg_pe / (eps_growth_rate * 100.0)

    # Fundamental filters
    if ratios["psr"] > 5.0 or ratios["pbr"] >= 10.0 or (ratios["pbr"] >= 5 and ratios["roe"] <= 15.0):
        return None
    if ratios["roe"] <= 5.0 or peg_ratio >= 5.0 or ratios["fcf_ratio"] < 5.0:
        return None

    intrinsic_value = avg_pe * latest_eps
    undervaluation = ((intrinsic_value - current) / current) * 100.0
    annualized_return = calculate_annualized_return(symbol, current)
    dividend_yield = fetch_dividend_yield(symbol, current)

    return {
        "Symbol": symbol,
        "Intrinsic Value": intrinsic_value,
        "Current Price": current,
        "Undervaluation %": undervaluation,
        "EPS Growth Rate": eps_growth_rate * 100.0,
        "PEG Ratio": peg_ratio,
        "P/S Ratio": ratios["psr"],
        "P/B Ratio": ratios["pbr"],
        "ROE %": ratios["roe"],
        "FCF Ratio %": ratios["fcf_ratio"],
        "Dividend Yield": dividend_yield,
        "Debt to Equity": ratios["d/e"] * 100.0,  # you were writing d/e with %, keeping that
        "Dividend Stock": (dividend_yield > 0),
        "Annualized Return %": annualized_return,
    }

# -----------------------------
# Batch helpers & main function
# -----------------------------
def batch_fetch_profiles(ticker_list):
    chunks = [ticker_list[i:i + 100] for i in range(0, len(ticker_list), 100)]
    results = {}
    for chunk in chunks:
        symbols = ",".join(chunk)
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbols}?apikey={API_KEY}"
        try:
            data = requests.get(url, timeout=30).json()
            for item in data:
                results[item["symbol"]] = item
        except Exception as e:
            # print(f"Error fetching profile batch {chunk}: {e}")
            pass
    return results

def find_undervalued_stocks(index: str = "nyse") -> pd.DataFrame:
    """Return a DataFrame of undervalued stocks (NO file writes)."""
    if index == "nyse":
        tickers = fetch_NYSE_tickers()
    elif index == "allNasdaq":
        tickers = fetch_all_nasdaq_tickers()
    else:
        tickers = []

    # Fix BRK.B style tickers
    tickers = [t.replace(".", "-") for t in tickers if t]

    # Pre-fetch profiles for market cap and price
    profiles = batch_fetch_profiles(tickers)

    rows = []
    for symbol in tickers:
        try:
            profile = profiles.get(symbol)
            if not profile:
                continue
            mkt = profile.get("mktCap")
            if not mkt or mkt < 10_000_000_000:  # >= $10B only
                continue

            info = calculate_intrinsic_value(symbol)
            if not info or info["Undervaluation %"] < 30.0:
                continue
            rows.append(info)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "Symbol","Intrinsic Value","Current Price","Undervaluation %",
            "EPS Growth Rate","PEG Ratio","P/S Ratio","P/B Ratio",
            "ROE %","FCF Ratio %","Dividend Yield","Debt to Equity",
            "Dividend Stock","Annualized Return %"
        ])

    df = pd.DataFrame(rows)
    # Pretty formatting is left to Streamlit; keep numeric types here.
    return df

if __name__ == "__main__":
    # Local test run (prints shape)
    df_test = find_undervalued_stocks("nasdaq")
    print(df_test.shape)

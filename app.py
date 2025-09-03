# app.py — Crypto correlations using CoinGecko (with range filters + token focus)
# Requirements this file meets:
# 1) Price AND Volatility correlation heatmaps + tables
# 2) Show only in-range correlations (min/max), default min=0.90, max=0.99; color = Blues
# 3) Token Focus: pick a token (e.g., ETH) to see who’s most correlated to it
# 4) Avoid 1.0 clones via max<1.0; exclude self in token focus

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------- Page ----------------
st.set_page_config(page_title="Crypto Correlations — CoinGecko", layout="wide")
st.title("🔗 Crypto Price & Volatility Correlations")
st.caption("Source: CoinGecko market data. We fetch top coins by mkt cap, daily USD prices, compute log-return and rolling-vol correlations.")

# ---------------- Auth / hosts ----------------
API_KEY = st.secrets.get("COINGECKO_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing COINGECKO_API_KEY in `.streamlit/secrets.toml`.")
    st.stop()

PUBLIC_BASE = "https://api.coingecko.com/api/v3"
PRO_BASE    = "https://pro-api.coingecko.com/api/v3"

def _headers_for(base_url: str):
    # Demo/Public uses x-cg-demo-api-key on PUBLIC_BASE; Pro uses x-cg-pro-api-key
    if base_url == PRO_BASE:
        return {"x-cg-pro-api-key": API_KEY, "Accept": "application/json", "Accept-Encoding": "gzip"}
    else:
        return {"x-cg-demo-api-key": API_KEY, "Accept": "application/json", "Accept-Encoding": "gzip"}

@st.cache_resource(show_spinner=False)
def select_working_cg_base() -> str:
    for base in (PUBLIC_BASE, PRO_BASE):
        try:
            r = requests.get(f"{base}/ping", headers=_headers_for(base), timeout=15)
            if r.status_code == 200:
                return base
            r2 = requests.get(
                f"{base}/coins/markets",
                params={"vs_currency": "usd", "per_page": 1, "page": 1, "order": "market_cap_desc"},
                headers=_headers_for(base), timeout=20
            )
            if r2.status_code == 200:
                return base
        except requests.RequestException:
            pass
    return PUBLIC_BASE

CG_BASE = select_working_cg_base()
st.caption(f"CoinGecko host: `{CG_BASE}`  •  key: …{API_KEY[-4:]}")

# ---------------- HTTP helper ----------------
def http_get(path: str, params=None, timeout=30, retries=3, backoff=2.0):
    url = f"{CG_BASE}{path}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=_headers_for(CG_BASE), timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 401:
                st.error("CoinGecko auth failed (401). Make sure the key matches the host (Demo vs Pro).")
                st.stop()
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            last_err = f"{r.status_code}: {r.text[:220]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(0.25)
    st.error(f"GET {path} failed after retries. Last error: {last_err}")
    return None

# ---------------- Universe ----------------
STABLE_SYMS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDE","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX","BUSD","EURT","EURS"
}

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_universe_top_n(top_n=100):
    r = http_get("/coins/markets", params={
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": ""
    }, timeout=40, retries=3)
    if not r:
        return [], [], pd.DataFrame()
    rows = r.json()
    if not isinstance(rows, list) or not rows:
        return [], [], pd.DataFrame()

    ids, syms, seen = [], [], set()
    for row in rows:
        sym = str(row.get("symbol", "")).upper()
        if not sym or sym in STABLE_SYMS:
            continue
        if sym in seen:
            continue
        cid = row.get("id")
        if not cid:
            continue
        seen.add(sym)
        ids.append(cid); syms.append(sym)
        if len(ids) >= top_n:
            break
    return ids, syms, pd.DataFrame(rows)

# ---------------- History ----------------
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily_series(coin_id: str, days: int = 365) -> pd.Series | None:
    days = max(30, min(365, int(days)))
    r = http_get(f"/coins/{coin_id}/market_chart", params={
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }, timeout=40, retries=3)
    if not r:
        return None
    data = r.json()
    prices = data.get("prices") or []
    if not prices:
        return None
    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
    s = pd.Series(pd.to_numeric(df["price"], errors="coerce").astype(float).values, index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    daily = s.resample("D").last().ffill(limit=3)
    daily = daily[daily > 0]
    if len(daily) < 30:
        return None
    daily.name = coin_id
    return daily

# ---------------- Math helpers ----------------
def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(price_df).diff()
    # clip tiny tails to de-spike without flattening structure
    for col in rets.columns:
        x = rets[col]
        if x.notna().sum() > 10:
            rets[col] = x.clip(x.quantile(0.001), x.quantile(0.999))
    return rets

def realized_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    minp = max(2, window // 2)
    return returns.rolling(window, min_periods=minp).std() * np.sqrt(365)

def pairwise_corr(df: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    Z = (df - df.mean()) / df.std(ddof=0)
    Z = Z.replace([np.inf, -np.inf], np.nan)
    cols = list(Z.columns)
    n = len(cols)
    M = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        M[i, i] = 1.0
        xi = Z.iloc[:, i].to_numpy()
        for j in range(i + 1, n):
            xj = Z.iloc[:, j].to_numpy()
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() >= min_overlap:
                c = np.corrcoef(xi[mask], xj[mask])[0, 1]
                if np.isfinite(c):
                    M[i, j] = c
                    M[j, i] = c
    return pd.DataFrame(M, index=cols, columns=cols)

def mask_corr_range(C: pd.DataFrame, min_corr: float, max_corr: float) -> pd.DataFrame:
    """Keep only values within [min_corr, max_corr]; set others (and diagonal) to NaN."""
    if C is None or C.empty:
        return C
    M = C.copy()
    # kill diagonal always
    for i in range(len(M)):
        M.iat[i, i] = np.nan
    good = (M >= min_corr) & (M <= max_corr)
    M = M.where(good, np.nan)
    # drop all-NaN rows/cols to keep plot readable
    M = M.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return M

def render_heatmap(C: pd.DataFrame, title: str, min_corr: float, max_corr: float, key: str):
    if C is None or C.empty:
        st.warning(f"No values in range [{min_corr:.2f}, {max_corr:.2f}] for {title}.")
        return
    # limit to most connected names if huge
    if C.shape[0] > 30:
        avg_abs = np.nanmean(np.where(np.isfinite(C.values), np.abs(C.values), np.nan), axis=1)
        order = np.argsort(avg_abs)[::-1][:30]
        keep = [C.index[i] for i in order]
        C = C.loc[keep, keep]

    fig = go.Figure(data=go.Heatmap(
        z=C.values,
        x=C.columns.tolist(),
        y=C.index.tolist(),
        zmin=min_corr,
        zmax=max_corr,
        colorscale="Blues",           # white→blue as correlation increases
        colorbar=dict(title="ρ"),
        hoverongaps=False,
        text=np.round(C.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=700,
        margin=dict(l=100, r=20, t=60, b=100),
    )
    st.plotly_chart(fig, key=key, width="stretch")

def top_pairs_from_matrix(C: pd.DataFrame, k=30) -> pd.DataFrame:
    if C is None or C.empty:
        return pd.DataFrame(columns=["A","B","ρ"])
    pairs = []
    cols = list(C.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = C.iat[i, j]
            if np.isfinite(v):
                pairs.append((cols[i], cols[j], float(v)))
    if not pairs:
        return pd.DataFrame(columns=["A","B","ρ"])
    df = pd.DataFrame(pairs, columns=["A","B","ρ"]).sort_values("ρ", ascending=False).head(k).reset_index(drop=True)
    return df

def token_focus_table(C: pd.DataFrame, token: str, min_corr: float, max_corr: float, k=50) -> pd.DataFrame:
    if C is None or C.empty or token not in C.columns:
        return pd.DataFrame(columns=["Token","ρ"])
    s = C[token].drop(index=token, errors="ignore")  # exclude self
    s = s[(s >= min_corr) & (s <= max_corr)]
    s = s.dropna().sort_values(ascending=False).head(k)
    return pd.DataFrame({"Token": s.index, "ρ": s.values})

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Universe size (by mkt cap, stables excluded)", 20, 100, 75, step=5)
    hist_days = st.select_slider("History to fetch (days)", options=[90, 180, 270, 365], value=365)
    corr_win = st.selectbox("Correlation lookback", ["30D", "90D"], index=1)
    vol_win  = st.selectbox("Volatility roll window (days)", [7, 14, 30], index=2)
    min_cov  = st.slider("Min % asset coverage at end date", 40, 100, 60, step=5)

    st.markdown("**Correlation filter (applies to plots & tables):**")
    min_corr = st.slider("Min correlation", 0.00, 1.00, 0.90, step=0.01)
    max_corr = st.slider("Max correlation", 0.00, 1.00, 0.99, step=0.01)

    st.markdown("**Token focus:**")
    # placeholder; real choices populated after we know symbols
    focus_placeholder = st.empty()

    st.divider()
    if st.button("🔄 Clear cache & reload", type="primary"):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# ---------------- Load universe & prices ----------------
ids, syms, _raw = fetch_universe_top_n(top_n=top_n)
if not ids:
    st.error("Failed to fetch universe.")
    st.stop()

progress = st.progress(0.0, text="Fetching price history …")
series = {}
fails  = []
for i, cid in enumerate(ids):
    s = fetch_hist_daily_series(cid, days=hist_days)
    if s is not None:
        series[cid] = s
    else:
        fails.append(cid)
    progress.progress((i + 1) / len(ids), text=f"{i+1}/{len(ids)} ok={len(series)} fail={len(fails)}")
    time.sleep(0.12)  # throttle to be gentle
progress.empty()

if not series:
    st.error("No historical data returned. Try smaller universe.")
    st.stop()

all_dates = pd.date_range(
    start=min(s.index.min() for s in series.values()),
    end=max(s.index.max() for s in series.values()),
    freq="D"
)
aligned = {cid: s.reindex(all_dates).ffill(limit=3) for cid, s in series.items()}
prices = pd.DataFrame(aligned)
id2sym = dict(zip(ids, syms))
prices.columns = [id2sym.get(c, c).upper() for c in prices.columns]
prices = prices.replace([np.inf, -np.inf], np.nan)
prices = prices.where(prices > 0)

# Populate token focus selector now that we know columns
with st.sidebar:
    focus_token = focus_placeholder.selectbox(
        "Pick token for focus",
        options=sorted(prices.columns.tolist()),
        index=sorted(prices.columns.tolist()).index("ETH") if "ETH" in prices.columns else 0
    )

st.write(f"Prices matrix: {prices.shape[1]} assets × {prices.shape[0]} days.")

# ---------------- Window selection ----------------
coverage = prices.notna().mean(axis=1)
th = min_cov / 100.0
eligible = coverage[coverage >= th]
if len(eligible) > 0:
    end_date = eligible.index[-1]
else:
    end_date = coverage.idxmax()

LOOK_MAP = {"30D": 30, "90D": 90}
wdays = LOOK_MAP[corr_win]
start_date = end_date - pd.Timedelta(days=wdays)
st.caption(f"Analysis window: {start_date.date()} → {end_date.date()}  •  coverage @ end: {coverage.loc[end_date]:.1%}")

# ---------------- Metrics ----------------
ret_full = compute_returns(prices)
vol_full = realized_vol(ret_full, window=int(vol_win))

ret_w = ret_full.loc[start_date:end_date]
vol_w = vol_full.loc[start_date:end_date]

min_obs = max(3, wdays // 6)
# ensure enough data inside the window
ret_w = ret_w.loc[:, ret_w.count() >= min_obs]
vol_w = vol_w.loc[:, vol_w.count() >= max(2, int(vol_win)//2)]

C_price = pairwise_corr(ret_w, min_overlap=min_obs)
C_vol   = pairwise_corr(vol_w, min_overlap=max(2, int(vol_win)//2))

# Apply correlation range mask for visuals/tables
C_price_masked = mask_corr_range(C_price, min_corr=min_corr, max_corr=max_corr)
C_vol_masked   = mask_corr_range(C_vol,   min_corr=min_corr, max_corr=max_corr)

# ---------------- UI: tabs ----------------
tab1, tab2, tab3 = st.tabs(["📈 Price correlation", "📊 Volatility correlation", "🎯 Token focus"])

with tab1:
    render_heatmap(C_price_masked, f"Price correlation — show {min_corr:.2f} to {max_corr:.2f}", min_corr, max_corr, key="hm_price")
    tp = top_pairs_from_matrix(C_price_masked, k=50)
    if not tp.empty:
        st.subheader("Top pairs by price correlation (filtered)")
        st.dataframe(tp, hide_index=True, width="stretch")
        st.download_button("Download price-corr matrix (CSV)", C_price_masked.to_csv().encode(),
                           "price_corr_filtered.csv", "text/csv")

with tab2:
    render_heatmap(C_vol_masked, f"Volatility correlation — show {min_corr:.2f} to {max_corr:.2f} (σ {int(vol_win)}d)", min_corr, max_corr, key="hm_vol")
    tv = top_pairs_from_matrix(C_vol_masked, k=50)
    if not tv.empty:
        st.subheader("Top pairs by volatility correlation (filtered)")
        st.dataframe(tv, hide_index=True, width="stretch")
        st.download_button("Download vol-corr matrix (CSV)", C_vol_masked.to_csv().encode(),
                           "vol_corr_filtered.csv", "text/csv")

with tab3:
    colA, colB = st.columns(2)
    with colA:
        st.subheader(f"{focus_token}: highest price correlations in range")
        tf_price = token_focus_table(C_price, token=focus_token, min_corr=min_corr, max_corr=max_corr, k=100)
        if not tf_price.empty:
            st.dataframe(tf_price, hide_index=True, width="stretch")
            st.download_button(f"Download {focus_token}-price-corrs (CSV)",
                               tf_price.to_csv(index=False).encode(),
                               f"{focus_token}_price_correlations.csv", "text/csv")
        else:
            st.info("No tokens in selected range for price correlation.")
    with colB:
        st.subheader(f"{focus_token}: highest volatility correlations in range")
        tf_vol = token_focus_table(C_vol, token=focus_token, min_corr=min_corr, max_corr=max_corr, k=100)
        if not tf_vol.empty:
            st.dataframe(tf_vol, hide_index=True, width="stretch")
            st.download_button(f"Download {focus_token}-vol-corrs (CSV)",
                               tf_vol.to_csv(index=False).encode(),
                               f"{focus_token}_vol_correlations.csv", "text/csv")
        else:
            st.info("No tokens in selected range for volatility correlation.")

# --------------- Debug (optional) ---------------
with st.expander("🔍 Debug"):
    st.json({
        "universe": len(ids),
        "series_ok": len(series),
        "prices_shape": prices.shape,
        "ret_window_shape": ret_w.shape,
        "vol_window_shape": vol_w.shape,
        "C_price_shape": list(C_price.shape) if not C_price.empty else [0,0],
        "C_vol_shape": list(C_vol.shape) if not C_vol.empty else [0,0],
        "filtered_price_shape": list(C_price_masked.shape) if not C_price_masked.empty else [0,0],
        "filtered_vol_shape": list(C_vol_masked.shape) if not C_vol_masked.empty else [0,0],
        "min_corr": min_corr, "max_corr": max_corr,
        "focus_token": focus_token
    })

st.caption("Tip: To hide clones near 1.0, keep max correlation < 1.0 (e.g., 0.99). Token focus excludes the token itself.")

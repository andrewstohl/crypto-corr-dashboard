# app.py — Token-Focus Correlations (CoinGecko) — lean UI, 365d history, lookbacks 7/14/30/90
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------------------- Page & minimal styling --------------------
st.set_page_config(page_title="Crypto Correlations — Token Focus", layout="wide")

# CSS: tighter top spacing, smaller title, black/blue accents, sidebar right padding
st.markdown("""
<style>
/* Reduce top padding and header height */
header[data-testid="stHeader"] { height: 36px; }
div.block-container { padding-top: 0.6rem; }

/* Smaller H1, black/blue palette */
h1, h2, h3 { color: #0b1220; }
h1 { font-size: 1.4rem; margin-bottom: 0.4rem; }

/* Sidebar right padding and min width so sliders don't clip */
section[data-testid="stSidebar"] { min-width: 360px; }
section[data-testid="stSidebar"] .block-container { padding-right: 14px; }

/* Buttons & metrics: blue accents */
.stButton > button { background:#0b1220; color:#60a5fa; border:1px solid #0b1220; }
.stButton > button:hover { background:#111827; border-color:#111827; }
[data-testid="stMetricValue"] { color:#0b1220; }
[data-testid="stMetricDeltaIndicator"] { color:#2563eb !important; }

/* Dataframe tweaks: compact row height */
div[data-testid="stDataFrame"] table { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔗 Token Focus — Price & Volatility Correlations")

# -------------------- Auth / hosts --------------------
API_KEY = st.secrets.get("COINGECKO_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing COINGECKO_API_KEY in `.streamlit/secrets.toml`.")
    st.stop()

PUBLIC_BASE = "https://api.coingecko.com/api/v3"
PRO_BASE    = "https://pro-api.coingecko.com/api/v3"

def _headers_for(base_url: str):
    # Public (demo) uses x-cg-demo-api-key; Pro uses x-cg-pro-api-key
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

# -------------------- Compact error logging --------------------
error_log = []
def log_err(msg: str):
    error_log.append(msg)

# -------------------- HTTP helper with retry --------------------
def http_get(path: str, params=None, timeout=30, retries=3, backoff=2.0):
    url = f"{CG_BASE}{path}"
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=_headers_for(CG_BASE), timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(backoff * (k + 1))
                continue
            last = f"{r.status_code}"
        except requests.RequestException as e:
            last = f"EXC {str(e)[:100]}"
        time.sleep(0.15)
    log_err(f"GET {path} failed ({last})")
    return None

# -------------------- Universe (top by mkt cap, up to 250) --------------------
STABLE_SYMS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDE","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX","BUSD","EURT","EURS"
}

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_universe_top_n(top_n=100):
    per_page = 250
    page = 1
    r = http_get("/coins/markets", params={
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
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
        if len(ids) >= min(top_n, per_page):
            break
    return ids, syms, pd.DataFrame(rows)

# -------------------- History (365 days daily) --------------------
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily_series(coin_id: str) -> pd.Series | None:
    r = http_get(f"/coins/{coin_id}/market_chart", params={
        "vs_currency": "usd",
        "days": 365,              # fixed per request
        "interval": "daily"
    }, timeout=40, retries=3)
    if not r:
        return None
    data = r.json()
    prices = data.get("prices") or []
    if not prices:
        log_err(f"no-prices:{coin_id}")
        return None
    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
    s = pd.Series(pd.to_numeric(df["price"], errors="coerce").astype(float).values, index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    daily = s.resample("D").last().ffill(limit=3)
    daily = daily[daily > 0]
    if len(daily) < 30:
        log_err(f"short:{coin_id}")
        return None
    daily.name = coin_id
    return daily

# -------------------- Math --------------------
def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(price_df).diff()
    for col in rets.columns:
        x = rets[col]
        if x.notna().sum() > 10:
            rets[col] = x.clip(x.quantile(0.001), x.quantile(0.999))
    return rets

def realized_vol(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
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

def focus_series(C: pd.DataFrame, token: str, rmin: float, rmax: float, topk: int = 100) -> pd.DataFrame:
    if C.empty or token not in C.columns:
        return pd.DataFrame(columns=["Token","ρ"])
    s = C[token].drop(index=token, errors="ignore")
    s = s[(s >= rmin) & (s <= rmax)].dropna().sort_values(ascending=False).head(topk)
    return pd.DataFrame({"Token": s.index, "ρ": s.values})

# -------------------- Sidebar (simple) --------------------
with st.sidebar:
    st.subheader("Token & Filters")

    # Token focus will be populated after data load; placeholder here
    token_placeholder = st.empty()

    st.subheader("Universe")
    top_n = st.slider("Universe size (by mkt cap)", 20, 250, 100, step=10)

    st.subheader("Window")
    corr_win = st.selectbox("Correlation lookback", ["7D", "14D", "30D", "90D"], index=3)
    min_cov  = st.slider("Min % coverage at end date", 40, 100, 60, step=5)

    st.subheader("Correlation range")
    min_corr = st.slider("Min corr", 0.00, 1.00, 0.90, step=0.01)
    max_corr = st.slider("Max corr", 0.00, 1.00, 0.99, step=0.01)

    st.divider()
    if st.button("🔄 Clear cache & reload", type="primary"):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# -------------------- Load data --------------------
ids, syms, _raw = fetch_universe_top_n(top_n=top_n)
if not ids:
    st.error("Failed to fetch universe."); st.stop()

# Fetch prices
progress = st.progress(0.0, text="Loading 365d prices …")
series = {}; fails = []
for i, cid in enumerate(ids):
    s = fetch_hist_daily_series(cid)
    if s is not None:
        series[cid] = s
    else:
        fails.append(cid)
    progress.progress((i + 1) / len(ids), text=f"{i+1}/{len(ids)} ok={len(series)} fail={len(fails)}")
    time.sleep(0.08)
progress.empty()

# Compact failure line (not spam)
if fails:
    st.caption(f"⚠️ Skipped {len(fails)} asset(s) with missing/short history. Examples: " +
               ", ".join(fails[:8]) + ("…" if len(fails) > 8 else ""))

if not series:
    st.error("No price series available. Try smaller universe."); st.stop()

# Aligned price matrix
all_dates = pd.date_range(
    start=min(s.index.min() for s in series.values()),
    end=max(s.index.max() for s in series.values()),
    freq="D"
)
aligned = {cid: s.reindex(all_dates).ffill(limit=3) for cid, s in series.items()}
prices = pd.DataFrame(aligned)

id2sym = dict(zip(ids, syms))
prices.columns = [id2sym.get(c, c).upper() for c in prices.columns]
prices = prices.replace([np.inf, -np.inf], np.nan).where(prices > 0)

# Sidebar token focus now that we know columns
with st.sidebar:
    focus_token = token_placeholder.selectbox(
        "Token focus",
        options=sorted(prices.columns.tolist()),
        index=sorted(prices.columns.tolist()).index("ETH") if "ETH" in prices.columns else 0
    )

st.write(f"Universe loaded: {prices.shape[1]} assets × {prices.shape[0]} days of prices.")

# -------------------- Window selection by coverage --------------------
coverage = prices.notna().mean(axis=1)
th = min_cov / 100.0
eligible = coverage[coverage >= th]
end_date = eligible.index[-1] if len(eligible) > 0 else coverage.idxmax()

LOOK_MAP = {"7D": 7, "14D": 14, "30D": 30, "90D": 90}
wdays = LOOK_MAP[corr_win]
start_date = end_date - pd.Timedelta(days=wdays)

st.write(f"Window: {start_date.date()} → {end_date.date()}  •  coverage @ end: {coverage.loc[end_date]:.1%}")

# -------------------- Metrics & correlations --------------------
ret_full = compute_returns(prices)
vol_full = realized_vol(ret_full, window=30)  # fixed 30d volatility for simplicity

ret_w = ret_full.loc[start_date:end_date]
vol_w = vol_full.loc[start_date:end_date]

min_obs = max(3, wdays // 6)
ret_w = ret_w.loc[:, ret_w.count() >= min_obs]
vol_w = vol_w.loc[:, vol_w.count() >= max(2, 30//2)]

C_price = pairwise_corr(ret_w, min_overlap=min_obs)
C_vol   = pairwise_corr(vol_w, min_overlap=max(2, 30//2))

# -------------------- Token focus only --------------------
st.subheader(f"🎯 {focus_token} — Highest correlations in range [{min_corr:.2f}, {max_corr:.2f}]")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Price correlation (ρ)**")
    tf_price = focus_series(C_price, focus_token, min_corr, max_corr, topk=150)
    if not tf_price.empty:
        st.dataframe(tf_price, hide_index=True, width="stretch")
        st.download_button(
            f"Download {focus_token} price correlations (CSV)",
            tf_price.to_csv(index=False).encode(),
            f"{focus_token}_price_correlations.csv", "text/csv"
        )
    else:
        st.info("No tokens in the selected range for price correlation.")

with col2:
    st.markdown("**Volatility correlation (ρ of σ(30d))**")
    tf_vol = focus_series(C_vol, focus_token, min_corr, max_corr, topk=150)
    if not tf_vol.empty:
        st.dataframe(tf_vol, hide_index=True, width="stretch")
        st.download_button(
            f"Download {focus_token} volatility correlations (CSV)",
            tf_vol.to_csv(index=False).encode(),
            f"{focus_token}_vol_correlations.csv", "text/csv"
        )
    else:
        st.info("No tokens in the selected range for volatility correlation.")

# -------------------- Compact API error note (bottom) --------------------
if error_log:
    st.caption(f"API issues on {len(error_log)} request(s).")
    with st.expander("See error details"):
        for e in error_log[:200]:
            st.text(e)

# Footer meta (moved to bottom as requested)
st.caption(f"Source: {CG_BASE}  •  Key: …{API_KEY[-4:]}  •  History: 365d  •  Vol window: 30d (fixed)")

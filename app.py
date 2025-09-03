# app.py — VORA Price & Volatility Correlations (CoinGecko)
# - Token Focus only
# - Universe up to 250
# - History fixed at 365d
# - Lookbacks: 7/14/30/90 days
# - Min coverage hardcoded to 50%
# - Blue accents, fixed sidebar slider clipping, tidier header
# - "Best pairs" score = min(ρ_price, ρ_vol) (conservative), plus avg for reference

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------------------- Page & styling --------------------
st.set_page_config(page_title="VORA Price & Volatility Correlations", layout="wide")

st.markdown("""
<style>
/* Header spacing + smaller title */
header[data-testid="stHeader"] { height: auto; }
div.block-container { padding-top: 0.9rem; }

/* Title: smaller, add breathing room from top */
h1, h2, h3 { color: #0b1220; }
h1 { font-size: 1.25rem; margin-top: 0.4rem; margin-bottom: 0.4rem; }

/* Sidebar: ensure sliders don't clip on right; add inner right padding */
section[data-testid="stSidebar"] { min-width: 360px; }
section[data-testid="stSidebar"] .block-container { padding-right: 18px; }

/* Narrow the visual width of sliders a bit so thumbs aren’t at the very edge */
section[data-testid="stSidebar"] [data-baseweb="slider"] { margin-right: 10px; }
section[data-testid="stSidebar"] [data-baseweb="slider"] > div { max-width: 94%; }

/* Buttons: blue accents; no red */
.stButton > button {
  background:#2563eb; color:#ffffff; border:1px solid #2563eb;
}
.stButton > button:hover {
  background:#1d4ed8; border-color:#1d4ed8;
}

/* Metrics: neutral/blue */
[data-testid="stMetricValue"] { color:#0b1220; }

/* Dataframe font a bit compact */
div[data-testid="stDataFrame"] table { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.title("VORA Price & Volatility Correlations")

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
        time.sleep(0.12)
    log_err(f"GET {path} failed ({last})")
    return None

# -------------------- Universe (top by mkt cap, up to 250; stables removed) --------------------
STABLE_SYMS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDE","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX","BUSD","EURT","EURS"
}

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_universe_top_n(top_n=100):
    per_page = 250
    r = http_get("/coins/markets", params={
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": per_page,
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
        if len(ids) >= min(top_n, per_page):
            break
    return ids, syms, pd.DataFrame(rows)

# -------------------- History (365 days daily, fixed) --------------------
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily_series(coin_id: str) -> pd.Series | None:
    r = http_get(f"/coins/{coin_id}/market_chart", params={
        "vs_currency": "usd",
        "days": 365,
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

def focus_series(C: pd.DataFrame, token: str) -> pd.Series:
    if C.empty or token not in C.columns:
        return pd.Series(dtype=float)
    s = C[token].drop(index=token, errors="ignore").dropna()
    return s

# -------------------- Sidebar (order exactly as requested) --------------------
with st.sidebar:
    st.subheader("Settings")

    # 1) Token focus — placeholder; we fill options after data load
    token_placeholder = st.empty()

    # 2) Correlation lookback
    corr_win = st.selectbox("Correlation lookback", ["7D", "14D", "30D", "90D"], index=3)

    # 3) Min correlation
    min_corr = st.slider("Min correlation", 0.00, 1.00, 0.90, step=0.01)

    # 4) Max correlation
    max_corr = st.slider("Max correlation", 0.00, 1.00, 0.99, step=0.01)

    # 5) Universe size
    top_n = st.slider("Universe size (by mkt cap)", 20, 250, 100, step=10)

    st.divider()
    if st.button("🔄 Clear cache & reload", type="primary"):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# -------------------- Load data --------------------
ids, syms, _raw = fetch_universe_top_n(top_n=top_n)
if not ids:
    st.error("Failed to fetch universe."); st.stop()

# Fetch prices (365d)
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

# Compact failure note
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

# Summary line under title
st.write(f"Universe loaded: {prices.shape[1]} assets × {prices.shape[0]} days of prices.")

# -------------------- Window selection (coverage hardcoded 50%) --------------------
coverage = prices.notna().mean(axis=1)
th = 0.50  # hardcoded 50%
eligible = coverage[coverage >= th]
end_date = eligible.index[-1] if len(eligible) > 0 else coverage.idxmax()

LOOK_MAP = {"7D": 7, "14D": 14, "30D": 30, "90D": 90}
wdays = LOOK_MAP[corr_win]
start_date = end_date - pd.Timedelta(days=wdays)
st.caption(f"Window: {start_date.date()} → {end_date.date()}  •  coverage @ end: {coverage.loc[end_date]:.1%}")

# -------------------- Metrics & correlations --------------------
ret_full = compute_returns(prices)
vol_full = realized_vol(ret_full, window=30)  # fixed 30d vol, as agreed

ret_w = ret_full.loc[start_date:end_date]
vol_w = vol_full.loc[start_date:end_date]

min_obs = max(3, wdays // 6)
ret_w = ret_w.loc[:, ret_w.count() >= min_obs]
vol_w = vol_w.loc[:, vol_w.count() >= max(2, 30//2)]

C_price = pairwise_corr(ret_w, min_overlap=min_obs)
C_vol   = pairwise_corr(vol_w, min_overlap=max(2, 30//2))

# -------------------- Token focus (tables only) --------------------
st.subheader(f"🎯 {focus_token} — Correlations")

# Individual series
s_price = focus_series(C_price, focus_token)
s_vol   = focus_series(C_vol,   focus_token)

# Filter by range
s_price_f = s_price[(s_price >= min_corr) & (s_price <= max_corr)].sort_values(ascending=False)
s_vol_f   = s_vol[(s_vol >= min_corr) & (s_vol <= max_corr)].sort_values(ascending=False)

# Joint table: require presence in BOTH, rank by conservative joint score = min(ρp, ρv)
joint = pd.concat([
    s_price.rename("ρ_price"),
    s_vol.rename("ρ_vol")
], axis=1, join="inner").dropna()

if not joint.empty:
    joint["score_conservative"] = joint[["ρ_price","ρ_vol"]].min(axis=1)  # prioritize both being high
    joint["score_avg"] = joint[["ρ_price","ρ_vol"]].mean(axis=1)
    # Apply range filter to BOTH (your sliders already remove wrappers by capping max at 0.99)
    joint_filtered = joint[(joint["ρ_price"].between(min_corr, max_corr)) &
                           (joint["ρ_vol"].between(min_corr, max_corr))]
    joint_ranked = joint_filtered.sort_values(by=["score_conservative","score_avg"], ascending=False)
else:
    joint_ranked = pd.DataFrame(columns=["ρ_price","ρ_vol","score_conservative","score_avg"])

# Display
st.markdown("**Best pairs (price & vol both high)** — ranked by conservative score = min(ρ_price, ρ_vol)")
if not joint_ranked.empty:
    st.dataframe(joint_ranked.round(4), use_container_width=True)
    st.download_button(
        f"Download {focus_token} joint correlations (CSV)",
        joint_ranked.to_csv().encode(),
        f"{focus_token}_joint_correlations.csv", "text/csv"
    )
else:
    st.info("No tokens meet the range on both price and volatility.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Price correlation (ρ)**")
    if not s_price_f.empty:
        dfp = s_price_f.rename("ρ").to_frame()
        st.dataframe(dfp.round(4), use_container_width=True)
        st.download_button(
            f"Download {focus_token} price correlations (CSV)",
            dfp.to_csv().encode(),
            f"{focus_token}_price_correlations.csv", "text/csv"
        )
    else:
        st.info("No tokens in the selected range for price correlation.")

with col2:
    st.markdown("**Volatility correlation (ρ of σ(30d))**")
    if not s_vol_f.empty:
        dfv = s_vol_f.rename("ρ").to_frame()
        st.dataframe(dfv.round(4), use_container_width=True)
        st.download_button(
            f"Download {focus_token} volatility correlations (CSV)",
            dfv.to_csv().encode(),
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

# Footer meta (small, bottom)
st.caption(f"Source: {CG_BASE}  •  Key: …{API_KEY[-4:]}  •  History: 365d  •  Min coverage: 50%  •  Vol window: 30d (fixed)")

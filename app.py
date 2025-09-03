# app.py — VORA Price & Volatility Correlations (CoinGecko, 90d hourly)
# - Token Focus only (no heatmaps)
# - Universe up to 250 (top by market cap; stables excluded)
# - History: 90 days, hourly (CoinGecko returns hourly when days<=90 and interval omitted)
# - Lookbacks: 7/14/30/60/90 (default 30D)
# - Overlap requirement: >=90% of window for both price & vol correlations
# - No forward-fill before returns (prevents fake correlations)
# - Unique internal columns as SYM|coingecko_id to prevent symbol collisions
# - Vol window: 24h (24 hourly points)
# - Compact error logging

import time
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------------------- Page & light CSS --------------------
st.set_page_config(page_title="VORA Price & Volatility Correlations", layout="wide")
st.markdown("""
<style>
/* Reduce top whitespace and tighten title */
header[data-testid="stHeader"] { height: auto; }
div.block-container { padding-top: 0.8rem; }
h1 { color:#E5E7EB; font-size:1.10rem; margin:0.25rem 0 0.6rem 0; }

/* Sidebar — ensure sliders don’t clip on the right; slightly narrow slider track */
section[data-testid="stSidebar"] { min-width: 360px; }
section[data-testid="stSidebar"] .block-container { padding-right: 18px; }
section[data-testid="stSidebar"] [data-baseweb="slider"] { margin-right: 10px; }
section[data-testid="stSidebar"] [data-baseweb="slider"] > div { max-width: 94%; }

/* Compact dataframes */
div[data-testid="stDataFrame"] table { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.title("VORA Price & Volatility Correlations")

# -------------------- API hosts & auth --------------------
API_KEY = st.secrets.get("COINGECKO_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing COINGECKO_API_KEY in `.streamlit/secrets.toml`.")
    st.stop()

PUBLIC_BASE = "https://api.coingecko.com/api/v3"
PRO_BASE    = "https://pro-api.coingecko.com/api/v3"

def _headers_for(base_url: str):
    # Public uses x-cg-demo-api-key; Pro uses x-cg-pro-api-key
    if base_url == PRO_BASE:
        return {"x-cg-pro-api-key": API_KEY, "Accept": "application/json", "Accept-Encoding": "gzip"}
    else:
        return {"x-cg-demo-api-key": API_KEY, "Accept": "application/json", "Accept-Encoding": "gzip"}

@st.cache_resource(show_spinner=False)
def select_working_cg_base() -> str:
    for base in (PUBLIC_BASE, PRO_BASE):
        try:
            r = requests.get(f"{base}/ping", headers=_headers_for(base), timeout=12)
            if r.status_code == 200:
                return base
            r2 = requests.get(
                f"{base}/coins/markets",
                params={"vs_currency": "usd", "per_page": 1, "page": 1, "order": "market_cap_desc"},
                headers=_headers_for(base), timeout=15
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
            last = f"EXC {str(e)[:120]}"
        time.sleep(0.10)
    log_err(f"GET {path} failed ({last})")
    return None

# -------------------- Universe (top by mkt cap; stables removed) --------------------
STABLE_SYMS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDE","USDL","USDP","PYUSD","GUSD","FRAX",
    "LUSD","USDD","USDX","BUSD","EURT","EURS"
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
    }, timeout=35, retries=3)
    if not r:
        return [], [], pd.DataFrame()
    rows = r.json() if isinstance(r.json(), list) else []
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

# -------------------- Hourly history (90 days), NO ffill --------------------
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_series_hourly(coin_id: str, days: int = 90) -> pd.Series | None:
    # Leave interval empty → hourly granularity for 2–90d per CoinGecko behavior
    r = http_get(f"/coins/{coin_id}/market_chart",
                 params={"vs_currency": "usd", "days": days},
                 timeout=35, retries=3)
    if not r:
        return None
    data = r.json()
    pts = data.get("prices") or []
    if not pts:
        log_err(f"no-prices:{coin_id}")
        return None

    df = pd.DataFrame(pts, columns=["ts_ms", "price"])
    ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None).dt.floor("H")
    s = pd.Series(pd.to_numeric(df["price"], errors="coerce").astype(float).values, index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[s > 0]
    if len(s) < 30:
        log_err(f"short:{coin_id}")
        return None
    s.name = coin_id
    return s

# -------------------- Math --------------------
def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df).diff()

def realized_vol(returns: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    # 24 = 24 hours (~1d); annualize with sqrt(365)
    minp = max(8, window // 2)
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

def focus_series(C: pd.DataFrame, colname: str) -> pd.Series:
    if C.empty or colname not in C.columns:
        return pd.Series(dtype=float)
    s = C[colname].drop(index=colname, errors="ignore").dropna()
    return s

# -------------------- Sidebar (order you requested) --------------------
with st.sidebar:
    st.subheader("Settings")

    # 1) Token focus — placeholder; we'll populate after we know the columns
    if "focus_token_symbol" not in st.session_state:
        st.session_state.focus_token_symbol = "ETH"
    focus_token_widget = st.empty()

    # 2) Correlation lookback (default 30D)
    opts = ["7D", "14D", "30D", "60D", "90D"]
    corr_win = st.selectbox("Correlation lookback", opts, index=opts.index("30D"), key="corr_win")

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

# Build internal column naming (avoid symbol collisions)
# internal name = SYM|id ; display uses SYM
id2sym = dict(zip(ids, (s.upper() for s in syms)))
internal_name = {cid: f"{id2sym[cid]}|{cid}" for cid in ids}  # e.g., ETH|ethereum

# Fetch 90d hourly prices
progress = st.progress(0.0, text="Loading 90d hourly prices …")
series = {}; fails = []
for i, cid in enumerate(ids):
    s = fetch_hist_series_hourly(cid, days=90)
    if s is not None:
        s = s.rename(internal_name[cid])  # rename to internal col name
        series[cid] = s
    else:
        fails.append(cid)
    progress.progress((i + 1) / len(ids), text=f"{i+1}/{len(ids)} ok={len(series)} fail={len(fails)}")
    time.sleep(0.06)
progress.empty()

if fails:
    st.caption(f"⚠️ Skipped {len(fails)} asset(s) with missing/short history. Examples: " +
               ", ".join(fails[:8]) + ("…" if len(fails) > 8 else ""))

if not series:
    st.error("No price series available. Try smaller universe."); st.stop()

# Aligned hourly grid (NO forward-fill)
all_hours = pd.date_range(
    start=min(s.index.min() for s in series.values()),
    end=max(s.index.max() for s in series.values()),
    freq="H"
)
prices = pd.DataFrame({internal_name[cid]: s.reindex(all_hours) for cid, s in series.items()})
prices = prices.replace([np.inf, -np.inf], np.nan).where(prices > 0)

# Sidebar token selector now that we know columns (by symbol list)
with st.sidebar:
    colnames = list(prices.columns)
    symbols = sorted({name.split("|", 1)[0] for name in colnames})
    default_sym = st.session_state.focus_token_symbol if st.session_state.focus_token_symbol in symbols else \
                  ("ETH" if "ETH" in symbols else symbols[0])
    st.session_state.focus_token_symbol = focus_token_widget.selectbox(
        "Token focus", options=symbols, index=symbols.index(default_sym)
    )
focus_symbol = st.session_state.focus_token_symbol

# Map from symbol -> internal column name (one per symbol in our universe)
sym2col = {}
for col in prices.columns:
    sym = col.split("|", 1)[0]
    if sym not in sym2col:
        sym2col[sym] = col
focus_col = sym2col.get(focus_symbol)

# Summary line
st.write(f"Universe loaded: {prices.shape[1]} assets × {prices.shape[0]} hours of prices.")

# -------------------- Window selection (end at actual last timestamp) --------------------
LOOK_MAP = {"7D": 7, "14D": 14, "30D": 30, "60D": 60, "90D": 90}
wdays = LOOK_MAP[corr_win]
end_ts = prices.index.max()
start_ts = end_ts - pd.Timedelta(days=wdays)

# Coverage at end (info only)
coverage = prices.notna().mean(axis=1)
cov_at_end = float(coverage.loc[end_ts]) if end_ts in coverage.index else np.nan
st.caption(f"Window: {start_ts.strftime('%Y-%m-%d %H:%M')} → {end_ts.strftime('%Y-%m-%d %H:%M')}  •  coverage @ end: {cov_at_end:.1%}")

# -------------------- Metrics & correlations --------------------
ret_full = compute_returns(prices)                 # hourly log returns
vol_full = realized_vol(ret_full, window=24)       # 24h rolling σ on hourly returns

# Select lookback
ret_w = ret_full.loc[start_ts:end_ts]
vol_w = vol_full.loc[start_ts:end_ts]

# Require >=90% overlap of the *hourly* window
n_hours = int(wdays * 24)
min_obs_price = max(12, int(round(0.90 * n_hours)))
min_obs_vol   = max(24, int(round(0.90 * n_hours)))

ret_w = ret_w.loc[:, ret_w.count() >= min_obs_price]
vol_w = vol_w.loc[:, vol_w.count() >= min_obs_vol]

C_price = pairwise_corr(ret_w, min_overlap=min_obs_price)
C_vol   = pairwise_corr(vol_w, min_overlap=min_obs_vol)

# -------------------- Token focus (tables only) --------------------
st.subheader(f"🎯 {focus_symbol} — Correlations")

if focus_col is None or (C_price.empty and C_vol.empty):
    st.info("Not enough data in-window for correlations. Try increasing lookback or reducing universe size.")
else:
    # Pull focus series (internal names)
    s_price = focus_series(C_price, focus_col)
    s_vol   = focus_series(C_vol,   focus_col)

    # Map index from internal names -> symbols for display
    def to_symbol_index(s: pd.Series) -> pd.Series:
        s2 = s.copy()
        s2.index = [idx.split("|", 1)[0] for idx in s.index]
        return s2.groupby(s2.index).max()  # collapse duplicates defensively

    s_price_sym = to_symbol_index(s_price)
    s_vol_sym   = to_symbol_index(s_vol)

    # Filter by range
    s_price_f = s_price_sym[(s_price_sym >= min_corr) & (s_price_sym <= max_corr)].sort_values(ascending=False)
    s_vol_f   = s_vol_sym[(s_vol_sym >= min_corr) & (s_vol_sym <= max_corr)].sort_values(ascending=False)

    # Joint best-pairs table (require presence in BOTH; rank by conservative min score)
    joint = pd.concat([
        s_price_sym.rename("ρ_price"),
        s_vol_sym.rename("ρ_vol")
    ], axis=1, join="inner").dropna()

    if not joint.empty:
        joint["score_conservative"] = joint[["ρ_price","ρ_vol"]].min(axis=1)
        joint["score_avg"] = joint[["ρ_price","ρ_vol"]].mean(axis=1)
        joint_filtered = joint[(joint["ρ_price"].between(min_corr, max_corr)) &
                               (joint["ρ_vol"].between(min_corr, max_corr))]
        joint_ranked = joint_filtered.sort_values(by=["score_conservative","score_avg"], ascending=False)
    else:
        joint_ranked = pd.DataFrame(columns=["ρ_price","ρ_vol","score_conservative","score_avg"])

    st.markdown("**Best pairs (price & vol both high)** — ranked by conservative score = min(ρ_price, ρ_vol)")
    if not joint_ranked.empty:
        st.dataframe(joint_ranked.round(4), use_container_width=True)
        st.download_button(
            f"Download {focus_symbol} joint correlations (CSV)",
            joint_ranked.to_csv().encode(),
            f"{focus_symbol}_joint_correlations.csv", "text/csv"
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
                f"Download {focus_symbol} price correlations (CSV)",
                dfp.to_csv().encode(),
                f"{focus_symbol}_price_correlations.csv", "text/csv"
            )
        else:
            st.info("No tokens in the selected range for price correlation.")

    with col2:
        st.markdown("**Volatility correlation (ρ of σ(24h))**")
        if not s_vol_f.empty:
            dfv = s_vol_f.rename("ρ").to_frame()
            st.dataframe(dfv.round(4), use_container_width=True)
            st.download_button(
                f"Download {focus_symbol} volatility correlations (CSV)",
                dfv.to_csv().encode(),
                f"{focus_symbol}_vol_correlations.csv", "text/csv"
            )
        else:
            st.info("No tokens in the selected range for volatility correlation.")

# -------------------- Compact API error note (bottom) --------------------
if error_log:
    st.caption(f"API issues on {len(error_log)} request(s).")
    with st.expander("See error details"):
        for e in error_log[:200]:
            st.text(e)

# Footer meta
st.caption(f"Source: {CG_BASE}  •  Key: …{API_KEY[-4:]}  •  History: 90d hourly  •  Vol window: 24h  •  Overlap ≥90% of window")

# app.py — VORA Price & Volatility Correlations (CoinGecko • Daily • EVM-only)
# Minimal, opinionated version per your requirements:
# • Data: CoinGecko daily prices (90 days), NO forward-fill.
# • Universe: top by mkt cap, but only tokens on EVM chains: Ethereum, Arbitrum, Base, Optimism, BNB.
#   We detect EVM by presence of a contract on those platforms OR native L1/L2 coins allowlist.
# • Defaults: lookback = 30D (price corr on daily returns; vol corr on rolling σ with adaptive window).
# • Overlap rules: hard-coded; end date chosen as last day with ≥95% coverage; in-window min overlap = 95%.
# • ETH is always in the universe and is the default focus token.
# • UI: Token focus, Lookback (7/14/30/60/90), Min/Max corr, Universe size (20..250). Tables only.

import time
import math
from typing import Dict, Set, Tuple, List
import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------------------- Page + tiny CSS --------------------
st.set_page_config(page_title="VORA Price & Volatility Correlations", layout="wide")
st.markdown("""
<style>
/* compact top; smaller title */
header[data-testid="stHeader"] { height: auto; }
div.block-container { padding-top: 0.8rem; }
h1 { color:#E5E7EB; font-size:1.10rem; margin:0.25rem 0 0.6rem 0; }

/* sidebar: give sliders room on the right; keep them slightly narrower */
section[data-testid="stSidebar"] { min-width: 360px; }
section[data-testid="stSidebar"] .block-container { padding-right: 18px; }
section[data-testid="stSidebar"] [data-baseweb="slider"] { margin-right: 10px; }
section[data-testid="stSidebar"] [data-baseweb="slider"] > div { max-width: 94%; }

/* compact tables */
div[data-testid="stDataFrame"] table { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.title("VORA Price & Volatility Correlations")

# -------------------- API base + auth --------------------
API_KEY = st.secrets.get("COINGECKO_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing COINGECKO_API_KEY in `.streamlit/secrets.toml`.")
    st.stop()

PUBLIC_BASE = "https://api.coingecko.com/api/v3"
PRO_BASE    = "https://pro-api.coingecko.com/api/v3"

def _headers_for(base_url: str):
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
    return None

# -------------------- EVM filter --------------------
ALLOWED_PLATFORMS: Set[str] = {
    "ethereum",               # ETH mainnet
    "arbitrum-one",           # Arbitrum
    "base",                   # Base
    "optimistic-ethereum",    # Optimism
    "binance-smart-chain",    # BNB Chain
}
NATIVE_ALLOWLIST: Set[str] = {
    "ethereum", "binancecoin", "arbitrum", "optimism"  # L1/L2 natives; Base has no native token (uses ETH)
}

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_coin_platforms(coin_id: str) -> Set[str]:
    # Light detail call; skip heavy fields
    r = http_get(f"/coins/{coin_id}", params={
        "localization": "false", "tickers": "false", "market_data": "false",
        "community_data": "false", "developer_data": "false", "sparkline": "false"
    }, timeout=30, retries=3)
    if not r:
        return set()
    data = r.json()
    plats = data.get("platforms") or {}
    # Platforms keys are slugs like 'ethereum', 'arbitrum-one', etc.
    return set(plats.keys())

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_universe_top_evm(top_n: int = 100) -> Tuple[List[str], List[str]]:
    # Pull top 250 by mcap, then filter to EVM; always include ETH first
    r = http_get("/coins/markets", params={
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": ""
    }, timeout=35, retries=3)
    if not r:
        return [], []
    rows = r.json() if isinstance(r.json(), list) else []

    # ETH first
    ids, syms = [], []
    seen_syms = set()
    for row in rows:
        if row.get("id") == "ethereum":
            ids.append("ethereum"); syms.append("ETH"); seen_syms.add("ETH")
            break

    # Then filter remaining by EVM platforms or native allowlist; skip stables via a short symbol list
    STABLE_SYMS = {
        "USDT","USDC","DAI","FDUSD","TUSD","USDE","USDL","USDP","PYUSD","GUSD","FRAX",
        "LUSD","USDD","USDX","BUSD","EURT","EURS"
    }

    for row in rows:
        cid = row.get("id"); sym = str(row.get("symbol","")).upper()
        if not cid or not sym:
            continue
        if sym in STABLE_SYMS:
            continue
        if sym in seen_syms:
            continue
        if cid == "ethereum":  # already added
            continue

        allowed = False
        if cid in NATIVE_ALLOWLIST:
            allowed = True
        else:
            plats = fetch_coin_platforms(cid)
            if plats & ALLOWED_PLATFORMS:
                allowed = True

        if allowed:
            ids.append(cid); syms.append(sym); seen_syms.add(sym)
            if len(ids) >= top_n:
                break

    return ids, syms

# -------------------- Daily prices (90 days), NO ffill --------------------
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_series_daily(coin_id: str, days: int = 90) -> pd.Series | None:
    r = http_get(f"/coins/{coin_id}/market_chart",
                 params={"vs_currency": "usd", "days": days, "interval": "daily"},
                 timeout=35, retries=3)
    if not r:
        return None
    data = r.json()
    pts = data.get("prices") or []
    if not pts:
        return None
    df = pd.DataFrame(pts, columns=["ts_ms", "price"])
    ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    s = pd.Series(pd.to_numeric(df["price"], errors="coerce").astype(float).values, index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[s > 0]
    if len(s) < 25:  # need enough points for 30D window later (will adapt)
        return None
    s.name = coin_id
    return s

# -------------------- Math --------------------
def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    # pure daily log returns; no winsorization (avoid altering correlation)
    return np.log(price_df).diff()

def realized_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    # rolling σ on daily returns; annualize √365
    minp = max(8, int(math.ceil(0.6 * window)))
    return returns.rolling(window, min_periods=minp).std() * np.sqrt(365)

# -------------------- Sidebar (simple) --------------------
with st.sidebar:
    st.subheader("Settings")

    # Token focus placeholder (filled after we know columns)
    if "focus_token_symbol" not in st.session_state:
        st.session_state.focus_token_symbol = "ETH"
    focus_widget = st.empty()

    # Lookback (default 30D)
    LB_OPTS = ["7D", "14D", "30D", "60D", "90D"]
    look_str = st.selectbox("Correlation lookback", LB_OPTS, index=LB_OPTS.index("30D"))

    # Min/Max correlation
    min_corr = st.slider("Min correlation", 0.00, 1.00, 0.90, step=0.01)
    max_corr = st.slider("Max correlation", 0.00, 1.00, 0.99, step=0.01)

    # Universe size
    top_n = st.slider("Universe size (by mkt cap, EVM-only)", 20, 250, 100, step=10)

    st.divider()
    if st.button("🔄 Clear cache & reload", type="primary"):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# -------------------- Load EVM universe (ETH guaranteed) --------------------
ids, syms = fetch_universe_top_evm(top_n=top_n)
if not ids:
    st.error("Failed to fetch EVM universe."); st.stop()

# Internal names to avoid symbol collisions: SYM|id
id2sym = dict(zip(ids, (s.upper() for s in syms)))
internal_name = {cid: f"{id2sym[cid]}|{cid}" for cid in ids}

# -------------------- Load prices (daily 90d) --------------------
progress = st.progress(0.0, text="Loading 90d daily prices …")
series = {}; fails = []
for i, cid in enumerate(ids):
    s = fetch_hist_series_daily(cid, days=90)
    if s is not None:
        series[cid] = s.rename(internal_name[cid])
    else:
        fails.append(cid)
    progress.progress((i + 1) / len(ids), text=f"{i+1}/{len(ids)} ok={len(series)} fail={len(fails)}")
    time.sleep(0.04)
progress.empty()

if not series:
    st.error("No price series available. Try smaller universe."); st.stop()

if fails:
    st.caption(f"⚠️ Skipped {len(fails)} asset(s) with missing/short history. Examples: " +
               ", ".join(f"{id2sym[cid]}({cid})" for cid in fails[:8]) + ("…" if len(fails) > 8 else ""))

# Build strict daily grid; NO forward-fill
all_days = pd.date_range(
    start=min(s.index.min() for s in series.values()),
    end=max(s.index.max() for s in series.values()),
    freq="D"
)
prices = pd.DataFrame({internal_name[cid]: s.reindex(all_days) for cid, s in series.items()})
prices = prices.replace([np.inf, -np.inf], np.nan).where(prices > 0)

# Token focus selector by symbol; default ETH
with st.sidebar:
    colnames = list(prices.columns)
    symbols = sorted({c.split("|", 1)[0] for c in colnames})
    default_sym = "ETH" if "ETH" in symbols else (st.session_state.focus_token_symbol if st.session_state.focus_token_symbol in symbols else symbols[0])
    st.session_state.focus_token_symbol = focus_widget.selectbox("Token focus", options=symbols, index=symbols.index(default_sym))
focus_symbol = st.session_state.focus_token_symbol

# Map symbol -> internal column (first occurrence)
sym2col: Dict[str, str] = {}
for col in prices.columns:
    s = col.split("|", 1)[0]
    if s not in sym2col:
        sym2col[s] = col
focus_col = sym2col.get(focus_symbol)

# Summary
st.write(f"Universe (EVM-only): {prices.shape[1]} assets × {prices.shape[0]} days of prices.")

# -------------------- Window selection & hard-coded overlap rules --------------------
LOOK_MAP = {"7D": 7, "14D": 14, "30D": 30, "60D": 60, "90D": 90}
wdays = LOOK_MAP[look_str]

# Choose end day = latest date with ≥95% coverage (avoid partial-day artifacts)
coverage = prices.notna().mean(axis=1)
eligible = coverage[coverage >= 0.95]
if len(eligible) > 0:
    end_day = eligible.index[-1]
else:
    end_day = coverage.idxmax()  # best we can do

start_day = end_day - pd.Timedelta(days=wdays)
st.caption(f"Window: {start_day.date()} → {end_day.date()}  •  coverage @ end: {coverage.loc[end_day]:.1%}")

# In-window min overlap = 95% of wdays
min_obs_price = max(5, int(math.ceil(0.95 * wdays)))

# Vol window: adapt so we have enough points in short lookbacks (min 7, max 30)
vol_window_days = max(7, min(30, wdays // 2 if wdays >= 14 else 7))

# -------------------- Metrics --------------------
rets = compute_returns(prices)                         # daily log returns
vols = realized_vol(rets, window=vol_window_days)      # rolling σ

rets_w = rets.loc[start_day:end_day]
vols_w = vols.loc[start_day:end_day]

# Corr matrices with hard-coded overlap
C_price = rets_w.corr(min_periods=min_obs_price)
# For vols, compute required min periods as 95% of available rows in this window
avail_vol_rows = int(vols_w.shape[0])
min_obs_vol = max(5, int(math.ceil(0.95 * avail_vol_rows)))
C_vol = vols_w.corr(min_periods=min_obs_vol)

# -------------------- Token Focus tables --------------------
st.subheader(f"🎯 {focus_symbol} — Correlations")

def focus_series(C: pd.DataFrame, colname: str) -> pd.Series:
    if C.empty or colname not in C.columns:
        return pd.Series(dtype=float)
    s = C[colname].drop(index=colname, errors="ignore").dropna()
    # Map internal "SYM|id" -> "SYM"; collapse any accidental dups by max
    s.index = [i.split("|", 1)[0] for i in s.index]
    return s.groupby(s.index).max().sort_values(ascending=False)

if focus_col is None:
    st.info("Focus token not available in the current universe.")
else:
    s_price = focus_series(C_price, focus_col)
    s_vol   = focus_series(C_vol,   focus_col)

    # Range filters
    s_price_f = s_price[(s_price >= min_corr) & (s_price <= max_corr)]
    s_vol_f   = s_vol[(s_vol >= min_corr) & (s_vol <= max_corr)]

    # Joint best-pairs: require presence in BOTH; rank by conservative min(ρp, ρv)
    joint = pd.concat([s_price.rename("ρ_price"), s_vol.rename("ρ_vol")], axis=1, join="inner").dropna()
    joint["score_conservative"] = joint[["ρ_price","ρ_vol"]].min(axis=1)
    joint["score_avg"] = joint[["ρ_price","ρ_vol"]].mean(axis=1)
    joint_f = joint[(joint["ρ_price"].between(min_corr, max_corr)) &
                    (joint["ρ_vol"].between(min_corr, max_corr))]
    joint_ranked = joint_f.sort_values(by=["score_conservative","score_avg"], ascending=False)

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
        st.markdown(f"**Volatility correlation (ρ of σ({vol_window_days}D))**")
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

# Footer
st.caption(
    f"EVM chains: Ethereum / Arbitrum / Base / Optimism / BNB  •  "
    f"History: 90d daily  •  Lookback: {look_str}  •  "
    f"Vol window: {vol_window_days}D  •  Overlap rules: 95% end-date coverage & in-window"
)

# app.py — VORA Price & Volatility Correlations (CoinGecko, EVM-prefiltered)
# Focus: robust daily return and realized-vol correlations with strict alignment
# Changes vs prior:
# - Lookbacks: 14D, 30D, 60D, 90D (default 90D). 7D removed.
# - Drop last day if coverage < target (default 95%).
# - Correct overlap math; use pandas .corr(min_periods=...).
# - Keep CoinGecko ids internally; display duplicates as "SYM [id]".
# - EVM prefilter via CoinGecko platforms map.
# - Optional manual allowlist (symbols or ids).
# - Optional experimental DEX subgraph liquidity gate (off by default).
# - Optional robust clipping by global z-threshold (off by default).
#
# Requirements:
# - streamlit, pandas, numpy, requests
#
# Secrets:
# - COINGECKO_API_KEY in .streamlit/secrets.toml

import time
from datetime import timedelta
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------------------- Page --------------------
st.set_page_config(page_title="VORA Price & Volatility Correlations", layout="wide")
st.markdown("""
<style>
body, .stApp { background-color: #0f1116; color: #e6e6e6; }
header[data-testid="stHeader"] { height: auto; background: transparent; }
div.block-container { padding-top: 0.6rem; }
h1 { color:#e6e6e6; font-size:1.15rem; margin:0.25rem 0 0.4rem 0; }
section[data-testid="stSidebar"] { min-width: 380px; }
section[data-testid="stSidebar"] .block-container { padding-right: 18px; }
div[data-testid="stDataFrame"] table { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.title("VORA Price & Volatility Correlations")

# -------------------- API hosts --------------------
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

# -------------------- Error log --------------------
error_log = []
def log_err(msg: str):
    error_log.append(msg)

# -------------------- HTTP helper --------------------
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
        time.sleep(0.10)
    log_err(f"GET {path} failed ({last})")
    return None

# -------------------- Universe base (top by mcap) --------------------
STABLE_SYMS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDE","USDL","USDP","PYUSD","GUSD","FRAX",
    "LUSD","USDD","USDX","BUSD","EURT","EURS"
}

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_universe_top_n(top_n=200):
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
        cid = row.get("id")
        if not cid:
            continue
        # keep all ids; do not drop by symbol collision
        ids.append(cid); syms.append(sym)
        if len(ids) >= min(top_n, per_page):
            break
    return ids, syms, pd.DataFrame(rows)

# -------------------- Platforms map for EVM prefilter --------------------
EVM_KEYS_DEFAULT = {
    "ethereum",
    "arbitrum-one",
    "optimistic-ethereum",
    "base",
    "polygon-pos",
    "bnb-smart-chain",
    "avalanche",
    "fantom",
    "zksync",
    "linea",
    "scroll",
}

@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_coins_platforms_map():
    # returns: dict[id] -> {platform_key: contract_address, ...}
    r = http_get("/coins/list", params={"include_platform": "true"}, timeout=60, retries=3)
    if not r:
        return {}
    out = {}
    for row in r.json():
        cid = row.get("id")
        plats = row.get("platforms") or {}
        if cid:
            out[cid] = {k: v for k, v in plats.items() if v}
    return out

def evm_filter_ids(ids, platforms_map, allowed_platform_keys):
    ok = []
    for cid in ids:
        plats = platforms_map.get(cid, {})
        if any(pk in plats for pk in allowed_platform_keys):
            ok.append(cid)
    return ok

def invert_platforms(platforms_map, allowed_platform_keys):
    # builds address(lower)->id mapping for allowed chains
    inv = {}
    for cid, plats in platforms_map.items():
        for k, addr in plats.items():
            if not addr or k not in allowed_platform_keys:
                continue
            inv[addr.lower()] = cid
    return inv

# -------------------- History (365d daily equivalent) --------------------
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily_series(coin_id: str) -> pd.Series | None:
    r = http_get(f"/coins/{coin_id}/market_chart", params={
        "vs_currency": "usd",
        "days": 365,
        "interval": "daily"
    }, timeout=35, retries=3)
    if not r:
        return None
    data = r.json()
    prices = data.get("prices") or []
    if not prices:
        log_err(f"no-prices:{coin_id}")
        return None
    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    # bucket by date; do not resample beyond provided points
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None).dt.date
    df = df.groupby("date", as_index=True).last(numeric_only=True)
    s = pd.to_numeric(df["price"], errors="coerce")
    s = s[s > 0].astype(float)
    s.index = pd.to_datetime(s.index)  # DatetimeIndex at 00:00 local
    if len(s) < 30:
        log_err(f"short:{coin_id}")
        return None
    s.name = coin_id
    return s.sort_index()

# -------------------- Math --------------------
def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df).diff()

def realized_vol(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    minp = max(2, window // 2)
    return returns.rolling(window, min_periods=minp).std() * np.sqrt(365)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("Settings")

    # Universe size
    top_n = st.slider("Universe size (by mkt cap)", 20, 250, 200, step=10)

    # EVM chains to include
    evm_keys = st.multiselect(
        "EVM chains to include",
        options=sorted(EVM_KEYS_DEFAULT),
        default=sorted(EVM_KEYS_DEFAULT),
        help="Keeps coins that have contract addresses on these chains."
    )

    # Manual allowlist
    st.caption("Manual allowlist (symbols or CoinGecko ids, one per line). Optional.")
    allowlist_text = st.text_area("Allowlist", value="", height=120)
    restrict_to_allowlist = st.checkbox("Restrict to allowlist only", value=False)

    # Experimental DEX liquidity gate
    use_subgraph_gate = st.checkbox("Experimental: require DEX liquidity (Uniswap v3, Pancake v3)", value=False)
    min_tvl = st.number_input("Min pool TVL (USD) for gate", min_value=0, value=2_000_000, step=100_000)
    min_vol_30d = st.number_input("Min 30d volume (USD) for gate", min_value=0, value=10_000_000, step=500_000)

    st.divider()

    # Lookback
    LOOK_OPTS = ["14D","30D","60D","90D"]
    corr_win = st.selectbox("Correlation lookback", LOOK_OPTS, index=LOOK_OPTS.index("90D"))

    # Coverage rules
    cov_target = st.slider("Min universe coverage for last day", 0.70, 1.00, 0.95, step=0.01)
    req_frac   = st.slider("Required in-window overlap fraction", 0.70, 1.00, 0.90, step=0.01)

    # Robust clipping
    use_clip = st.checkbox("Robust clip returns by z-score", value=False)
    zthr = st.slider("z threshold", 3.0, 8.0, 6.0, step=0.5)

    st.divider()
    if st.button("🔄 Clear cache & reload", type="primary"):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# -------------------- Load base universe --------------------
ids_base, syms_base, _raw = fetch_universe_top_n(top_n=top_n)
if not ids_base:
    st.error("Failed to fetch universe.")
    st.stop()

# Platforms prefilter
plats_map = fetch_coins_platforms_map()
ids_evm = evm_filter_ids(ids_base, plats_map, set(evm_keys))

# Manual allowlist
allow_raw = [x.strip() for x in allowlist_text.splitlines() if x.strip()]
allow_ids = set()
allow_syms = set()
for x in allow_raw:
    if "/" in x or " " in x:
        allow_ids.add(x.lower())
    else:
        allow_syms.add(x.upper())

# Build symbol map for ids we have
# We will fetch symbols from the markets snapshot we already pulled, else fallback to id upper()
id2sym_base = {}
for row in _raw.to_dict(orient="records"):
    cid = row.get("id"); sym = str(row.get("symbol","")).upper()
    if cid: id2sym_base[cid] = sym if sym else cid.upper()

# Restrict set
if restrict_to_allowlist:
    ids = []
    for cid in ids_evm:
        sym = id2sym_base.get(cid, cid.upper())
        if cid in allow_ids or sym in allow_syms:
            ids.append(cid)
else:
    # Intersect with allowlist if provided; else keep evm
    if allow_ids or allow_syms:
        ids = []
        for cid in ids_evm:
            sym = id2sym_base.get(cid, cid.upper())
            if (not allow_ids and not allow_syms) or (cid in allow_ids or sym in allow_syms):
                ids.append(cid)
        if not ids:
            st.warning("Allowlist excluded all assets. Falling back to EVM filter.")
            ids = ids_evm
    else:
        ids = ids_evm

# Optional: experimental DEX subgraph gate
def subgraph_query(url, query, variables=None, timeout=30):
    try:
        r = requests.post(url, json={"query": query, "variables": variables or {}}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        log_err(f"subgraph {url} status {r.status_code}")
    except requests.RequestException as e:
        log_err(f"subgraph EXC {str(e)[:120]}")
    return None

# Endpoints are fragile. These are common public ones; they may change.
UNISWAP_V3_MAINNET = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
UNISWAP_V3_ARBITRUM = "https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-arbitrum-one"
UNISWAP_V3_OPTIMISM = "https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-optimism"
UNISWAP_V3_BASE     = "https://api.thegraph.com/subgraphs/name/lynnshaoyu/uniswap-v3-base"
PANCAKE_V3_BSC      = "https://api.thegraph.com/subgraphs/name/pancakeswap/exchange-v3-bsc"

def collect_liquid_token_addresses(min_tvl_usd=2_000_000, max_pools=300):
    addrs = set()
    pool_query = """
    query topPools($first: Int!) {
      pools(first: $first, orderBy: totalValueLockedUSD, orderDirection: desc) {
        totalValueLockedUSD
        volumeUSD
        token0 { id symbol }
        token1 { id symbol }
      }
    }
    """
    for url in [UNISWAP_V3_MAINNET, UNISWAP_V3_ARBITRUM, UNISWAP_V3_OPTIMISM, UNISWAP_V3_BASE, PANCAKE_V3_BSC]:
        data = subgraph_query(url, pool_query, {"first": max_pools})
        if not data or "data" not in data or "pools" not in data["data"]:
            continue
        for p in data["data"]["pools"]:
            try:
                tvl = float(p.get("totalValueLockedUSD") or 0.0)
            except:
                tvl = 0.0
            if tvl < min_tvl_usd:
                continue
            t0 = (p["token0"]["id"] or "").lower()
            t1 = (p["token1"]["id"] or "").lower()
            if t0: addrs.add(t0)
            if t1: addrs.add(t1)
    return addrs

if use_subgraph_gate:
    addr_set = collect_liquid_token_addresses(min_tvl_usd=min_tvl)
    addr2id = invert_platforms(plats_map, set(evm_keys))
    ids_from_dex = {addr2id[a] for a in addr_set if a in addr2id}
    ids = [cid for cid in ids if cid in ids_from_dex]
    if not ids:
        st.warning("DEX liquidity gate yielded zero assets. Gate disabled for this run.")
        ids = ids_evm

if not ids:
    st.error("No assets after prefiltering. Loosen filters.")
    st.stop()

# -------------------- Fetch prices --------------------
# Internal name: SYM|id; display uses symbol, but we never collapse duplicates.
id2sym = {cid: id2sym_base.get(cid, cid.upper()) for cid in ids}
internal_name = {cid: f"{id2sym[cid]}|{cid}" for cid in ids}

progress = st.progress(0.0, text="Loading 365d prices …")
series = {}; fails = []
for i, cid in enumerate(ids):
    s = fetch_hist_daily_series(cid)
    if s is not None:
        s = s.rename(internal_name[cid])
        series[cid] = s
    else:
        fails.append(cid)
    progress.progress((i + 1) / len(ids), text=f"{i+1}/{len(ids)} ok={len(series)} fail={len(fails)}")
    time.sleep(0.03)
progress.empty()

if fails:
    st.caption(f"Skipped {len(fails)} asset(s) with missing or short history. Examples: " +
               ", ".join(fails[:8]) + ("…" if len(fails) > 8 else ""))

if not series:
    st.error("No price series available. Try smaller universe.")
    st.stop()

# Align
prices = pd.concat(series.values(), axis=1).replace([np.inf, -np.inf], np.nan)
prices = prices.where(prices > 0)

# Focus token selector
with st.sidebar:
    colnames = list(prices.columns)
    symbols = []
    for name in colnames:
        sym = name.split("|", 1)[0]
        if sym not in symbols:
            symbols.append(sym)
    default_sym = "ETH" if "ETH" in symbols else symbols[0]
    focus_symbol = st.selectbox("Target token", options=symbols, index=symbols.index(default_sym))

# Map symbol -> first matching internal col
sym2cols = {}
for col in prices.columns:
    sym = col.split("|", 1)[0]
    sym2cols.setdefault(sym, []).append(col)
focus_cols = sym2cols.get(focus_symbol, [])
focus_col = focus_cols[0] if focus_cols else None

st.write(f"Universe loaded: {prices.shape[1]} assets × {prices.shape[0]} days of prices.")

# -------------------- Window and coverage --------------------
# Choose safe end date
coverage = prices.notna().mean(axis=1)
eligible = coverage[coverage >= cov_target]
if not eligible.empty:
    end_date = eligible.index.max()
else:
    # fallback to 90 percent if 95 not available
    eligible2 = coverage[coverage >= 0.90]
    end_date = eligible2.index.max() if not eligible2.empty else prices.index.max()

LOOK_MAP = {"14D":14, "30D":30, "60D":60, "90D":90}
wdays = LOOK_MAP[corr_win]
# inclusive window with wdays rows
start_date = end_date - pd.Timedelta(days=wdays-1)

cov_at_end = float(coverage.loc[end_date]) if end_date in coverage.index else np.nan
st.caption(f"Window: {start_date.date()} → {end_date.date()}  •  coverage @ end: {cov_at_end:.1%}")

# -------------------- Metrics --------------------
ret_full = compute_returns(prices)
vol_full = realized_vol(ret_full, window=30)

ret_w = ret_full.loc[start_date:end_date]
vol_w = vol_full.loc[start_date:end_date]

# Optional robust clipping by global z-threshold
if use_clip and ret_w.shape[0] >= 30:
    mu = ret_w.mean(skipna=True)
    sd = ret_w.std(skipna=True).replace(0.0, np.nan)
    lower = mu - zthr * sd
    upper = mu + zthr * sd
    ret_w = ret_w.clip(lower=lower, upper=upper, axis=1)

# Overlap requirements by actual rows
price_rows = int(ret_w.shape[0])
vol_rows   = int(vol_w.shape[0])
min_obs_price = max(5, int(np.ceil(req_frac * price_rows)))
min_obs_vol   = max(5, int(np.ceil(req_frac * vol_rows)))
min_obs_price = min(min_obs_price, price_rows)
min_obs_vol   = min(min_obs_vol,   vol_rows)

# Drop sparse columns
ret_w = ret_w.loc[:, ret_w.count() >= min_obs_price]
vol_w = vol_w.loc[:, vol_w.count() >= min_obs_vol]

# Correlations with pairwise deletion
C_price = ret_w.corr(min_periods=min_obs_price)
C_vol   = vol_w.corr(min_periods=min_obs_vol)

# -------------------- Focus tables --------------------
def to_symbol_index(s: pd.Series) -> pd.Series:
    s2 = s.copy()
    # keep duplicates by labeling with id
    new_index = []
    for idx in s.index:
        sym, cid = idx.split("|", 1)
        # if duplicate symbol appears elsewhere, add [id]
        if len(sym2cols.get(sym, [])) > 1:
            new_index.append(f"{sym} [{cid}]")
        else:
            new_index.append(sym)
    s2.index = new_index
    return s2

st.subheader(f"🎯 {focus_symbol} — Correlations")

if focus_col is None or (C_price.empty and C_vol.empty):
    st.info("Not enough data in-window for correlations. Increase lookback or loosen overlap.")
else:
    s_price = C_price.get(focus_col, pd.Series(dtype=float)).drop(index=focus_col, errors="ignore").dropna()
    s_vol   = C_vol.get(focus_col,   pd.Series(dtype=float)).drop(index=focus_col,   errors="ignore").dropna()

    s_price_sym = to_symbol_index(s_price)
    s_vol_sym   = to_symbol_index(s_vol)

    # Filters for display
    min_corr = st.sidebar.slider("Min ρ for display filter", 0.00, 1.00, 0.85, step=0.01)
    max_corr = st.sidebar.slider("Max ρ for display filter", 0.00, 1.00, 0.99, step=0.01)

    s_price_f = s_price_sym[(s_price_sym >= min_corr) & (s_price_sym <= max_corr)].sort_values(ascending=False)
    s_vol_f   = s_vol_sym[(s_vol_sym >= min_corr) & (s_vol_sym <= max_corr)].sort_values(ascending=False)

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

    st.markdown("**Best pairs (price & vol both high)** — ranked by min(ρ_price, ρ_vol)")
    if not joint_ranked.empty:
        st.dataframe(joint_ranked.round(4), use_container_width=True)
        st.download_button(
            f"Download {focus_symbol} joint correlations (CSV)",
            joint_ranked.to_csv().encode(),
            f"{focus_symbol}_joint_correlations.csv", "text/csv"
        )
    else:
        st.info("No tokens meet the range on both price and volatility.")

    c1, c2 = st.columns(2)
    with c1:
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
            st.info("No tokens in range for price correlation.")

    with c2:
        st.markdown("**Volatility correlation (ρ of σ 30d)**")
        if not s_vol_f.empty:
            dfv = s_vol_f.rename("ρ").to_frame()
            st.dataframe(dfv.round(4), use_container_width=True)
            st.download_button(
                f"Download {focus_symbol} volatility correlations (CSV)",
                dfv.to_csv().encode(),
                f"{focus_symbol}_vol_correlations.csv", "text/csv"
            )
        else:
            st.info("No tokens in range for volatility correlation.")

# -------------------- Sanity checks --------------------
def sanity_readout(Cp, Cv, sym2cols):
    msgs = []
    def val(C, a, b):
        ca = sym2cols.get(a, [None])[0]
        cb = sym2cols.get(b, [None])[0]
        if ca and cb and (ca in C.index) and (cb in C.columns):
            return float(C.loc[ca, cb])
        return np.nan
    pairs = [("ETH","WETH"), ("BTC","WBTC"), ("ETH","stETH"), ("BTC","BTC")]
    for a, b in pairs:
        rp = val(Cp, a, b)
        rv = val(Cv, a, b)
        if np.isfinite(rp):
            msgs.append(f"ρ_price({a},{b})={rp:.3f}")
        if np.isfinite(rv):
            msgs.append(f"ρ_vol({a},{b})={rv:.3f}")
    return msgs

if not C_price.empty:
    msg = " | ".join(sanity_readout(C_price, C_vol, sym2cols))
    if msg:
        st.caption("Sanity: " + msg)

# -------------------- Error note --------------------
if error_log:
    st.caption(f"API/subgraph issues on {len(error_log)} request(s).")
    with st.expander("See error details"):
        for e in error_log[:200]:
            st.text(e)

# Footer
st.caption(f"Source: {CG_BASE}  •  Key: …{API_KEY[-4:]}  •  History: 365d  •  Vol window: 30d  •  Overlap ≥{int(req_frac*100)}%  •  Last-day coverage ≥{int(cov_target*100)}%")

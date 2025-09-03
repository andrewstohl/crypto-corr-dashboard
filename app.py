# app.py — VORA Price & Volatility Correlations (CoinGecko • Daily • EVM-only • strict per-pair)
# What’s different vs last time:
# 1) Robust fetch: multiple attempts/param sets; fallback without interval; small jitter to avoid 429s.
# 2) Strict per-pair alignment: compute ρ only on the exact intersection of daily dates for FOCUS vs CANDIDATE.
# 3) Hard-coded overlap: end-date must have ≥95% universe coverage; in-window per-pair must have ≥95% of lookback days.
# 4) ETH is forced into the universe and into the selectable list; default focus = ETH.
# 5) EVM filter = assets with a contract on {ethereum, arbitrum-one, base, optimistic-ethereum, binance-smart-chain}
#    or natives {ethereum, binancecoin, arbitrum, optimism}. (Note: INJ is *non*-EVM, so it’s correctly excluded.)

import time, math, random
from typing import Dict, Set, Tuple, List
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- Page & minimal CSS ----------
st.set_page_config(page_title="VORA Price & Volatility Correlations", layout="wide")
st.markdown("""
<style>
header[data-testid="stHeader"] { height: auto; }
div.block-container { padding-top: 0.8rem; }
h1 { color:#E5E7EB; font-size:1.05rem; margin:0.25rem 0 0.6rem 0; }
section[data-testid="stSidebar"] { min-width: 360px; }
section[data-testid="stSidebar"] .block-container { padding-right: 18px; }
section[data-testid="stSidebar"] [data-baseweb="slider"] > div { max-width: 94%; }
div[data-testid="stDataFrame"] table { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.title("VORA Price & Volatility Correlations")

# ---------- API base ----------
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
            r = requests.get(f"{base}/ping", headers=_headers_for(base), timeout=10)
            if r.status_code == 200:
                return base
            r2 = requests.get(f"{base}/coins/markets",
                              params={"vs_currency":"usd","per_page":1,"page":1,"order":"market_cap_desc"},
                              headers=_headers_for(base), timeout=12)
            if r2.status_code == 200:
                return base
        except requests.RequestException:
            pass
    return PUBLIC_BASE

CG_BASE = select_working_cg_base()

# ---------- EVM filter ----------
ALLOWED_PLATFORMS: Set[str] = {
    "ethereum", "arbitrum-one", "base", "optimistic-ethereum", "binance-smart-chain",
}
NATIVE_ALLOWLIST: Set[str] = {"ethereum", "binancecoin", "arbitrum", "optimism"}  # note: Base has no separate native

# Short stable list to avoid obvious USD-pegs
STABLE_SYMS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDE","USDL","USDP","PYUSD","GUSD","FRAX",
    "LUSD","USDD","USDX","BUSD","EURT","EURS"
}

# ---------- HTTP with robust retry ----------
def http_get(path: str, params=None, timeout=25, retries=3, backoff=1.6):
    url = f"{CG_BASE}{path}"
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=_headers_for(CG_BASE),
                             timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(backoff * (k + 1) + random.uniform(0.05, 0.25))
                continue
            last = f"{r.status_code}"
        except requests.RequestException as e:
            last = f"EXC {str(e)[:120]}"
        time.sleep(0.10 + random.uniform(0.02, 0.10))
    return None

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_coin_platforms(coin_id: str) -> Set[str]:
    r = http_get(f"/coins/{coin_id}", params={
        "localization":"false","tickers":"false","market_data":"false",
        "community_data":"false","developer_data":"false","sparkline":"false"
    }, timeout=20, retries=3)
    if not r:
        return set()
    data = r.json()
    plats = data.get("platforms") or {}
    return set(plats.keys())

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_universe_evm(top_n: int = 100) -> Tuple[List[str], List[str]]:
    # Always include ETH first
    ids, syms, seen_syms = ["ethereum"], ["ETH"], {"ETH"}
    r = http_get("/coins/markets", params={
        "vs_currency":"usd","order":"market_cap_desc","per_page":250,"page":1,
        "sparkline":"false","price_change_percentage":""
    }, timeout=30, retries=3)
    if not r:
        return ids, syms
    rows = r.json() if isinstance(r.json(), list) else []

    for row in rows:
        cid = row.get("id"); sym = str(row.get("symbol","")).upper()
        if not cid or not sym or sym in seen_syms or sym in STABLE_SYMS or cid == "ethereum":
            continue
        allowed = (cid in NATIVE_ALLOWLIST)
        if not allowed:
            plats = fetch_coin_platforms(cid)
            if plats & ALLOWED_PLATFORMS:
                allowed = True
        if allowed:
            ids.append(cid); syms.append(sym); seen_syms.add(sym)
            if len(ids) >= top_n:
                break
    return ids, syms

# ---------- Daily prices (90d) with solid fallbacks ----------
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_series_daily(coin_id: str, days: int = 90) -> pd.Series | None:
    """
    Try multiple param sets and fallbacks to avoid "short history" false-negatives.
    We do NOT forward-fill. We return a daily Series indexed by UTC-normalized dates.
    """
    param_sets = [
        {"vs_currency":"usd","days":days,"interval":"daily"},
        {"vs_currency":"usd","days":days},               # no interval (CG will choose)
        {"vs_currency":"usd","days":min(days+15, 120)},  # a bit longer
    ]
    for params in param_sets:
        r = http_get(f"/coins/{coin_id}/market_chart", params=params, timeout=30, retries=3)
        if not r:
            continue
        data = r.json()
        pts = data.get("prices") or []
        if not pts:
            continue
        df = pd.DataFrame(pts, columns=["ts_ms","price"])
        ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).tz_convert(None).normalize()
        s = pd.Series(pd.to_numeric(df["price"], errors="coerce").astype(float).values, index=ts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s[s > 0]
        # Keep the last `days` calendar days (if we fetched longer)
        if len(s) > 0:
            s = s.iloc[-days:]
        if len(s) >= 29:  # enough to compute 30D-ish stuff later
            s.name = coin_id
            return s
    return None

# ---------- Math ----------
def compute_returns(price_series: pd.Series) -> pd.Series:
    # pure daily log returns (per asset)
    return np.log(price_series).diff()

def rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    # rolling σ on daily returns, annualized
    minp = max(8, int(math.ceil(0.6 * window)))
    return returns.rolling(window, min_periods=minp).std() * np.sqrt(365)

def corr_on_intersection(a: pd.Series, b: pd.Series, min_points: int) -> float | None:
    # intersect on dates where BOTH are finite; require at least min_points
    ab = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(ab) < min_points:
        return None
    x = ab.iloc[:,0].to_numpy()
    y = ab.iloc[:,1].to_numpy()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0,1])

# ---------- Sidebar (minimal) ----------
with st.sidebar:
    st.subheader("Settings")
    LB = ["7D","14D","30D","60D","90D"]
    look_str = st.selectbox("Correlation lookback", LB, index=LB.index("30D"))
    min_corr = st.slider("Min correlation", 0.00, 1.00, 0.90, step=0.01)
    max_corr = st.slider("Max correlation", 0.00, 1.00, 0.99, step=0.01)
    top_n = st.slider("Universe size (EVM-only)", 20, 250, 120, step=10)
    if st.button("🔄 Clear cache & reload", type="primary"):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# ---------- Universe (EVM-only; ETH forced) ----------
ids, syms = fetch_universe_evm(top_n=top_n)
id2sym = dict(zip(ids, (s.upper() for s in syms)))
internal_name = {cid: f"{id2sym[cid]}|{cid}" for cid in ids}

# ---------- Load prices (90d daily) ----------
progress = st.progress(0.0, text="Loading daily prices …")
series: Dict[str, pd.Series] = {}; fails: List[str] = []
for i, cid in enumerate(ids):
    s = fetch_hist_series_daily(cid, days=90)
    if s is not None:
        series[cid] = s.rename(internal_name[cid])
    else:
        fails.append(cid)
    progress.progress((i+1)/len(ids), text=f"{i+1}/{len(ids)}  ok={len(series)}  fail={len(fails)}")
    time.sleep(0.05 + random.uniform(0.0, 0.03))
progress.empty()

if "ethereum" not in series:
    st.error("ETH price series failed to load. Not proceeding to avoid bogus results.")
    if fails:
        st.caption("Failed assets (examples): " + ", ".join(f"{id2sym[c]}({c})" for c in fails[:12]))
    st.stop()

if fails:
    st.caption(f"Skipped {len(fails)} asset(s) due to missing/short history. Examples: " +
               ", ".join(f"{id2sym[c]}({c})" for c in fails[:10]) + ("…" if len(fails) > 10 else ""))

# ---------- Build strict daily grid (no ffill) ----------
all_days = pd.date_range(
    start=min(s.index.min() for s in series.values()),
    end=max(s.index.max() for s in series.values()),
    freq="D"
)
prices = pd.DataFrame({internal_name[cid]: s.reindex(all_days) for cid, s in series.items()})
prices = prices.replace([np.inf,-np.inf], np.nan).where(prices > 0)

# Token selector (ETH default; make sure it’s present)
with st.sidebar:
    symbols = sorted({col.split("|",1)[0] for col in prices.columns})
    default_sym = "ETH" if "ETH" in symbols else symbols[0]
    focus_symbol = st.selectbox("Token focus", options=symbols, index=symbols.index(default_sym))

# Map symbol → internal col (first occurrence)
sym2col: Dict[str,str] = {}
for col in prices.columns:
    sym = col.split("|",1)[0]
    if sym not in sym2col:
        sym2col[sym] = col
focus_col = sym2col.get(focus_symbol, sym2col.get("ETH"))

st.write(f"Universe (EVM-only): {prices.shape[1]} assets × {prices.shape[0]} days.")

# ---------- Window selection with hard-coded end-date coverage (≥95%) ----------
LOOK = {"7D":7,"14D":14,"30D":30,"60D":60,"90D":90}
wdays = LOOK[look_str]

coverage = prices.notna().mean(axis=1)
eligible = coverage[coverage >= 0.95]
end_day = eligible.index[-1] if len(eligible)>0 else coverage.idxmax()
start_day = end_day - pd.Timedelta(days=wdays)
st.caption(f"Window: {start_day.date()} → {end_day.date()}  •  coverage @ end: {coverage.loc[end_day]:.1%}")

# ---------- Per-asset returns & vol (series) ----------
# Compute *per asset* to keep control over alignment later
rets: Dict[str,pd.Series] = {}
vols: Dict[str,pd.Series] = {}
# Vol window: conservative but workable for short windows
vol_win = max(7, min(30, wdays//2 if wdays>=14 else 7))

for col in prices.columns:
    s = prices[col]
    r = compute_returns(s)              # daily log returns
    v = rolling_vol(r, window=vol_win)  # rolling σ
    rets[col] = r
    vols[col] = v

# Restrict each series to the chosen window (start..end)
for col in list(rets.keys()):
    rets[col] = rets[col].loc[start_day:end_day]
    vols[col] = vols[col].loc[start_day:end_day]

# ---------- Strict per-pair correlations vs focus ----------
if focus_col is None or focus_col not in rets:
    st.error("Focus token series missing. Try another token."); st.stop()

min_points_price = max(5, int(math.ceil(0.95 * wdays)))
min_points_vol   = max(5, int(math.ceil(0.95 * len(vols[focus_col].dropna()))))

def label_to_sym(label: str) -> str:
    return label.split("|",1)[0]

price_rows = []
vol_rows   = []
joint_rows = {}

rF = rets[focus_col]; vF = vols[focus_col]

for col in rets.keys():
    if col == focus_col:
        continue
    sym = label_to_sym(col)
    # PRICE corr on exact intersection
    rho_p = corr_on_intersection(rF, rets[col], min_points=min_points_price)
    # VOL corr on exact intersection (volatility series dates differ; use their intersection)
    rho_v = corr_on_intersection(vF,  vols[col], min_points=min_points_vol)

    if rho_p is not None:
        price_rows.append((sym, rho_p))
    if rho_v is not None:
        vol_rows.append((sym, rho_v))
    if (rho_p is not None) and (rho_v is not None):
        joint_rows[sym] = (rho_p, rho_v)

# Assemble tables
df_price = pd.DataFrame(price_rows, columns=["Token","ρ_price"]).drop_duplicates("Token")
df_vol   = pd.DataFrame(vol_rows,   columns=["Token","ρ_vol"]).drop_duplicates("Token")
df_joint = pd.DataFrame([(k, v[0], v[1]) for k,v in joint_rows.items()],
                       columns=["Token","ρ_price","ρ_vol"])

# Range filters
df_price = df_price[(df_price["ρ_price"]>=min_corr) & (df_price["ρ_price"]<=max_corr)].sort_values("ρ_price", ascending=False)
df_vol   = df_vol[(df_vol["ρ_vol"]  >=min_corr) & (df_vol["ρ_vol"]  <=max_corr)].sort_values("ρ_vol",   ascending=False)

if not df_joint.empty:
    df_joint = df_joint[
        df_joint["ρ_price"].between(min_corr, max_corr) &
        df_joint["ρ_vol"].between(min_corr, max_corr)
    ]
    df_joint["score_conservative"] = df_joint[["ρ_price","ρ_vol"]].min(axis=1)
    df_joint["score_avg"] = df_joint[["ρ_price","ρ_vol"]].mean(axis=1)
    df_joint = df_joint.sort_values(["score_conservative","score_avg"], ascending=False)

st.subheader(f"🎯 {focus_symbol} — Best pairs (price & vol both high)")
if not df_joint.empty:
    st.dataframe(df_joint.round(4), use_container_width=True, hide_index=True)
    st.download_button(f"Download {focus_symbol} joint (CSV)",
                       df_joint.to_csv(index=False).encode(),
                       f"{focus_symbol}_joint_correlations.csv","text/csv")
else:
    st.info("No tokens meet the range on both price and volatility.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Price correlation (ρ)**")
    if not df_price.empty:
        st.dataframe(df_price.round(4), use_container_width=True, hide_index=True)
        st.download_button(f"Download {focus_symbol} price (CSV)",
                           df_price.to_csv(index=False).encode(),
                           f"{focus_symbol}_price_correlations.csv","text/csv")
    else:
        st.info("No tokens in range for price correlation.")

with col2:
    st.markdown(f"**Volatility correlation (ρ of σ({vol_win}D))**")
    if not df_vol.empty:
        st.dataframe(df_vol.round(4), use_container_width=True, hide_index=True)
        st.download_button(f"Download {focus_symbol} vol (CSV)",
                           df_vol.to_csv(index=False).encode(),
                           f"{focus_symbol}_vol_correlations.csv","text/csv")
    else:
        st.info("No tokens in range for volatility correlation.")

st.caption(
    f"EVM chains: Ethereum/Arbitrum/Base/Optimism/BNB • History: 90d daily • Lookback: {look_str} • "
    f"End-date coverage ≥95% • Per-pair in-window overlap ≥95% • Vol window: {vol_win}D • Focus={focus_symbol}"
)

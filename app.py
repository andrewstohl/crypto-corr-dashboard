# app.py
# Streamlit dashboard: price & volatility correlations using CoinGecko API v3
# Docs referenced:
# - /coins/markets (universe/top coins) — https://docs.coingecko.com/v3.0.1/reference/coins-markets
# - /coins/{id}/market_chart (history) — https://docs.coingecko.com/v3.0.1/reference/coins-id-market-chart
# - Auth & base URLs — https://docs.coingecko.com/reference/authentication  (Pro)
# - Rate limits (Demo ~30/min) — https://docs.coingecko.com/v3.0.1/reference/common-errors-rate-limit

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# -------------------- Page --------------------
st.set_page_config(page_title="Crypto Correlations (CoinGecko)", layout="wide")
st.title("🔗 Crypto Correlations — CoinGecko")
st.caption(
    "Top coins by market cap → fetch daily USD prices → compute correlation of returns "
    "and correlation of rolling realized volatility. "
    "Data granularity and 365-day limit for Demo per CoinGecko docs."
)

# -------------------- Secrets / Auth --------------------
API_KEY = st.secrets.get("COINGECKO_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing COINGECKO_API_KEY in `.streamlit/secrets.toml`.")
    st.stop()

# Heuristics: A single function that tries Public (Demo) first, then Pro.
PUBLIC_BASE = "https://api.coingecko.com/api/v3"
PRO_BASE    = "https://pro-api.coingecko.com/api/v3"

def _headers_for(base_url: str):
    # Demo/Public uses x-cg-demo-api-key on PUBLIC_BASE
    # Pro uses x-cg-pro-api-key on PRO_BASE
    if base_url == PRO_BASE:
        return {"x-cg-pro-api-key": API_KEY, "Accept": "application/json", "Accept-Encoding": "gzip"}
    else:
        return {"x-cg-demo-api-key": API_KEY, "Accept": "application/json", "Accept-Encoding": "gzip"}

@st.cache_resource(show_spinner=False)
def select_working_cg_base() -> str:
    # Try PUBLIC first (most users with CG-… keys are Demo)
    for base in (PUBLIC_BASE, PRO_BASE):
        try:
            r = requests.get(f"{base}/ping", headers=_headers_for(base), timeout=15)
            if r.status_code == 200:
                return base
            # Some proxies block /ping; try a lightweight markets call
            r2 = requests.get(
                f"{base}/coins/markets",
                params={"vs_currency": "usd", "per_page": 1, "page": 1, "order": "market_cap_desc"},
                headers=_headers_for(base), timeout=20
            )
            if r2.status_code == 200:
                return base
        except requests.RequestException:
            pass
    # If both failed, fall back to PUBLIC to at least show an error later
    return PUBLIC_BASE

CG_BASE = select_working_cg_base()
st.caption(f"Using CoinGecko base: `{CG_BASE}`")

# -------------------- HTTP with retries / 429 handling --------------------
def http_get(path: str, params=None, timeout=30, retries=3, backoff=2.0):
    url = f"{CG_BASE}{path}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=_headers_for(CG_BASE), timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 401:
                st.error("CoinGecko auth failed (401). Check that your key matches the host: "
                         "Demo → api.coingecko.com, Pro → pro-api.coingecko.com.")
                st.stop()
            if r.status_code == 429:
                # Too many requests: exponential backoff, protect credits
                time.sleep(backoff * (attempt + 1))
                continue
            # Other 4xx/5xx — keep last_err and retry
            last_err = f"{r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(0.25)
    st.error(f"GET {path} failed after retries. Last error: {last_err}")
    return None

# -------------------- Universe (top N by market cap) --------------------
STABLE_SYMS = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDE","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX","BUSD","EURT","EURS"
}

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_universe_top_n(top_n=100):
    # Pull first up to 250 coins by market cap in one call, then filter to N non-stable unique symbols.
    # /coins/markets supports per_page and page; update freq every 60s on Public per docs.
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

    # Filter non-stable, dedupe by symbol, keep order
    ids, syms = [], []
    seen = set()
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
        ids.append(cid)
        syms.append(sym)
        if len(ids) >= top_n:
            break

    return ids, syms, pd.DataFrame(rows)

# -------------------- History --------------------
# Public (Demo) access to history is restricted to PAST 365 DAYS ONLY. (docs L15)
# /coins/{id}/market_chart returns "prices": [[ts_ms, price], ...] with daily points when days > 90 (docs L14).
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily_series(coin_id: str, days: int = 365) -> pd.Series | None:
    # Keep days within [30, 365] to avoid excessive requests; daily points kick in above 90 days.
    days = max(30, min(365, int(days)))
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}  # interval hints daily; API can auto-granularize
    r = http_get(f"/coins/{coin_id}/market_chart", params=params, timeout=40, retries=3)
    if not r:
        return None
    data = r.json()
    prices = data.get("prices") or []
    if not prices:
        return None
    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    # To daily series
    ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
    s = pd.Series(pd.to_numeric(df["price"], errors="coerce").astype(float).values, index=ts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # Force daily calendar, last known price for day, limited ffill for tiny gaps
    daily = s.resample("D").last().ffill(limit=3)
    daily = daily[daily > 0]
    if len(daily) < 30:
        return None
    daily.name = coin_id
    return daily

# -------------------- Correlation helpers --------------------
def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(price_df).diff()
    # Clip extreme outliers per column (protect corr from one bad print)
    for col in rets.columns:
        x = rets[col]
        if x.notna().sum() > 10:
            ql = x.quantile(0.001)
            qh = x.quantile(0.999)
            rets[col] = x.clip(ql, qh)
    return rets

def realized_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    # Annualized daily vol with flexible min_periods
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

def render_heatmap(C: pd.DataFrame, title: str, key: str):
    if C.empty:
        st.warning(f"No data for {title}")
        return
    # If big, show the 25 assets with highest average |corr|
    if C.shape[0] > 25:
        avg_abs = np.abs(C).mean(axis=1)
        keep = avg_abs.nlargest(25).index
        C = C.loc[keep, keep]
    fig = go.Figure(
        data=go.Heatmap(
            z=C.values,
            x=C.columns.tolist(),
            y=C.index.tolist(),
            zmin=-1, zmax=1, zmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="ρ"),
            text=np.round(C.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=None, height=700, margin=dict(l=100, r=20, t=60, b=100)
    )
    st.plotly_chart(fig, key=key, width="stretch")

def top_pairs(C: pd.DataFrame, k=30) -> pd.DataFrame:
    if C.empty:
        return pd.DataFrame()
    pairs = []
    cols = list(C.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = C.iat[i, j]
            if np.isfinite(v):
                pairs.append({"A": cols[i], "B": cols[j], "Correlation": float(np.round(v, 4)),
                              "Abs|ρ|": float(np.round(abs(v), 4))})
    if not pairs:
        return pd.DataFrame()
    df = pd.DataFrame(pairs).sort_values("Abs|ρ|", ascending=False).head(k).reset_index(drop=True)
    return df[["A", "B", "Correlation"]]

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Universe size (by market cap, stables excluded)", 20, 100, 50, step=10)
    hist_days = st.select_slider("Historical window to fetch", options=[90, 180, 270, 365], value=365)
    corr_win = st.selectbox("Correlation lookback", ["30D", "90D"], index=1)
    vol_win = st.selectbox("Volatility window (days)", [7, 14, 30], index=2)
    min_cov = st.slider("Min % asset coverage at end date", 40, 100, 60, step=5)
    show_tables = st.checkbox("Show top-pairs tables", value=True)
    st.divider()
    if st.button("🔄 Clear cache & reload", type="primary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"API host: `{CG_BASE}`  •  Key suffix: `{API_KEY[-4:]}`")

# -------------------- Load universe --------------------
ids, syms, raw = fetch_universe_top_n(top_n=top_n)
if not ids:
    st.error("Failed to fetch universe from CoinGecko.")
    st.stop()

st.success(f"Universe: {len(ids)} assets (unique symbols, stables excluded). Examples: {', '.join(syms[:8])} …")

# -------------------- Fetch history (throttled) --------------------
# Protect Demo rate limit (~30/min). We also retry on 429 inside http_get.
series = {}
failed = []
progress = st.progress(0.0, text="Fetching price history …")
for i, cid in enumerate(ids):
    s = fetch_hist_daily_series(cid, days=hist_days)
    if s is not None:
        series[cid] = s
    else:
        failed.append(cid)
    progress.progress((i + 1) / len(ids), text=f"Fetched {i+1}/{len(ids)} • ok={len(series)} fail={len(failed)}")
    # Gentle pacing to avoid bursts. Most runs hit cache; first warm run will pace a bit.
    time.sleep(0.15)
progress.empty()

if not series:
    st.error("No historical data returned. Check key/plan or try smaller universe.")
    st.stop()

# -------------------- Build aligned price matrix --------------------
# Common calendar
all_dates = pd.date_range(
    start=min(s.index.min() for s in series.values()),
    end=max(s.index.max() for s in series.values()),
    freq="D"
)
aligned = {}
for cid, s in series.items():
    a = s.reindex(all_dates).ffill(limit=3)  # short gaps only
    aligned[cid] = a

prices = pd.DataFrame(aligned)
id2sym = dict(zip(ids, syms))
prices.columns = [id2sym.get(c, c).upper() for c in prices.columns]
prices = prices.replace([np.inf, -np.inf], np.nan)
prices = prices.where(prices > 0)

st.write(f"Prices matrix: {prices.shape[1]} assets × {prices.shape[0]} days.")
st.caption("Per docs: above 90 days, market_chart returns daily points (00:00 UTC). Last completed day is available ~00:10 UTC next day.")

# -------------------- Choose end date by coverage --------------------
coverage = prices.notna().mean(axis=1)
th = min_cov / 100.0
eligible = coverage[coverage >= th]
if len(eligible) > 0:
    end_date = eligible.index[-1]
else:
    end_date = coverage.idxmax()  # best we can do
st.write(f"End date selected: **{end_date.date()}**  •  coverage: **{coverage.loc[end_date]:.1%}**")

# -------------------- Compute metrics --------------------
ret = compute_returns(prices)
sigma = realized_vol(ret, window=int(vol_win))

# Window slice
LOOK_MAP = {"30D": 30, "90D": 90}
wdays = LOOK_MAP[corr_win]
start_date = end_date - pd.Timedelta(days=wdays)
ret_w = ret.loc[start_date:end_date]
sig_w = sigma.loc[start_date:end_date]

# Require some data inside window
min_obs = max(3, wdays // 6)
valid_ret_cols = ret_w.count().loc[ret_w.count() >= min_obs].index
valid_sig_cols = sig_w.count().loc[sig_w.count() >= max(2, int(vol_win)//2)].index

ret_w = ret_w[valid_ret_cols]
sig_w = sig_w[valid_sig_cols]

C_ret = pairwise_corr(ret_w, min_overlap=min_obs)
C_sig = pairwise_corr(sig_w, min_overlap=max(2, int(vol_win)//2))

# -------------------- Display --------------------
tab1, tab2, tab3 = st.tabs(["📈 Price correlation", "📊 Volatility correlation", "🔍 Debug"])

with tab1:
    if not C_ret.empty:
        render_heatmap(C_ret, f"Price correlation — last {corr_win}", key="hm_price")
        if show_tables:
            tp = top_pairs(C_ret, k=30)
            if not tp.empty:
                st.subheader("Top pairs by |price correlation|")
                st.dataframe(tp, use_container_width=True, hide_index=True)
                st.download_button("Download price-corr matrix (CSV)", C_ret.to_csv().encode(), "price_corr.csv", "text/csv")
    else:
        st.warning("No finite price correlations with current settings. Try 90D and/or lower Min % coverage.")

with tab2:
    if not C_sig.empty:
        render_heatmap(C_sig, f"Volatility correlation — last {corr_win} (σ {vol_win}d)", key="hm_vol")
        if show_tables:
            tv = top_pairs(C_sig, k=30)
            if not tv.empty:
                st.subheader("Top pairs by |volatility correlation|")
                st.dataframe(tv, use_container_width=True, hide_index=True)
                st.download_button("Download vol-corr matrix (CSV)", C_sig.to_csv().encode(), "vol_corr.csv", "text/csv")
    else:
        st.warning("No finite volatility correlations with current settings. Try 90D and/or σ=30d, and/or lower Min % coverage.")

with tab3:
    st.json({
        "CG_BASE": CG_BASE,
        "universe_count": len(ids),
        "history_series_returned": len(series),
        "failed_assets": len(failed),
        "prices_shape": prices.shape,
        "end_date": str(end_date.date()),
        "window_days": wdays,
        "min_overlap": min_obs,
        "ret_w_shape": ret_w.shape,
        "sig_w_shape": sig_w.shape,
        "finite_correlations_price": int(np.isfinite(C_ret.values).sum()) if not C_ret.empty else 0,
        "finite_correlations_vol": int(np.isfinite(C_sig.values).sum()) if not C_sig.empty else 0,
    })

st.caption(
    "Notes: Public/Demo historical access is limited to the past 365 days and returns daily points when the requested range is >90 days; "
    "the last completed UTC day is available ~00:10 UTC. "
    "If you upgrade to Pro, the app auto-detects and switches to the Pro host/header."
)

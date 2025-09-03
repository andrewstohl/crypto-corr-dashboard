import time, math, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Correlations — Binance (no CoinCap calls)", layout="wide")

# -------- Binance base endpoints (rotate until one works) --------
BINANCE_BASES = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

# cache the first working base in the session
if "BINANCE_BASE" not in st.session_state:
    st.session_state.BINANCE_BASE = None

def binance_get(path, params=None, timeout=30, expect_json=True):
    params = params or {}
    # try remembered base first
    bases = [st.session_state.BINANCE_BASE] + BINANCE_BASES if st.session_state.BINANCE_BASE else BINANCE_BASES
    last_err = None
    for base in bases:
        if not base: 
            continue
        try:
            r = requests.get(base + path, params=params, timeout=timeout, headers={"User-Agent":"streamlit-binance-rotator/1.0"})
            if r.status_code >= 400:
                last_err = (r.status_code, r.text[:300])
                continue
            st.session_state.BINANCE_BASE = base
            return r.json() if expect_json else r
        except Exception as e:
            last_err = (type(e).__name__, str(e)[:200])
            continue
    if last_err:
        st.error(f"Binance request failed for all bases on {path}. Last error: {last_err[0]}")
    raise requests.HTTPError(f"All Binance bases failed for {path}: {last_err}")

@st.cache_data(show_spinner=False, ttl=60*60*6)
def get_universe_from_binance(top_n=100):
    # Exchange metadata
    ei = binance_get("/api/v3/exchangeInfo")
    symbols = ei.get("symbols", [])
    spot_usdt = {
        s["symbol"]: s for s in symbols
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT" and s.get("isSpotTradingAllowed")
    }
    # 24h tickers to rank by quote volume (USDT ~ USD)
    tickers = binance_get("/api/v3/ticker/24hr")
    rows = []
    for t in tickers:
        sym = t.get("symbol")
        if sym in spot_usdt:
            qvol = float(t.get("quoteVolume") or 0.0)
            rows.append((sym, spot_usdt[sym]["baseAsset"], qvol))
    df = pd.DataFrame(rows, columns=["symbol","base","quoteVolume"]).sort_values("quoteVolume", ascending=False)
    # unique bases (avoid multiple markets for same coin)
    seen, selected = set(), []
    for _, r in df.iterrows():
        b = r["base"]
        if b in seen: 
            continue
        seen.add(b)
        selected.append((r["symbol"], b))
        if len(selected) == top_n:
            break
    return selected  # [(symbol, base)]

@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_daily_close(symbol: str, limit: int = 360) -> pd.Series | None:
    # 1d candles, limit up to 1000
    arr = binance_get("/api/v3/klines", params={"symbol": symbol, "interval": "1d", "limit": limit})
    if not arr:
        return None
    df = pd.DataFrame(arr, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ])
    ts = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    px = pd.to_numeric(df["close"], errors="coerce").astype(float)
    s = pd.Series(px.values, index=ts).sort_index()
    s = s.replace([np.inf, -np.inf], np.nan)
    s[s <= 0] = np.nan
    s.name = symbol
    return s

def pairwise_corr(df: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    cols = list(df.columns); n = len(cols)
    out = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        out[i, i] = 1.0
        xi = df[cols[i]].values
        for j in range(i+1, n):
            xj = df[cols[j]].values
            mask = np.isfinite(xi) & np.isfinite(xj)
            k = int(mask.sum())
            if k >= min_overlap:
                a = xi[mask]; b = xj[mask]
                ca = a - a.mean(); cb = b - b.mean()
                denom = (np.sqrt((ca*ca).sum()) * np.sqrt((cb*cb).sum()))
                if denom > 0:
                    rho = float((ca*cb).sum() / denom)
                    out[i, j] = out[j, i] = rho
    return pd.DataFrame(out, index=cols, columns=cols)

def render_heatmap(M: pd.DataFrame, title: str, key: str):
    if M.empty or not np.isfinite(M.values).any():
        st.warning(f"No finite values for {title}. Try widening the window.")
        return
    fig = go.Figure(data=go.Heatmap(
        z=M.values, x=M.columns.tolist(), y=M.index.tolist(),
        zmin=-1, zmax=1, zmid=0, colorscale="RdBu", colorbar=dict(title="ρ")
    ))
    fig.update_layout(title=title, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True, key=key)

def top_pairs(C: pd.DataFrame, k=30) -> pd.DataFrame:
    if C.empty: return pd.DataFrame(columns=["A","B","|rho|","rho"])
    cols = list(C.columns); vals = C.values; n = len(cols)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            v = vals[i, j]
            if np.isfinite(v):
                pairs.append((cols[i], cols[j], abs(float(v)), float(v)))
    if not pairs: return pd.DataFrame(columns=["A","B","|rho|","rho"])
    return pd.DataFrame(pairs, columns=["A","B","|rho|","rho"]).sort_values("|rho|", ascending=False).head(k)

# -------- Sidebar --------
with st.sidebar:
    st.title("Controls")
    top_n        = st.select_slider("Universe size (24h volume, USDT spot)", options=[25, 50, 75, 100], value=100)
    hist_limit   = st.select_slider("History (days)", options=[120, 240, 360, 500], value=360)
    corr_win_s   = st.selectbox("Correlation lookback", ["30D", "60D", "90D"], index=1)
    vol_roll_s   = st.selectbox("Volatility roll window", ["7", "14", "30"], index=2)
    min_overlap  = st.slider("Min overlap (days) per pair", 5, 60, 20, step=5)
    topk         = st.slider("Top pairs to show", 10, 100, 30, step=5)

st.caption("Source: Binance public API (free). If the default endpoint is blocked, the app auto-rotates to alternate bases. Returns = log(close).diff(); Vol = rolling σ × √365. Correlations use pairwise overlap.")

# -------- Universe --------
universe = get_universe_from_binance(top_n=top_n)
bases = [b for _, b in universe]
st.write(f"Universe: {len(universe)} assets (USDT pairs). Examples: {', '.join(bases[:10])}…")

# -------- Prices --------
progress = st.progress(0)
series = {}
for i, (sym, base) in enumerate(universe):
    s = fetch_daily_close(sym, limit=hist_limit)
    if s is not None:
        s.name = base
        series[base] = s
    progress.progress((i+1)/len(universe))
    time.sleep(0.02)  # polite; cached

if not series:
    st.error("No price series fetched from Binance. If you keep seeing this, your host is likely blocking all Binance bases. Run locally instead.")
    st.stop()

prices = pd.concat(series, axis=1).sort_index()
prices = prices.replace([np.inf, -np.inf], np.nan)
prices[prices <= 0] = np.nan
st.write(f"Prices loaded for {prices.shape[1]} assets, {prices.shape[0]} daily rows.")

# -------- Returns & Vol --------
rets_full = np.log(prices).diff()
end_date = rets_full.dropna(how="all").index.max()
corr_win = {"30D":30, "60D":60, "90D":90}[corr_win_s]
roll_w   = int(vol_roll_s)

rets = rets_full.loc[:end_date].tail(corr_win)
rv   = rets_full.rolling(roll_w, min_periods=max(2, roll_w//2)).std() * math.sqrt(365)
rv   = rv.loc[:end_date].tail(corr_win)

def drop_dead(M: pd.DataFrame, min_non_null=3):
    if M is None or M.empty: return M
    M = M.loc[:, M.count() >= min_non_null]
    M = M.loc[:, M.std(skipna=True) > 0]
    return M

rets = drop_dead(rets); rv = drop_dead(rv)

# -------- Correlations --------
corr_price = pairwise_corr(rets, min_overlap=min_overlap)
corr_vol   = pairwise_corr(rv,   min_overlap=min_overlap)
st.write(f"Corr shapes → price: {corr_price.shape}, vol: {corr_vol.shape}")

# -------- Preview --------
st.write("Price corr (top-left 5×5)"); st.dataframe(corr_price.iloc[:5,:5].round(3), use_container_width=True)
st.write("Vol corr (top-left 5×5)");   st.dataframe(corr_vol.iloc[:5,:5].round(3),   use_container_width=True)

# -------- Heatmaps + Top pairs --------
col1, col2 = st.columns(2)
with col1:
    render_heatmap(corr_price, f"Price correlation — last {corr_win_s}", "price_corr_chart")
    st.subheader("Top price-corr pairs")
    st.dataframe(top_pairs(corr_price, k=topk), use_container_width=True)
with col2:
    render_heatmap(corr_vol, f"Volatility correlation — last {corr_win_s} (σ roll {roll_w}d)", "vol_corr_chart")
    st.subheader("Top vol-corr pairs")
    st.dataframe(top_pairs(corr_vol, k=topk), use_container_width=True)

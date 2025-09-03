import time, math, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go

# ---------- CONFIG ----------
BINANCE_API = "https://api.binance.com/api/v3"
st.set_page_config(page_title="Crypto Correlations — Binance (free)", layout="wide")

# ---------- HTTP ----------
def http_get(url, params=None, timeout=60):
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r

# ---------- UNIVERSE: top USDT spot pairs by volume ----------
@st.cache_data(show_spinner=False, ttl=60*60*6)
def get_universe_from_binance(top_n=100):
    # Exchange info (symbols metadata)
    ei = http_get(f"{BINANCE_API}/exchangeInfo").json()
    symbols = ei.get("symbols", [])
    spot_usdt = {
        s["symbol"]: s
        for s in symbols
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT" and s.get("isSpotTradingAllowed")
    }

    # 24h ticker stats (volume ranking)
    tickers = http_get(f"{BINANCE_API}/ticker/24hr").json()
    rows = []
    for t in tickers:
        sym = t.get("symbol")
        if sym in spot_usdt:
            try:
                qvol = float(t.get("quoteVolume", "0"))  # USD-ish since quote is USDT
            except:
                qvol = 0.0
            rows.append((sym, spot_usdt[sym]["baseAsset"], qvol))
    df = pd.DataFrame(rows, columns=["symbol", "base", "quoteVolume"]).sort_values("quoteVolume", ascending=False)
    # take top_n unique base assets (avoid multiple markets for same base)
    seen = set()
    selected = []
    for _, r in df.iterrows():
        base = r["base"]
        if base in seen:
            continue
        seen.add(base)
        selected.append((r["symbol"], base))
        if len(selected) == top_n:
            break
    return selected  # list of (symbol, base)

# ---------- HISTORY: daily klines for a symbol ----------
@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_daily_close(symbol: str, limit: int = 500) -> pd.Series | None:
    # 1d candles, up to 1000 max; limit=500 ~ 1.4 years
    r = http_get(f"{BINANCE_API}/klines", params={"symbol": symbol, "interval": "1d", "limit": limit})
    arr = r.json()
    if not arr:
        return None
    df = pd.DataFrame(arr, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ])
    ts = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    px = pd.to_numeric(df["close"], errors="coerce").astype(float)
    s = pd.Series(px.values, index=ts)
    s.name = symbol
    # sanitize
    s = s.replace([np.inf, -np.inf], np.nan)
    s[s <= 0] = np.nan
    return s

# ---------- CORRELATIONS ----------
def pairwise_corr(df: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    cols = list(df.columns)
    n = len(cols)
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
                    out[i, j] = out[j, i] = float((ca*cb).sum() / denom)
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

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Controls")
    top_n        = st.select_slider("Universe size (by 24h volume)", options=[25, 50, 75, 100], value=100)
    hist_limit   = st.select_slider("History (days)", options=[120, 240, 360, 500], value=360)
    corr_win_s   = st.selectbox("Correlation lookback", ["30D", "60D", "90D"], index=1)
    vol_roll_s   = st.selectbox("Volatility roll window", ["7", "14", "30"], index=2)
    min_overlap  = st.slider("Min overlap (days) per pair", 5, 60, 20, step=5)
    topk         = st.slider("Top pairs to show", 10, 100, 30, step=5)

st.caption("Source: Binance public API (free). Universe = top USDT spot pairs by 24h volume. Returns = log(close).diff(); Vol = rolling σ × √365. Correlations use pairwise overlap with a minimum-days threshold.")

# ---------- UNIVERSE ----------
universe = get_universe_from_binance(top_n=top_n)
bases = [b for _, b in universe]
st.write(f"Universe: {len(universe)} assets (USDT pairs) — examples: {', '.join(bases[:10])}…")

# ---------- PRICES ----------
progress = st.progress(0)
series = {}
for i, (sym, base) in enumerate(universe):
    s = fetch_daily_close(sym, limit=hist_limit)
    if s is not None:
        s.name = base  # label by base asset, e.g., BTC, ETH
        series[base] = s
    progress.progress((i+1)/len(universe))
    time.sleep(0.03)  # polite; cached after first run

if not series:
    st.error("No price series fetched from Binance.")
    st.stop()

prices = pd.concat(series, axis=1).sort_index()
prices = prices.replace([np.inf, -np.inf], np.nan)
prices[prices <= 0] = np.nan
st.write(f"Prices loaded for {prices.shape[1]} assets, {prices.shape[0]} daily rows.")

# ---------- RETURNS & VOL ----------
rets_full = np.log(prices).diff()

# Choose end date = last available across the panel
end_date = rets_full.dropna(how="all").index.max()
corr_win = {"30D":30, "60D":60, "90D":90}[corr_win_s]
roll_w   = int(vol_roll_s)

rets = rets_full.loc[:end_date].tail(corr_win)
rv   = rets_full.rolling(roll_w, min_periods=max(2, roll_w//2)).std() * math.sqrt(365)
rv   = rv.loc[:end_date].tail(corr_win)

# Drop zero-variance columns in-window
def drop_dead(M: pd.DataFrame, min_non_null=3):
    if M is None or M.empty: return M
    M = M.loc[:, M.count() >= min_non_null]
    M = M.loc[:, M.std(skipna=True) > 0]
    return M

rets, rv = drop_dead(rets), drop_dead(rv)

# ---------- CORRELATIONS (pairwise, min overlap) ----------
corr_price = pairwise_corr(rets, min_overlap=min_overlap)
corr_vol   = pairwise_corr(rv,   min_overlap=min_overlap)

st.write(f"Corr shapes → price: {corr_price.shape}, vol: {corr_vol.shape}")

# ---------- PREVIEW ----------
st.write("Price corr (top-left 5×5)")
st.dataframe(corr_price.iloc[:5,:5].round(3), use_container_width=True)
st.write("Vol corr (top-left 5×5)")
st.dataframe(corr_vol.iloc[:5,:5].round(3),   use_container_width=True)

# ---------- HEATMAPS + TOP PAIRS ----------
col1, col2 = st.columns(2)
with col1:
    render_heatmap(corr_price, f"Price correlation — last {corr_win_s}", "price_corr_chart")
    st.subheader("Top price-corr pairs")
    st.dataframe(top_pairs(corr_price, k=topk), use_container_width=True)
with col2:
    render_heatmap(corr_vol, f"Volatility correlation — last {corr_win_s} (σ roll {roll_w}d)", "vol_corr_chart")
    st.subheader("Top vol-corr pairs")
    st.dataframe(top_pairs(corr_vol, k=topk), use_container_width=True)

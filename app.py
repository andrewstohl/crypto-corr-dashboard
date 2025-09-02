import time, math, requests, pandas as pd, numpy as np, streamlit as st
import plotly.express as px
import plotly.graph_objects as go

API_BASE = "https://rest.coincap.io/v3"  # CoinCap v3
STABLE_SYMS = {"USDT","USDC","DAI","FDUSD","TUSD","USDe","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX"}

st.set_page_config(page_title="Crypto Correlations (CoinCap v3)", layout="wide")

API_KEY = st.secrets.get("COINCAP_API_KEY", "")
if not API_KEY:
    st.error("Missing COINCAP_API_KEY in Streamlit Secrets.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept-Encoding": "gzip",
    "User-Agent": "streamlit-crypto-corr/1.0"
}

def http_get(url, params=None, timeout=60):
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
    if r.status_code >= 400:
        st.error(f"HTTP {r.status_code} calling {url}")
        st.code(r.text[:400])
    r.raise_for_status()
    return r

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_top100():
    r = http_get(f"{API_BASE}/assets", params={"limit": 200, "apiKey": API_KEY})
    rows = (r.json().get("data") or [])
    rows = [x for x in rows if x.get("rank")]
    rows.sort(key=lambda x: int(x["rank"]))
    ids, symbols = [], []
    for row in rows:
        sym = str(row.get("symbol","")).upper()
        name = str(row.get("name","")).lower()
        if sym in STABLE_SYMS or "stable" in name:
            continue
        ids.append(row["id"])
        symbols.append(sym)
        if len(ids) == 100:
            break
    return ids, symbols, rows

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily(asset_id: str, start_days: int = 365) -> pd.Series | None:
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=start_days)).timestamp() * 1000)
    r = http_get(
        f"{API_BASE}/assets/{asset_id}/history",
        params={"interval": "d1", "start": start_ms, "end": end_ms, "apiKey": API_KEY},
    )
    arr = r.json().get("data", [])
    if not arr:
        return None
    df = pd.DataFrame(arr)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    s = pd.to_numeric(df["priceUsd"], errors="coerce")
    s = pd.Series(s.values, index=df["ts"]).asfreq("D").ffill()
    s.name = asset_id
    return s

def winsorize(s: pd.Series, q=0.005) -> pd.Series:
    lo, hi = s.quantile(q), s.quantile(1-q)
    return s.clip(lower=lo, upper=hi)

def top_pairs_df(C: pd.DataFrame, k=30) -> pd.DataFrame:
    cols = list(C.columns)
    pairs = []
    n = len(cols)
    for i in range(n):
        for j in range(i+1, n):
            val = C.iat[i, j]
            if np.isfinite(val):
                pairs.append((cols[i], cols[j], abs(float(val)), float(val)))
    if not pairs:
        return pd.DataFrame(columns=["A","B","|rho|","rho"])
    df = pd.DataFrame(pairs, columns=["A","B","|rho|","rho"]).sort_values("|rho|", ascending=False)
    return df.head(k)

# Sidebar
with st.sidebar:
    st.title("Controls")
    lookback_days = st.select_slider("History window", options=[180, 270, 365], value=365)
    corr_window = st.selectbox("Correlation lookback", ["7D", "30D", "90D"], index=0)
    vol_roll = st.selectbox("Volatility roll window", ["7", "30", "90"], index=0)
    min_coverage = st.slider("Min coverage over lookback (%)", 50, 100, 50, step=5)
    topn = st.slider("Top pairs to show", 10, 100, 30, step=5)
    if st.button("Refresh data (clear cache)"):
        fetch_top100.clear(); fetch_hist_daily.clear()
        st.success("Cache cleared. Data will refetch.")

st.caption("Data: CoinCap v3 (API key). Daily prices → log-returns → Pearson corr. Volatility = rolling σ (annualized).")

# Universe
ids, symbols, raw_rows = fetch_top100()
st.write(f"Universe fetched: {len(raw_rows)} rows from API, {len(ids)} non-stable assets selected.")

# Prices
progress = st.progress(0)
series = {}
for i, cid in enumerate(ids):
    s = fetch_hist_daily(cid, start_days=lookback_days)
    if s is not None:
        series[cid] = s
    progress.progress((i+1)/len(ids))
    time.sleep(0.1)

if not series:
    st.error("No price series fetched.")
    st.stop()

prices = pd.concat(series, axis=1).ffill()
id_to_sym = dict(zip(ids, symbols))
prices.columns = [id_to_sym.get(c, c).upper() for c in prices.columns]
st.write(f"Prices loaded for {prices.shape[1]} assets, {prices.shape[0]} daily rows.")

# Filter by coverage on the *price* window, then again on returns
win = {"7D":7, "30D":30, "90D":90}[corr_window]
last_prices = prices.tail(win)
keep_price = [c for c in last_prices.columns if last_prices[c].notna().mean() >= (min_coverage/100)]
prices = prices[keep_price]

# Returns + realized vol
pct = prices.pct_change().apply(winsorize)
rets = np.log1p(pct).dropna(how="all")

ann = math.sqrt(365)
rv = rets.rolling(int(vol_roll)).std() * ann

# Second coverage filter: require enough non-nulls in the actual window used for corr
last_rets = rets.tail(win)
cols_keep = [c for c in last_rets.columns if last_rets[c].notna().sum() >= max(3, win-2)]
rets = rets[cols_keep]
rv = rv[cols_keep]

# Correlations
corr_price = rets.tail(win).corr(min_periods=3)
corr_vol   = rv.tail(win).corr(min_periods=3)

# If everything is NaN (shouldn’t happen), say so plainly
finite_price = int(np.isfinite(corr_price.values).sum())
finite_vol   = int(np.isfinite(corr_vol.values).sum())
st.write(f"Corr shapes → price: {corr_price.shape} (finite={finite_price}), vol: {corr_vol.shape} (finite={finite_vol})")

# Small preview so you can see numbers
st.write("Price corr (top-left 5×5)"); st.dataframe(corr_price.iloc[:5,:5].round(3), use_container_width=True)
st.write("Vol corr (top-left 5×5)");   st.dataframe(corr_vol.iloc[:5,:5].round(3),   use_container_width=True)

# Heatmaps (robust rendering)
def render_heatmap(M: pd.DataFrame, title: str, key: str):
    if M.empty or not np.isfinite(M.values).any():
        st.warning(f"No finite values for {title} — try widening the window or lowering coverage.")
        return
    fig = go.Figure(data=go.Heatmap(
        z=M.values,
        x=M.columns.tolist(),
        y=M.index.tolist(),
        zmin=-1, zmax=1, colorscale="RdBu", zmid=0, colorbar=dict(title="ρ")
    ))
    fig.update_layout(title=title, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True, key=key)

col1, col2 = st.columns(2)
with col1:
    render_heatmap(corr_price, f"Price correlation — last {corr_window}", "price_corr_chart")
    tp = top_pairs_df(corr_price, k=topn)
    st.subheader("Top price-corr pairs")
    st.dataframe(tp, use_container_width=True)
    st.download_button("Download corr (price) CSV", corr_price.to_csv().encode(), file_name="corr_price.csv", mime="text/csv")

with col2:
    render_heatmap(corr_vol, f"Volatility correlation — last {corr_window} (σ roll {vol_roll}d)", "vol_corr_chart")
    tpv = top_pairs_df(corr_vol, k=topn)
    st.subheader("Top vol-corr pairs")
    st.dataframe(tpv, use_container_width=True)
    st.download_button("Download corr (vol) CSV", corr_vol.to_csv().encode(), file_name="corr_vol.csv", mime="text/csv")

st.caption("If a heatmap shows a warning, widen lookback (30D/90D) or lower coverage. Once you have a shortlist, you can check pool liquidity/fees separately.")

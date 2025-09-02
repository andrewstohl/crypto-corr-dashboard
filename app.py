import time, math, requests, pandas as pd, numpy as np, streamlit as st, plotly.express as px

# CoinCap (requires Bearer token). Put COINCAP_API_KEY in Streamlit Secrets.
API_BASE = "https://api.coincap.io/v2"
STABLE_SYMS = {"USDT","USDC","DAI","FDUSD","TUSD","USDe","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX"}

st.set_page_config(page_title="Crypto Correlations (Free, CoinCap)", layout="wide")

# Ensure the key exists
if "COINCAP_API_KEY" not in st.secrets or not st.secrets["COINCAP_API_KEY"]:
    st.error("Missing COINCAP_API_KEY in Streamlit Secrets. Add it under Manage app → Settings → Secrets.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {st.secrets['COINCAP_API_KEY']}",
    "Accept-Encoding": "gzip",
    "User-Agent": "streamlit-crypto-corr/1.0"
}

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_top100():
    # Ranked assets; we ask for 200 and then filter out stables, taking first 100 non-stables.
    r = requests.get(f"{API_BASE}/assets", params={"limit": 200}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    rows = r.json().get("data", [])
    rows = [x for x in rows if x.get("rank")]
    rows.sort(key=lambda x: int(x["rank"]))
    ids, symbols = [], []
    for row in rows:
        sym = str(row.get("symbol","")).upper()
        name = str(row.get("name","")).lower()
        if sym in STABLE_SYMS or "stable" in name:
            continue
        ids.append(row["id"])   # e.g., "bitcoin"
        symbols.append(sym)     # e.g., "BTC"
        if len(ids) == 100:
            break
    return ids, symbols

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily(asset_id: str, start_days: int = 365) -> pd.Series | None:
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=start_days)).timestamp() * 1000)
    r = requests.get(
        f"{API_BASE}/assets/{asset_id}/history",
        params={"interval": "d1", "start": start_ms, "end": end_ms},
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    arr = r.json().get("data", [])
    if not arr:
        return None
    df = pd.DataFrame(arr)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    s = pd.to_numeric(df["priceUsd"], errors="coerce")
    s = pd.Series(s.values, index=df["ts"]).asfreq("D").ffill()
    s.name = asset_id
    return s

def winsorize(series: pd.Series, q=0.005) -> pd.Series:
    lo, hi = series.quantile(q), series.quantile(1-q)
    return series.clip(lower=lo, upper=hi)

def top_pairs(C: pd.DataFrame, k: int = 30) -> pd.DataFrame:
    cols = list(C.columns)
    out = []
    for i, a in enumerate(cols):
        for j in range(i+1, len(cols)):
            rho = float(C.iat[i, j])
            if np.isfinite(rho):
                out.append((a, cols[j], abs(rho), rho))
    out.sort(key=lambda x: x[2], reverse=True)
    return pd.DataFrame(out[:k], columns=["A","B","|rho|","rho"])

with st.sidebar:
    st.title("Controls")
    lookback_days = st.select_slider("History window", options=[180, 270, 365], value=365)
    corr_window = st.selectbox("Correlation lookback", ["7D", "30D", "90D"], index=1)
    vol_roll = st.selectbox("Volatility roll window", ["7", "30", "90"], index=1)
    min_coverage = st.slider("Min coverage over lookback (%)", 50, 100, 80, step=5)
    topn = st.slider("Top pairs to show", 10, 100, 30, step=5)
    refresh = st.button("Refresh data (clear cache)")
    if refresh:
        fetch_top100.clear()
        fetch_hist_daily.clear()
        st.success("Cache cleared. Data will refetch.")

st.caption("Data source: CoinCap (API key). Daily prices; correlations are Pearson on log-returns. Volatility = rolling realized σ (annualized).")

ids, symbols = fetch_top100()
st.write(f"Universe: {len(ids)} assets (top-ranked, stables excluded).")

progress = st.progress(0)
series = {}
for i, cid in enumerate(ids):
    s = fetch_hist_daily(cid, start_days=lookback_days)
    if s is not None:
        series[cid] = s
    progress.progress((i+1)/len(ids))
    time.sleep(0.15)  # polite to the API, still fast enough

if not series:
    st.error("No price series fetched. Try again later.")
    st.stop()

prices = pd.concat(series, axis=1).ffill()
id_to_sym = dict(zip(ids, symbols))
prices.columns = [id_to_sym.get(c, c).upper() for c in prices.columns]

win = {"7D": 7, "30D": 30, "90D": 90}[corr_window]
last = prices.tail(win)
keep = [c for c in last.columns if last[c].notna().mean() >= (min_coverage/100)]
prices = prices[keep]

if prices.shape[1] < 3:
    st.warning("Too few assets after coverage filter. Lower the threshold or widen the window.")
    st.stop()

pct = prices.pct_change()
pct = pct.apply(winsorize)
rets = np.log1p(pct).dropna(how="all")

ann = math.sqrt(365)
vol_window = int(vol_roll)
rv = rets.rolling(vol_window).std() * ann

corr_price = rets.tail(win).corr()
corr_vol   = rv.tail(win).corr()

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Price correlation — last {corr_window}")
    fig = px.imshow(corr_price, zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    tp = top_pairs(corr_price, k=topn)
    st.dataframe(tp, use_container_width=True)
    st.download_button("Download corr (price) CSV", corr_price.to_csv().encode(), file_name="corr_price.csv", mime="text/csv")
with col2:
    st.subheader(f"Volatility correlation — last {corr_window} (σ roll {vol_window}d)")
    fig2 = px.imshow(corr_vol, zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)
    tpv = top_pairs(corr_vol, k=topn)
    st.dataframe(tpv, use_container_width=True)
    st.download_button("Download corr (vol) CSV", corr_vol.to_csv().encode(), file_name="corr_vol.csv", mime="text/csv")

st.caption("Tip: 7D reacts fastest; 30D is a good baseline; 90D sanity-checks stability. High price-corr + high absolute vol is usually best for LPs with lower IL.")

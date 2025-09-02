import time, math, requests, pandas as pd, numpy as np, streamlit as st, plotly.express as px

API_BASE = "https://rest.coincap.io/v3"  # v3 base
STABLE_SYMS = {"USDT","USDC","DAI","FDUSD","TUSD","USDe","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX"}

st.set_page_config(page_title="Crypto Correlations (CoinCap v3)", layout="wide")

# Fail fast if key missing
API_KEY = st.secrets.get("COINCAP_API_KEY", "")
if not API_KEY:
    st.error("Missing COINCAP_API_KEY in Streamlit Secrets (Manage app → Settings → Secrets).")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept-Encoding": "gzip",
    "User-Agent": "streamlit-crypto-corr/1.0"
}

def http_get(url, params=None, timeout=60):
    """Wrapper that surfaces API errors clearly in the app."""
    try:
        r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
        if r.status_code >= 400:
            st.error(f"HTTP {r.status_code} calling {url}")
            # show a small body excerpt to diagnose
            txt = r.text[:300]
            st.code(txt)
        r.raise_for_status()
        return r
    except Exception as e:
        st.exception(e)
        raise

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
    return ids, symbols, rows  # include raw rows for debugging

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily(asset_id: str, start_days: int = 365) -> pd.Series | None:
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=start_days)).timestamp() * 1000)
    r = http_get(f"{API_BASE}/assets/{asset_id}/history",
                 params={"interval": "d1", "start": start_ms, "end": end_ms, "apiKey": API_KEY})
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
    cols = list(C.columns); out = []
    for i, a in enumerate(cols):
        for j in range(i+1, len(cols)):
            rho = float(C.iat[i, j])
            if np.isfinite(rho):
                out.append((a, cols[j], abs(rho), rho))
    out.sort(key=lambda x: x[2], reverse=True)
    return pd.DataFrame(out[:k], columns=["A","B","|rho|","rho"])

# Sidebar controls
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

st.caption("Data: CoinCap v3 (with API key). Daily prices; correlations are Pearson on log-returns. Volatility = rolling realized σ (annualized).")

# DEBUG banner so we always see *something*
st.info("App loaded. Fetching universe…")

ids, symbols, raw_rows = fetch_top100()
st.write(f"Universe fetched: {len(raw_rows)} rows from API, {len(ids)} non-stable assets selected.")
# Show a quick peek so you see data is flowing
st.dataframe(pd.DataFrame(raw_rows[:10]), use_container_width=True)

# Progress + fetch prices
progress = st.progress(0)
series = {}
for i, cid in enumerate(ids):
    s = fetch_hist_daily(cid, start_days=lookback_days)
    if s is not None:
        series[cid] = s
    progress.progress((i+1)/len(ids))
    time.sleep(0.12)

if not series:
    st.error("No price series fetched. If the table above looks fine but this is empty, the /history endpoint may be gated. Try again or reduce lookback.")
    st.stop()

prices = pd.concat(series, axis=1).ffill()
id_to_sym = dict(zip(ids, symbols))
prices.columns = [id_to_sym.get(c, c).upper() for c in prices.columns]
st.write(f"Prices loaded for {prices.shape[1]} assets, {prices.shape[0]} daily rows.")

# Coverage filter over the corr window
win = {"7D": 7, "30D": 30, "90D": 90}[corr_window]
last = prices.tail(win)
keep = [c for c in last.columns if last[c].notna().mean() >= (min_coverage/100)]
prices = prices[keep]

if prices.shape[1] < 3:
    st.warning("Too few assets after coverage filter. Lower the threshold or widen the window.")
    st.stop()

# Returns/vol
pct = prices.pct_change().apply(winsorize)
rets = np.log1p(pct).dropna(how="all")
ann = math.sqrt(365)
rv = rets.rolling(int(vol_roll)).std() * ann

corr_price = rets.tail(win).corr()
corr_vol   = rv.tail(win).corr()

st.write(f"Corr shapes → price: {corr_price.shape}, vol: {corr_vol.shape}")
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Price correlation — last {corr_window}")
    fig = px.imshow(corr_price, zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto")
    st.plotly_chart(fig, use_container_width=True, key="price_corr_chart")
    tp = top_pairs(corr_price, k=topn)
    st.dataframe(tp, use_container_width=True)
    st.download_button("Download corr (price) CSV", corr_price.to_csv().encode(), file_name="corr_price.csv", mime="text/csv")
with col2:
    st.subheader(f"Volatility correlation — last {corr_window} (σ roll {vol_roll}d)")
    fig2 = px.imshow(corr_vol, zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto")
    st.plotly_chart(fig2, use_container_width=True, key="vol_corr_chart")
    tpv = top_pairs(corr_vol, k=topn)
    st.dataframe(tpv, use_container_width=True)
    st.download_button("Download corr (vol) CSV", corr_vol.to_csv().encode(), file_name="corr_vol.csv", mime="text/csv")

st.caption("Tip: 7D reacts fastest; 30D baseline; 90D checks stability.")

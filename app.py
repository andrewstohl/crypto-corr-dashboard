import time, math, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go

API_BASE = "https://rest.coincap.io/v3"
STABLE_SYMS = {"USDT","USDC","DAI","FDUSD","TUSD","USDe","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX"}

st.set_page_config(page_title="Crypto Correlations (CoinCap v3)", layout="wide")

API_KEY = st.secrets.get("COINCAP_API_KEY", "")
if not API_KEY:
    st.error("Missing COINCAP_API_KEY in Streamlit Secrets.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept-Encoding": "gzip",
    "User-Agent": "streamlit-crypto-corr/1.2"
}

def http_get(url, params=None, timeout=60):
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
    if r.status_code >= 400:
        st.error(f"HTTP {r.status_code} calling {url}")
        st.code(r.text[:400])
    r.raise_for_status()
    return r

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_top200():
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
        if len(ids) == 120:   # take some extra; we will de-dup symbols
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
    return s.astype(float)

def top_pairs_df(C: pd.DataFrame, k=30) -> pd.DataFrame:
    if C.empty:
        return pd.DataFrame(columns=["A","B","|rho|","rho"])
    cols = list(C.columns); vals = C.values; pairs=[]
    n = len(cols)
    for i in range(n):
        for j in range(i+1, n):
            v = vals[i, j]
            if np.isfinite(v):
                pairs.append((cols[i], cols[j], abs(float(v)), float(v)))
    if not pairs:
        return pd.DataFrame(columns=["A","B","|rho|","rho"])
    df = pd.DataFrame(pairs, columns=["A","B","|rho|","rho"]).sort_values("|rho|", ascending=False)
    return df.head(k)

def render_heatmap(M: pd.DataFrame, title: str, key: str):
    if M.empty or not np.isfinite(M.values).any():
        st.warning(f"No finite values for {title}. Try 30D/90D or lower coverage.")
        return
    fig = go.Figure(data=go.Heatmap(
        z=M.values, x=M.columns.tolist(), y=M.index.tolist(),
        zmin=-1, zmax=1, zmid=0, colorscale="RdBu", colorbar=dict(title="ρ")
    ))
    fig.update_layout(title=title, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True, key=key)

# Sidebar
with st.sidebar:
    st.title("Controls")
    lookback_days = st.select_slider("History window", options=[180, 270, 365], value=365)
    corr_window = st.selectbox("Correlation lookback", ["7D", "30D", "90D"], index=1)
    vol_roll = st.selectbox("Volatility roll window", ["7", "30", "90"], index=1)
    min_coverage = st.slider("Min coverage over lookback (%)", 50, 100, 80, step=5)
    topn = st.slider("Top pairs to show", 10, 100, 30, step=5)
    if st.button("Refresh data (clear cache)"):
        fetch_top200.clear(); fetch_hist_daily.clear()
        st.success("Cache cleared. Data will refetch.")

st.caption("Data: CoinCap v3 (API key). Returns = log(price).diff(); Vol = rolling σ × √365; Pearson corr on last window.")

# Universe and symbol de-dup
ids, symbols, raw_rows = fetch_top200()
rank_by_id = {row["id"]: int(row["rank"]) for row in raw_rows if row.get("rank")}
sym_to_id = {}
ordered_ids = []
for cid, sym in zip(ids, symbols):
    if sym not in sym_to_id:
        sym_to_id[sym] = cid; ordered_ids.append(cid)
    else:
        if rank_by_id.get(cid, 1e9) < rank_by_id.get(sym_to_id[sym], 1e9):
            sym_to_id[sym] = cid
            ordered_ids = [cid if x == sym_to_id[sym] else x for x in ordered_ids]
keep_ids = ordered_ids[:100]
keep_syms = [next(row["symbol"].upper() for row in raw_rows if row["id"]==cid) for cid in keep_ids]

st.write(f"Universe selected: {len(keep_ids)} assets (unique symbols, stables excluded).")

# Prices
progress = st.progress(0)
series = {}
for i, cid in enumerate(keep_ids):
    s = fetch_hist_daily(cid, start_days=lookback_days)
    if s is not None:
        series[cid] = s
    progress.progress((i+1)/len(keep_ids))
    time.sleep(0.06)  # polite to API

if not series:
    st.error("No price series fetched.")
    st.stop()

prices = pd.concat(series, axis=1).sort_index()
id_to_sym = dict(zip(keep_ids, keep_syms))
prices.columns = [id_to_sym.get(c, c).upper() for c in prices.columns]
prices = prices.apply(pd.to_numeric, errors="coerce")

# Hard sanitize: remove non-positive, infinities
prices = prices.replace([np.inf, -np.inf], np.nan)
prices = prices.where(prices > 0)

st.write(f"Prices loaded for {prices.shape[1]} assets, {prices.shape[0]} daily rows.")

# Window & coverage on prices (need win+1 valid points to form win returns)
win = {"7D":7, "30D":30, "90D":90}[corr_window]
last_prices = prices.tail(win + 1)
ok_cols = [c for c in last_prices.columns if last_prices[c].notna().sum() >= (win + 1) * (min_coverage/100)]
prices = prices[ok_cols]

if prices.shape[1] < 3:
    st.warning("Too few assets after price coverage filter. Lower threshold or widen window.")
    st.stop()

# Returns
rets = np.log(prices).diff()
# Drop columns that in the *active* window have zero variance or too few finite points
R = rets.tail(win)
count_ok = R.count()
R = R.loc[:, count_ok >= max(3, win-2)]  # need at least a few points
R = R.loc[:, R.std(skipna=True) > 0]

# Volatility
roll = int(vol_roll)
rv_full = rets.rolling(roll, min_periods=max(2, roll//2)).std() * math.sqrt(365)
V = rv_full.tail(win)
count_ok_v = V.count()
V = V.loc[:, count_ok_v >= max(3, win-2)]
V = V.loc[:, V.std(skipna=True) > 0]

# Correlations (pairwise complete obs)
corr_price = R.corr(min_periods=2)
corr_vol   = V.corr(min_periods=2)

finite_price = int(np.isfinite(corr_price.values).sum())
finite_vol   = int(np.isfinite(corr_vol.values).sum())
st.write(f"Corr shapes → price: {corr_price.shape} (finite={finite_price}), vol: {corr_vol.shape} (finite={finite_vol})")

# Quick numeric previews
st.write("Price corr (top-left 5×5)"); st.dataframe(corr_price.iloc[:5,:5].round(3), use_container_width=True)
st.write("Vol corr (top-left 5×5)");   st.dataframe(corr_vol.iloc[:5,:5].round(3),   use_container_width=True)

# Heatmaps + top pairs
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

st.caption("If 7D still shows few values, switch to 30D. Short windows can go singular; this build now strips non-positive prices, zero-variance series, and enforces minimum coverage.")

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
    "User-Agent": "streamlit-crypto-corr/1.1"
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
    # build ids + symbols (drop stables)
    ids, symbols = [], []
    for row in rows:
        sym = str(row.get("symbol","")).upper()
        name = str(row.get("name","")).lower()
        if sym in STABLE_SYMS or "stable" in name:
            continue
        ids.append(row["id"])
        symbols.append(sym)
        if len(ids) == 120:  # grab a bit extra; we’ll drop dup symbols
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
    # enforce numeric
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    s = pd.to_numeric(df["priceUsd"], errors="coerce")
    s = pd.Series(s.values, index=df["ts"]).asfreq("D").ffill()
    s.name = asset_id
    return s.astype(float)

def top_pairs_df(C: pd.DataFrame, k=30) -> pd.DataFrame:
    if C.empty: 
        return pd.DataFrame(columns=["A","B","|rho|","rho"])
    cols = list(C.columns)
    vals = C.values
    pairs = []
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

# ---- Sidebar ----
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

st.caption("Data: CoinCap v3 (API key). Returns = log(price).diff(); Pearson corr on the last window. Volatility = rolling σ × √365.")

# ---- Universe ----
ids, symbols, raw_rows = fetch_top200()
# de-duplicate symbols (keep best rank)
rank_by_id = {row["id"]: int(row["rank"]) for row in raw_rows if row.get("rank")}
sym_map = {}
keep_ids = []
for i, (cid, sym) in enumerate(zip(ids, symbols)):
    if sym not in sym_map:
        sym_map[sym] = cid
        keep_ids.append(cid)
    else:
        # keep the one with better (smaller) rank
        old = sym_map[sym]
        if rank_by_id.get(cid, 1e9) < rank_by_id.get(old, 1e9):
            sym_map[sym] = cid
            keep_ids.remove(old)
            keep_ids.append(cid)

# take first 100 non-stable, unique symbols
keep_ids = keep_ids[:100]
keep_syms = [next(row["symbol"].upper() for row in raw_rows if row["id"]==cid) for cid in keep_ids]

st.write(f"Universe selected: {len(keep_ids)} assets (unique symbols, stables excluded).")

# ---- Prices ----
progress = st.progress(0)
series = {}
for i, cid in enumerate(keep_ids):
    s = fetch_hist_daily(cid, start_days=lookback_days)
    if s is not None:
        series[cid] = s
    progress.progress((i+1)/len(keep_ids))
    time.sleep(0.08)

if not series:
    st.error("No price series fetched.")
    st.stop()

prices = pd.concat(series, axis=1).ffill()
id_to_sym = dict(zip(keep_ids, keep_syms))
prices.columns = [id_to_sym.get(c, c).upper() for c in prices.columns]
prices = prices.apply(pd.to_numeric, errors="coerce")
st.write(f"Prices loaded for {prices.shape[1]} assets, {prices.shape[0]} daily rows.")

# ---- Coverage filter on price window ----
win = {"7D":7, "30D":30, "90D":90}[corr_window]
last_prices = prices.tail(win+1)  # +1 to make sure returns have win rows
keep_cols = [c for c in last_prices.columns if last_prices[c].notna().mean() >= (min_coverage/100)]
prices = prices[keep_cols]

if prices.shape[1] < 3:
    st.warning("Too few assets after coverage filter. Lower threshold or widen window.")
    st.stop()

# ---- Returns & Vol ----
rets = np.log(prices).diff()  # simple, robust
# drop columns that are all NaN or zero-variance within the active window
last_rets = rets.tail(win)
var = last_rets.var(skipna=True)
good = var[var > 0].index.tolist()
rets = rets[good]

ann = math.sqrt(365)
rv = rets.rolling(int(vol_roll), min_periods=max(2, int(vol_roll)//2)).std() * ann

# Ensure enough data points in the active window
last_rets = rets.tail(win)
last_rv = rv.tail(win)
min_pts = max(3, win//2)
X = last_rets.dropna(axis=1, thresh=min_pts)
Y = last_rv.dropna(axis=1, thresh=min_pts)

# Re-check zero-variance after drops
X = X.loc[:, X.std(skipna=True) > 0]
Y = Y.loc[:, Y.std(skipna=True) > 0]

# ---- Correlations ----
corr_price = X.corr(min_periods=2)
corr_vol   = Y.corr(min_periods=2)

finite_price = int(np.isfinite(corr_price.values).sum())
finite_vol   = int(np.isfinite(corr_vol.values).sum())
st.write(f"Corr shapes → price: {corr_price.shape} (finite={finite_price}), vol: {corr_vol.shape} (finite={finite_vol})")

# ---- Preview numbers ----
st.write("Price corr (top-left 5×5)"); st.dataframe(corr_price.iloc[:5,:5].round(3), use_container_width=True)
st.write("Vol corr (top-left 5×5)");   st.dataframe(corr_vol.iloc[:5,:5].round(3),   use_container_width=True)

def render_heatmap(M: pd.DataFrame, title: str, key: str):
    if M.empty or not np.isfinite(M.values).any():
        st.warning(f"No finite values for {title}. Try 30D/90D or lower coverage.")
        return
    fig = go.Figure(data=go.Heatmap(
        z=M.values, x=M.columns.tolist(), y=M.index.tolist(),
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

st.caption("If you still see warnings on 7D, switch Corr lookback to 30D — short windows can go singular. This version avoids all the earlier pitfalls.")

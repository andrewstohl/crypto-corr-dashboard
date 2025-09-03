import time, math, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go

# ---------- Config ----------
API_BASE = "https://rest.coincap.io/v3"    # CoinCap v3
STABLE_SYMS = {"USDT","USDC","DAI","FDUSD","TUSD","USDe","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX"}
st.set_page_config(page_title="Crypto Correlations (CoinCap v3)", layout="wide")

# ---------- Secrets / Headers ----------
API_KEY = st.secrets.get("COINCAP_API_KEY", "")
if not API_KEY:
    st.error("Missing COINCAP_API_KEY in Streamlit Secrets.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept-Encoding": "gzip",
    "User-Agent": "crypto-corr/2.0"
}

# ---------- HTTP helper ----------
def http_get(url, params=None, timeout=60):
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
    if r.status_code >= 400:
        st.error(f"HTTP {r.status_code} on {url}")
        st.code(r.text[:400])
    r.raise_for_status()
    return r

# ---------- Data fetch ----------
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_universe(limit=200):
    r = http_get(f"{API_BASE}/assets", params={"limit": limit, "apiKey": API_KEY})
    rows = r.json().get("data", []) or []
    # rank sort, drop stables, dedupe symbols (keep best rank)
    rows = [x for x in rows if x.get("rank")]
    rows.sort(key=lambda x: int(x["rank"]))
    seen = set()
    ids, symbols = [], []
    for row in rows:
        sym = str(row.get("symbol","")).upper()
        name = str(row.get("name","")).lower()
        if sym in STABLE_SYMS or "stable" in name: 
            continue
        if sym in seen: 
            continue
        seen.add(sym)
        ids.append(row["id"])
        symbols.append(sym)
        if len(ids) == 100:
            break
    return ids, symbols, rows

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily(asset_id: str, days: int = 365) -> pd.Series | None:
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp() * 1000)
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
    s = pd.Series(s.values, index=df["ts"]).asfreq("D")  # align to calendar days
    s = s.ffill()                                        # fill gaps within each series
    s = s.replace([np.inf, -np.inf], np.nan).astype(float)
    s[s <= 0] = np.nan                                   # non-positive prices are invalid
    s.name = asset_id
    return s

# ---------- Utils ----------
def render_heatmap(M: pd.DataFrame, title: str, key: str):
    if M.empty or not np.isfinite(M.values).any():
        st.warning(f"No finite values for {title}. Try widening the window or lowering coverage.")
        return
    fig = go.Figure(data=go.Heatmap(
        z=M.values, x=M.columns.tolist(), y=M.index.tolist(),
        zmin=-1, zmax=1, zmid=0, colorscale="RdBu", colorbar=dict(title="ρ")
    ))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True, key=key)

def top_pairs(C: pd.DataFrame, k=30) -> pd.DataFrame:
    if C.empty: 
        return pd.DataFrame(columns=["A","B","|rho|","rho"])
    cols = list(C.columns); vals = C.values
    out = []
    n = len(cols)
    for i in range(n):
        for j in range(i+1, n):
            v = vals[i, j]
            if np.isfinite(v):
                out.append((cols[i], cols[j], abs(float(v)), float(v)))
    if not out: 
        return pd.DataFrame(columns=["A","B","|rho|","rho"])
    df = pd.DataFrame(out, columns=["A","B","|rho|","rho"]).sort_values("|rho|", ascending=False)
    return df.head(k)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Controls")
    hist_days   = st.select_slider("History window", options=[180, 270, 365], value=365)
    corr_win_s  = st.selectbox("Correlation lookback", ["7D", "30D", "90D"], index=1)
    vol_roll_s  = st.selectbox("Volatility roll window", ["7", "30", "90"], index=1)
    min_cov_pct = st.slider("Min coverage (%)", 50, 100, 70, step=5)
    topn        = st.slider("Top pairs to show", 10, 100, 30, step=5)
    if st.button("Refresh data (clear cache)"):
        fetch_universe.clear(); fetch_hist_daily.clear()
        st.success("Cache cleared. Data will refetch.")

st.caption("Data: CoinCap v3 (API key). Returns = log(price).diff(); Volatility = rolling σ × √365; Pearson corr on the last window. The app auto-picks the best common end-date to avoid sparse last rows.")

# ---------- Universe ----------
ids, symbols, raw = fetch_universe()
st.write(f"Universe selected: {len(ids)} assets (unique symbols, stables excluded).")

# ---------- Prices ----------
progress = st.progress(0)
series = {}
for i, cid in enumerate(ids):
    s = fetch_hist_daily(cid, days=hist_days)
    if s is not None:
        series[cid] = s
    progress.progress((i+1)/len(ids))
    time.sleep(0.05)  # polite to API; cached after first run

if not series:
    st.error("No price series fetched.")
    st.stop()

prices = pd.concat(series, axis=1).sort_index()
# map IDs -> symbols for column labels
id2sym = dict(zip(ids, symbols))
prices.columns = [id2sym.get(c, c).upper() for c in prices.columns]
prices = prices.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

st.write(f"Prices loaded for {prices.shape[1]} assets, {prices.shape[0]} daily rows.")

# ---------- Pick the best common end-date (key fix) ----------
row_frac = prices.notna().mean(axis=1)
# choose the latest date with coverage >= threshold; if none, choose the date with MAX coverage
threshold = min_cov_pct / 100.0
eligible = row_frac[row_frac >= threshold]
if len(eligible):
    end_date = eligible.index.max()
else:
    end_date = row_frac.idxmax()

# You get the idea: we end our window on the best-covered date, not necessarily the last row.
st.write(f"Using end date: {end_date.date()} with coverage {row_frac.loc[end_date]:.0%}")

# ---------- Slice windows for returns/vol ----------
corr_win = {"7D":7, "30D":30, "90D":90}[corr_win_s]
vol_roll = int(vol_roll_s)

# build returns over full history, then take the last window ending at end_date
rets_full = np.log(prices).diff()
# cap extreme returns a bit to avoid numeric issues (no heavy winsorization)
q = 0.999
clip_lo = rets_full.quantile(1-q); clip_hi = rets_full.quantile(q)
rets_full = rets_full.clip(lower=clip_lo, upper=clip_hi)

# realized vol over full history
rv_full = rets_full.rolling(vol_roll, min_periods=max(2, vol_roll//2)).std() * math.sqrt(365)

# align to the chosen end_date
rets = rets_full.loc[:end_date].tail(corr_win)
rv   = rv_full.loc[:end_date].tail(corr_win)

# column coverage inside the ACTIVE window
min_pts = max(3, corr_win//2)
good_cols_r = rets.count() >= min_pts
good_cols_v = rv.count()   >= min_pts

rets = rets.loc[:, good_cols_r]
rv   = rv.loc[:,   good_cols_v]

# drop zero-variance columns in-window (they blow up corr)
rets = rets.loc[:, rets.std(skipna=True) > 0]
rv   = rv.loc[:,   rv.std(skipna=True)   > 0]

if rets.shape[1] < 3:
    st.warning("Too few assets after in-window checks for returns. Lower Min coverage or increase lookback.")
    st.stop()
if rv.shape[1] < 3:
    st.warning("Too few assets after in-window checks for volatility. Lower Min coverage or increase lookback.")
    st.stop()

# ---------- Correlations ----------
corr_price = rets.corr(min_periods=2)
corr_vol   = rv.corr(min_periods=2)

finite_price = int(np.isfinite(corr_price.values).sum())
finite_vol   = int(np.isfinite(corr_vol.values).sum())
st.write(f"Corr shapes → price: {corr_price.shape} (finite={finite_price}), vol: {corr_vol.shape} (finite={finite_vol})")

# ---------- Preview ----------
st.write("Price corr (top-left 5×5)")
st.dataframe(corr_price.iloc[:5, :5].round(3), use_container_width=True)
st.write("Vol corr (top-left 5×5)")
st.dataframe(corr_vol.iloc[:5, :5].round(3),   use_container_width=True)

# ---------- Heatmaps + Top pairs ----------
col1, col2 = st.columns(2)
with col1:
    render_heatmap(corr_price, f"Price correlation — last {corr_win_s}", "price_corr_chart")
    st.subheader("Top price-corr pairs")
    tpp = top_pairs(corr_price, k=topn)
    st.dataframe(tpp, use_container_width=True)
    st.download_button("Download corr (price) CSV", corr_price.to_csv().encode(), file_name="corr_price.csv", mime="text/csv")
with col2:
    render_heatmap(corr_vol, f"Volatility correlation — last {corr_win_s} (σ roll {vol_roll}d)", "vol_corr_chart")
    st.subheader("Top vol-corr pairs")
    tpv = top_pairs(corr_vol, k=topn)
    st.dataframe(tpv, use_container_width=True)
    st.download_button("Download corr (vol) CSV", corr_vol.to_csv().encode(), file_name="corr_vol.csv", mime="text/csv")

st.caption("If you still see sparse data, raise history to 365, set Min coverage ~60–70%, and use 30D/90D windows. The end date is auto-chosen to maximize coverage.")

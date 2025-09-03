import time, math, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go

API_BASE = "https://rest.coincap.io/v3"
STABLE_SYMS = {"USDT","USDC","DAI","FDUSD","TUSD","USDe","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX"}

st.set_page_config(page_title="Crypto Correlations (CoinCap v3)", layout="wide")

# --- Secrets / headers ---
API_KEY = st.secrets.get("COINCAP_API_KEY", "")
if not API_KEY:
    st.error("Missing COINCAP_API_KEY in Streamlit Secrets.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept-Encoding": "gzip",
    "User-Agent": "crypto-corr/3.0"
}

def http_get(url, params=None, timeout=60):
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
    if r.status_code >= 400:
        st.error(f"HTTP {r.status_code} on {url}")
        st.code(r.text[:400])
    r.raise_for_status()
    return r

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_universe(limit=250):
    r = http_get(f"{API_BASE}/assets", params={"limit": limit, "apiKey": API_KEY})
    rows = r.json().get("data", []) or []
    rows = [x for x in rows if x.get("rank")]
    rows.sort(key=lambda x: int(x["rank"]))
    seen = set()
    ids, syms = [], []
    for row in rows:
        sym = str(row.get("symbol","")).upper()
        name = str(row.get("name","")).lower()
        if sym in STABLE_SYMS or "stable" in name:
            continue
        if sym in seen:
            continue
        seen.add(sym)
        ids.append(row["id"]); syms.append(sym)
        if len(ids) == 100: break
    return ids, syms, rows

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily(asset_id: str, days: int = 365) -> pd.Series | None:
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp() * 1000)
    r = http_get(f"{API_BASE}/assets/{asset_id}/history",
                 params={"interval": "d1", "start": start_ms, "end": end_ms, "apiKey": API_KEY})
    data = r.json().get("data", [])
    if not data:
        return None
    df = pd.DataFrame(data)
    ts = pd.to_datetime(df["time"], unit="ms", utc=True)
    px = pd.to_numeric(df["priceUsd"], errors="coerce").astype(float)
    s = pd.Series(px.values, index=ts).asfreq("D").ffill()  # daily calendar, forward-fill gaps
    s[s <= 0] = np.nan                                     # invalid prices
    s.name = asset_id
    return s

def pairwise_corr(df: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    """Robust pairwise Pearson correlation with per-pair NaN handling."""
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
                # fast, stable corr
                ca = a - a.mean()
                cb = b - b.mean()
                denom = (np.sqrt((ca*ca).sum()) * np.sqrt((cb*cb).sum()))
                if denom > 0:
                    rho = float((ca*cb).sum() / denom)
                    out[i, j] = out[j, i] = rho
    return pd.DataFrame(out, index=cols, columns=cols)

def render_heatmap(M: pd.DataFrame, title: str, key: str):
    if M.empty or not np.isfinite(M.values).any():
        st.warning(f"No finite values for {title}. Try 30D/90D.")
        return
    fig = go.Figure(data=go.Heatmap(
        z=M.values, x=M.columns.tolist(), y=M.index.tolist(),
        zmin=-1, zmax=1, zmid=0, colorscale="RdBu", colorbar=dict(title="ρ")
    ))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0))
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
    if not pairs:
        return pd.DataFrame(columns=["A","B","|rho|","rho"])
    return pd.DataFrame(pairs, columns=["A","B","|rho|","rho"]).sort_values("|rho|", ascending=False).head(k)

# --- Sidebar ---
with st.sidebar:
    st.title("Controls")
    hist_days   = st.select_slider("History window", options=[180, 270, 365], value=365)
    corr_win_s  = st.selectbox("Correlation lookback", ["7D", "30D", "90D"], index=1)
    vol_roll_s  = st.selectbox("Volatility roll window", ["7", "30", "90"], index=1)
    # This is now used ONLY to choose the end-date, not to drop columns
    min_cov_pct = st.slider("Min daily coverage to anchor end-date (%)", 40, 100, 60, step=5)
    topn        = st.slider("Top pairs to show", 10, 100, 30, step=5)
    if st.button("Refresh data (clear cache)"):
        fetch_universe.clear(); fetch_hist_daily.clear()
        st.success("Cache cleared. Data will refetch.")

st.caption("Data: CoinCap v3 (API key). Returns = log(price).diff(); Vol = rolling σ × √365. We anchor the window to the best-covered date and compute pairwise correlations with per-pair NaN handling.")

# --- Universe & prices (cached) ---
ids, syms, raw = fetch_universe()
st.write(f"Universe selected: {len(ids)} assets (unique symbols, stables excluded).")

progress = st.progress(0)
series = {}
for i, cid in enumerate(ids):
    s = fetch_hist_daily(cid, days=hist_days)
    if s is not None: series[cid] = s
    progress.progress((i+1)/len(ids))
    time.sleep(0.04)  # polite; still fast; cached after first run

if not series:
    st.error("No price series fetched.")
    st.stop()

prices = pd.concat(series, axis=1).sort_index()
id2sym = dict(zip(ids, syms))
prices.columns = [id2sym.get(c, c).upper() for c in prices.columns]
prices = prices.replace([np.inf, -np.inf], np.nan)
prices[prices <= 0] = np.nan

st.write(f"Prices loaded for {prices.shape[1]} assets, {prices.shape[0]} daily rows.")

# --- Choose end-date with max coverage above threshold (or global max if none) ---
row_cov = prices.notna().mean(axis=1)
thr = min_cov_pct/100.0
eligible = row_cov[row_cov >= thr]
end_date = eligible.index.max() if len(eligible) else row_cov.idxmax()
st.write(f"Anchoring to end date: {end_date.date()} (coverage {row_cov.loc[end_date]:.0%})")

# --- Build windows ---
corr_win = {"7D":7, "30D":30, "90D":90}[corr_win_s]
roll_w   = int(vol_roll_s)

# returns over full history, then window-ending at end_date
rets_full = np.log(prices).diff()
# trim insane outliers a bit to stabilize small windows (very loose)
q = 0.999
rets_full = rets_full.clip(lower=rets_full.quantile(1-q), upper=rets_full.quantile(q))

rets_win = rets_full.loc[:end_date].tail(corr_win)
# drop columns that are entirely NaN or zero variance in-window
mask_keep = rets_win.count() >= 2
rets_win = rets_win.loc[:, mask_keep]
std_win = rets_win.std(skipna=True)
rets_win = rets_win.loc[:, std_win > 0]

# realized vol series, then window-ending at end_date
rv_full = rets_full.rolling(roll_w, min_periods=max(2, roll_w//2)).std() * math.sqrt(365)
rv_win  = rv_full.loc[:end_date].tail(corr_win)
mask_keep_v = rv_win.count() >= 2
rv_win  = rv_win.loc[:, mask_keep_v]
std_win_v = rv_win.std(skipna=True)
rv_win  = rv_win.loc[:, std_win_v > 0]

# --- Pairwise correlations with small, sensible min overlaps
pair_min = {7:3, 30:5, 90:10}[corr_win]
corr_price = pairwise_corr(rets_win, min_overlap=pair_min)
corr_vol   = pairwise_corr(rv_win,   min_overlap=pair_min)

finite_price = int(np.isfinite(corr_price.values).sum())
finite_vol   = int(np.isfinite(corr_vol.values).sum())
st.write(f"Corr shapes → price: {corr_price.shape} (finite={finite_price}), vol: {corr_vol.shape} (finite={finite_vol})")

# --- Previews ---
st.write("Price corr (top-left 5×5)"); st.dataframe(corr_price.iloc[:5,:5].round(3), use_container_width=True)
st.write("Vol corr (top-left 5×5)");   st.dataframe(corr_vol.iloc[:5,:5].round(3),   use_container_width=True)

# --- Heatmaps + Top pairs ---
col1, col2 = st.columns(2)
with col1:
    render_heatmap(corr_price, f"Price correlation — last {corr_win_s}", "price_corr_chart")
    st.subheader("Top price-corr pairs")
    tpp = top_pairs(corr_price, k=topn)
    st.dataframe(tpp, use_container_width=True)
    st.download_button("Download corr (price) CSV", corr_price.to_csv().encode(), file_name="corr_price.csv", mime="text/csv")
with col2:
    render_heatmap(corr_vol, f"Volatility correlation — last {corr_win_s} (σ roll {roll_w}d)", "vol_corr_chart")
    st.subheader("Top vol-corr pairs")
    tpv = top_pairs(corr_vol, k=topn)
    st.dataframe(tpv, use_container_width=True)
    st.download_button("Download corr (vol) CSV", corr_vol.to_csv().encode(), file_name="corr_vol.csv", mime="text/csv")

st.caption("If a heatmap still looks sparse on 7D, switch to 30D or 90D. This build computes pairwise overlaps directly, so you’ll always get numbers when any overlap exists.")

# app.py
import time, math, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go

# ================== CONFIG ==================
# CoinCap v3 endpoints (rotate on failure)
COINCAP_BASES = [
    "https://pro-api.coincap.io/v3",
    "https://api.coincap.io/v3",
    "https://rest.coincap.io/v3",
]

# Obvious stablecoins to exclude
STABLE = {
    "USDT","USDC","DAI","FDUSD","TUSD","USDe","USDL","USDP","PYUSD","GUSD","FRAX",
    "LUSD","USDD","USDX","BUSD"
}

st.set_page_config(page_title="Crypto Correlations — CoinCap v3", layout="wide")

# ================== AUTH ====================
API_KEY = st.secrets.get("COINCAP_API_KEY", "")
if not API_KEY:
    st.error("Missing COINCAP_API_KEY in Streamlit Secrets.")
    st.stop()

def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "User-Agent": "crypto-corr/solid-2.0"
    }

# Remember last working base during session
if "COINCAP_BASE" not in st.session_state:
    st.session_state.COINCAP_BASE = COINCAP_BASES[0]

# ================== HTTP ====================
def http_get(path, params=None, timeout=30, retries=2):
    """
    GET against CoinCap with retry and base failover. Path must start with '/'.
    """
    params = params or {}
    # Also pass apiKey as query to satisfy some gateways
    params = dict(params)
    params.setdefault("apiKey", API_KEY)

    bases = [st.session_state.COINCAP_BASE] + [b for b in COINCAP_BASES if b != st.session_state.COINCAP_BASE]
    last_err = None

    for base in bases:
        url = base + path
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    last_err = (r.status_code, r.text[:300])
                    break  # try next base
                st.session_state.COINCAP_BASE = base
                return r
            except Exception as e:
                last_err = (type(e).__name__, str(e)[:200])
                time.sleep(0.5)
                continue

    if last_err:
        st.error(f"CoinCap request failed for {path}. Last error: {last_err[0]}")
        st.code(str(last_err[1]))
    raise requests.HTTPError(f"All CoinCap bases failed for {path}: {last_err}")

# =============== DATA FETCH =================
@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_universe(limit=250):
    """
    Top-ranked assets, drop stables, dedupe by symbol, cap at 100.
    Returns: (ids, symbols)
    """
    r = http_get("/assets", params={"limit": limit})
    data = r.json()
    rows = data.get("data", data.get("assets", [])) if isinstance(data, dict) else (data or [])
    rows = [x for x in rows if isinstance(x, dict) and x.get("rank")]
    for x in rows:
        try:
            x["_rank"] = int(x["rank"])
        except:
            x["_rank"] = 10**9
    rows.sort(key=lambda x: x["_rank"])

    ids, syms, seen = [], [], set()
    for row in rows:
        aid = row.get("id", "")
        sym = str(row.get("symbol", "")).upper()
        name = str(row.get("name", "")).lower()
        if not aid or not sym:
            continue
        if sym in STABLE or "stable" in name or sym.endswith("USD"):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        ids.append(aid); syms.append(sym)
        if len(ids) >= 100:
            break
    return ids, syms

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily(asset_id: str, days: int = 365) -> pd.Series | None:
    """
    Daily history from CoinCap v3. No forward fill here. Returns pd.Series(price, index=UTC dates).
    """
    end_ms   = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp() * 1000)

    r = http_get(
        f"/assets/{asset_id}/history",
        params={"interval": "d1", "start": start_ms, "end": end_ms},
        retries=1
    )
    data = r.json()
    arr = data.get("data", data.get("history", data.get("prices", []))) if isinstance(data, dict) else (data or [])
    if not arr:
        return None

    df = pd.DataFrame(arr)
    time_col = None
    price_col = None
    for col in df.columns:
        cl = col.lower()
        if cl in ("time", "timestamp", "date", "datetime"):
            time_col = col
        if cl in ("priceusd", "price", "close", "price_usd"):
            price_col = col
    if time_col is None or price_col is None:
        return None

    tvals = pd.to_numeric(df[time_col], errors="coerce")
    if tvals.max() > 1e10:
        ts = pd.to_datetime(tvals, unit="ms", utc=True)
    else:
        ts = pd.to_datetime(tvals, unit="s", utc=True)

    px = pd.to_numeric(df[price_col], errors="coerce").astype(float)
    s = pd.Series(px.values, index=ts).sort_index()
    s = s.replace([np.inf, -np.inf], np.nan)
    s[s <= 0] = np.nan
    # Coerce to calendar daily grid ending at its own last point (no ffill here)
    # We'll align later on a common grid.
    return s

# =============== HELPERS ====================
def uniqueify(labels):
    out, used = [], {}
    for x in labels:
        if x not in used:
            used[x] = 1; out.append(x)
        else:
            used[x] += 1; out.append(f"{x}#{used[x]}")
    return out

def choose_best_end_date(prices: pd.DataFrame, lookback_days: int, search_days: int = 180, min_points_per_col: int = 10):
    """
    Scan the last ~search_days and pick the end-date that maximizes how many columns
    have at least min_points_per_col valid points inside the rolling window (lookback_days).
    """
    if prices.empty:
        return None
    counts = prices.notna().rolling(lookback_days, min_periods=1).sum()
    recent_idx = counts.index[counts.index >= counts.index.max() - pd.Timedelta(days=search_days)]
    sub = counts.loc[recent_idx] if len(recent_idx) else counts
    score = (sub >= min_points_per_col).sum(axis=1)
    if score.empty:
        return None
    return score.idxmax()

def prune_in_window(M: pd.DataFrame, min_non_null=3):
    if M is None or M.empty:
        return M
    M = M.loc[:, M.count() >= min_non_null]
    if M.empty:
        return M
    M = M.loc[:, M.std(skipna=True) > 0]
    return M

def pairwise_corr(df: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    cols = list(df.columns); n = len(cols)
    if n == 0: return pd.DataFrame()
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
                denom = np.sqrt((ca*ca).sum()) * np.sqrt((cb*cb).sum())
                if denom > 0:
                    out[i, j] = out[j, i] = float((ca*cb).sum() / denom)
    return pd.DataFrame(out, index=cols, columns=cols)

def render_heatmap(M: pd.DataFrame, title: str, key: str):
    if M.empty or not np.isfinite(M.values).any():
        st.warning(f"No finite values for {title}. Widen lookback or lower min overlap.")
        return
    # If very large, show top-20 by avg |corr| to keep it readable
    if M.shape[0] > 24:
        avg_abs = np.abs(M).mean(axis=1).sort_values(ascending=False)
        keep = avg_abs.head(24).index
        M = M.loc[keep, keep]
    fig = go.Figure(data=go.Heatmap(
        z=M.values, x=M.columns.tolist(), y=M.index.tolist(),
        zmin=-1, zmax=1, zmid=0, colorscale="RdBu", colorbar=dict(title="ρ")
    ))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0))
    # Unique key prevents DuplicateElementId
    st.plotly_chart(fig, key=key)

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

# =============== SIDEBAR ====================
with st.sidebar:
    st.title("Controls")
    universe_size = st.select_slider("Universe size (by rank)", options=[50, 75, 100], value=100)
    hist_days     = st.select_slider("History window (days)", options=[180, 270, 365], value=365)
    lookback_s    = st.selectbox("Correlation lookback", ["7D","30D","90D"], index=1)
    vol_roll_s    = st.selectbox("Volatility roll window (days)", ["7","14","30","60"], index=2)
    min_overlap   = st.slider("Min overlap (days) per pair", 3, 60, 10, step=1)
    topk          = st.slider("Top pairs to show", 10, 100, 30, step=5)
    demo_mode     = st.toggle("Demo mode (no API calls)", value=False, help="Loads a tiny baked sample so you can test UI without spending credits.")
    if st.button("Clear cache & refetch"):
        fetch_universe.clear(); fetch_hist_daily.clear()
        st.success("Cache cleared. Data will refetch on next run.")

st.caption("Source: CoinCap v3 (API key). We auto-pick the best end date for your lookback, avoid forward-filling before returns, and compute pairwise correlations with per-pair overlap.")

# =============== DEMO (optional) =============
if demo_mode:
    # Small embedded demo: BTC/ETH/SOL/BNB synthetic-ish series to verify UI without API calls
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=120, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    def synth(mu=0.0007, sigma=0.03):
        r = rng.normal(mu, sigma, size=len(idx))
        return pd.Series(np.exp(r).cumprod(), index=idx)
    prices = pd.DataFrame({
        "BTC": 30000 * synth(0.0006, 0.025),
        "ETH": 2000  * synth(0.0007, 0.03),
        "SOL": 40    * synth(0.0010, 0.05),
        "BNB": 250   * synth(0.0005, 0.02),
    })
    st.info("Demo mode active — no API calls made.")
else:
    # =============== UNIVERSE =================
    ids, syms = fetch_universe()
    if len(ids) > universe_size:
        ids, syms = ids[:universe_size], syms[:universe_size]
    st.write(f"Universe selected: {len(ids)} assets (stables excluded; unique symbols).")

    # =============== PRICES ==================
    progress = st.progress(0)
    series = {}
    for i, aid in enumerate(ids):
        s = fetch_hist_daily(aid, days=hist_days)
        if s is not None and len(s) >= 3:
            series[aid] = s
        progress.progress((i+1)/len(ids))
        time.sleep(0.02)  # polite; cached after first run
    progress.empty()

    if not series:
        st.error("No price series fetched from CoinCap.")
        st.stop()

    # Align all series on a bounded daily grid (last `hist_days` ending at global max end)
    max_end = max(s.index.max() for s in series.values())
    all_dates = pd.date_range(end=max_end, periods=hist_days, freq="D", tz="UTC")

    aligned = {}
    for aid, s in series.items():
        aligned[aid] = s.reindex(all_dates)  # NO forward-fill

    prices = pd.DataFrame(aligned).sort_index()
    # Map columns to unique symbols (avoid collapsing same symbol)
    id2sym = dict(zip(ids, syms))
    col_labels = [id2sym.get(c, c).upper() for c in prices.columns]
    prices.columns = uniqueify(col_labels)
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices[prices <= 0] = np.nan

st.write(f"Prices matrix: {prices.shape[1]} assets × {prices.shape[0]} days.")

# =============== WINDOW PICK =================
lookback = {"7D":7, "30D":30, "90D":90}[lookback_s]
min_pts_per_col = max(3, int(math.ceil(lookback/3)))

end_date = choose_best_end_date(prices, lookback_days=lookback, search_days=180, min_points_per_col=min_pts_per_col)
if end_date is None:
    st.error("No usable end date found. Increase history or lower min overlap.")
    st.stop()
start_date = end_date - pd.Timedelta(days=lookback-1)
st.write(f"Window: {start_date.date()} → {end_date.date()} (min points/col in-window = {min_pts_per_col})")

# =============== RETURNS & VOL ===============
rets_full = np.log(prices).diff()
# gentle outlier clipping per column to stabilize small windows
q_low, q_high = 0.001, 0.999
rets_full = rets_full.apply(lambda s: s.clip(lower=s.quantile(q_low), upper=s.quantile(q_high)))

rv_full = rets_full.rolling(int(st.session_state.get("VOL_ROLL", 0) or 1), min_periods=1).std()  # temporary init

roll_w = int(vol_roll_s)
rv_full = rets_full.rolling(roll_w, min_periods=max(2, roll_w//2)).std() * math.sqrt(365)

rets_win = rets_full.loc[start_date:end_date]
rv_win   = rv_full.loc[start_date:end_date]

# Drop dead columns in-window only
rets_win = prune_in_window(rets_win, min_non_null=max(3, lookback//6))
rv_win   = prune_in_window(rv_win,   min_non_null=max(3, lookback//6))

# =============== CORRELATIONS ================
corr_price = pairwise_corr(rets_win, min_overlap=min_overlap)
corr_vol   = pairwise_corr(rv_win,   min_overlap=min_overlap)
st.write(f"Corr shapes → price: {corr_price.shape}, vol: {corr_vol.shape}")

# Quick previews
st.write("Price corr (top-left 5×5)")
st.dataframe(corr_price.iloc[:5, :5].round(3), width="stretch")
st.write("Vol corr (top-left 5×5)")
st.dataframe(corr_vol.iloc[:5, :5].round(3), width="stretch")

# =============== HEATMAPS & TABLES ===========
col1, col2 = st.columns(2)
with col1:
    render_heatmap(corr_price, f"Price correlation — last {lookback_s}", key="heat_price")
    st.subheader("Top price-corr pairs")
    tp = top_pairs(corr_price, k=topk)
    if not tp.empty:
        st.dataframe(tp, width="stretch")
        st.download_button("Download price-corr matrix CSV", corr_price.to_csv().encode(),
                           file_name="corr_price.csv", mime="text/csv", key="dl_price")
    else:
        st.info("No pairs met the overlap/variance thresholds.")
with col2:
    render_heatmap(corr_vol, f"Volatility correlation — last {lookback_s} (σ roll {roll_w}d)", key="heat_vol")
    st.subheader("Top vol-corr pairs")
    tv = top_pairs(corr_vol, k=topk)
    if not tv.empty:
        st.dataframe(tv, width="stretch")
        st.download_button("Download vol-corr matrix CSV", corr_vol.to_csv().encode(),
                           file_name="corr_vol.csv", mime="text/csv", key="dl_vol")
    else:
        st.info("No pairs met the overlap/variance thresholds.")

# =============== FOOTER ======================
st.caption(
    "Notes: No forward-fill is applied before returns to avoid zero-variance columns. "
    "We pick the end date that maximizes in-window coverage over the last ~6 months. "
    "Pairwise correlations require a minimum overlap you control in the sidebar."
)

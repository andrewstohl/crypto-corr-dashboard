import time, math, requests, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# CoinCap Pro API v3 - Try different endpoints if one fails
API_ENDPOINTS = [
    "https://pro-api.coincap.io/v3",
    "https://api.coincap.io/v3",
    "https://rest.coincap.io/v3"
]

STABLE_SYMS = {"USDT","USDC","DAI","FDUSD","TUSD","USDe","USDL","USDP","PYUSD","GUSD","FRAX","LUSD","USDD","USDX","BUSD"}

st.set_page_config(page_title="Crypto Correlations (CoinCap v3)", layout="wide")

# --- API Configuration ---
API_KEY = st.secrets.get("COINCAP_API_KEY", "")
if not API_KEY:
    st.error("❌ Missing COINCAP_API_KEY in Streamlit Secrets. Please add your CoinCap API key.")
    st.info("Add to `.streamlit/secrets.toml`:\n```\nCOINCAP_API_KEY = \"your-key-here\"\n```")
    st.stop()

# Global variable to track working endpoint
if 'api_base' not in st.session_state:
    st.session_state.api_base = API_ENDPOINTS[0]

def get_headers():
    """Get request headers with proper authorization"""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept-Encoding": "gzip",
        "Accept": "application/json",
        "User-Agent": "crypto-corr/3.0"
    }

def http_get(url, params=None, timeout=30, retry_count=3):
    """HTTP GET with retry logic and endpoint fallback"""
    last_error = None
    
    for attempt in range(retry_count):
        try:
            r = requests.get(url, params=params or {}, headers=get_headers(), timeout=timeout)
            
            if r.status_code == 200:
                return r
            elif r.status_code == 401:
                st.error("❌ Authentication failed. Please check your API key.")
                st.stop()
            elif r.status_code == 429:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
            elif r.status_code >= 400:
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            time.sleep(1)
    
    # If all retries failed, try next endpoint
    for endpoint in API_ENDPOINTS:
        if endpoint != st.session_state.api_base:
            try:
                test_url = url.replace(st.session_state.api_base, endpoint)
                r = requests.get(test_url, params=params or {}, headers=get_headers(), timeout=timeout)
                if r.status_code == 200:
                    st.session_state.api_base = endpoint
                    st.success(f"✅ Switched to working endpoint: {endpoint}")
                    return r
            except:
                continue
    
    st.error(f"Failed after {retry_count} attempts: {last_error}")
    return None

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_universe(limit=250):
    """Fetch top crypto assets excluding stablecoins"""
    url = f"{st.session_state.api_base}/assets"
    
    try:
        r = http_get(url, params={"limit": limit})
        if not r:
            return [], [], []
            
        data = r.json()
        
        # Handle different possible response structures
        if isinstance(data, dict):
            rows = data.get("data", data.get("assets", []))
        elif isinstance(data, list):
            rows = data
        else:
            st.error(f"Unexpected API response structure: {type(data)}")
            return [], [], []
        
        if not rows:
            st.error("No assets returned from API")
            return [], [], []
        
        # Validate and sort by rank
        valid_rows = []
        for row in rows:
            if isinstance(row, dict) and row.get("rank"):
                try:
                    row["rank_int"] = int(row["rank"])
                    valid_rows.append(row)
                except:
                    continue
        
        valid_rows.sort(key=lambda x: x["rank_int"])
        
        seen = set()
        ids, syms = [], []
        
        for row in valid_rows:
            asset_id = row.get("id", "")
            sym = str(row.get("symbol", "")).upper()
            name = str(row.get("name", "")).lower()
            
            # Skip if no ID or symbol
            if not asset_id or not sym:
                continue
                
            # Skip stablecoins
            if sym in STABLE_SYMS or "stable" in name or "usd" in sym.lower():
                continue
            
            # Skip duplicates
            if sym in seen:
                continue
                
            seen.add(sym)
            ids.append(asset_id)
            syms.append(sym)
            
            if len(ids) >= 100:
                break
        
        return ids, syms, valid_rows
        
    except Exception as e:
        st.error(f"Failed to fetch universe: {e}")
        return [], [], []

@st.cache_data(show_spinner=False, ttl=60*60*12)
def fetch_hist_daily(asset_id: str, days: int = 365) -> pd.Series | None:
    """Fetch historical daily prices with proper error handling"""
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Format timestamps (try both milliseconds and seconds)
        end_ms = int(end_time.timestamp() * 1000)
        start_ms = int(start_time.timestamp() * 1000)
        
        url = f"{st.session_state.api_base}/assets/{asset_id}/history"
        
        # Try different parameter combinations
        param_sets = [
            {"interval": "d1", "start": start_ms, "end": end_ms},
            {"interval": "1d", "start": start_ms, "end": end_ms},
            {"interval": "day", "start": start_ms, "end": end_ms},
            {"interval": "d1", "start": int(start_time.timestamp()), "end": int(end_time.timestamp())},
        ]
        
        for params in param_sets:
            r = http_get(url, params=params, retry_count=2)
            
            if r:
                data = r.json()
                
                # Handle different response structures
                if isinstance(data, dict):
                    price_data = data.get("data", data.get("history", data.get("prices", [])))
                elif isinstance(data, list):
                    price_data = data
                else:
                    continue
                
                if price_data and len(price_data) > 0:
                    df = pd.DataFrame(price_data)
                    
                    # Find the time and price columns (might have different names)
                    time_col = None
                    price_col = None
                    
                    for col in df.columns:
                        if col.lower() in ["time", "timestamp", "date", "datetime"]:
                            time_col = col
                        if col.lower() in ["priceusd", "price", "close", "price_usd"]:
                            price_col = col
                    
                    if time_col and price_col:
                        # Parse timestamps (handle both ms and seconds)
                        time_vals = pd.to_numeric(df[time_col], errors="coerce")
                        
                        # Detect if timestamps are in seconds or milliseconds
                        if time_vals.max() > 1e10:  # Likely milliseconds
                            ts = pd.to_datetime(time_vals, unit="ms", utc=True)
                        else:  # Likely seconds
                            ts = pd.to_datetime(time_vals, unit="s", utc=True)
                        
                        # Parse prices
                        px = pd.to_numeric(df[price_col], errors="coerce").astype(float)
                        
                        # Create series
                        s = pd.Series(px.values, index=ts)
                        
                        # Remove duplicates and sort
                        s = s[~s.index.duplicated(keep='first')].sort_index()
                        
                        # Resample to daily frequency with forward fill
                        s = s.resample("D").last().ffill(limit=7)  # Fill up to 7 days of gaps
                        
                        # Remove invalid prices
                        s = s[s > 0]
                        
                        # Need at least 30 data points
                        if len(s) >= 30:
                            s.name = asset_id
                            return s
        
        return None
        
    except Exception as e:
        # Silently fail for individual assets
        return None

def pairwise_corr(df: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    """Calculate pairwise correlations with robust NaN handling"""
    if df.empty:
        return pd.DataFrame()
    
    # Standardize the data first (helps with numerical stability)
    df_std = (df - df.mean()) / df.std()
    df_std = df_std.replace([np.inf, -np.inf], np.nan)
    
    cols = list(df_std.columns)
    n = len(cols)
    corr_matrix = np.full((n, n), np.nan, dtype=float)
    
    for i in range(n):
        corr_matrix[i, i] = 1.0
        
        for j in range(i + 1, n):
            xi = df_std.iloc[:, i].values
            xj = df_std.iloc[:, j].values
            
            # Find valid pairs
            valid_mask = np.isfinite(xi) & np.isfinite(xj)
            n_valid = valid_mask.sum()
            
            if n_valid >= min_overlap:
                # Use numpy's corrcoef for numerical stability
                valid_xi = xi[valid_mask]
                valid_xj = xj[valid_mask]
                
                if np.std(valid_xi) > 1e-10 and np.std(valid_xj) > 1e-10:
                    corr = np.corrcoef(valid_xi, valid_xj)[0, 1]
                    
                    if np.isfinite(corr):
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
    
    return pd.DataFrame(corr_matrix, index=cols, columns=cols)

def render_heatmap(M: pd.DataFrame, title: str, key: str):
    """Render correlation heatmap with improved visuals"""
    if M.empty or M.shape[0] < 2:
        st.warning(f"Insufficient data for {title}")
        return
    
    # Filter to show only top correlations for readability
    if M.shape[0] > 20:
        # Get assets with highest average absolute correlation
        avg_corr = np.abs(M).mean(axis=1)
        top_assets = avg_corr.nlargest(20).index
        M = M.loc[top_assets, top_assets]
    
    fig = go.Figure(data=go.Heatmap(
        z=M.values,
        x=M.columns.tolist(),
        y=M.index.tolist(),
        zmin=-1,
        zmax=1,
        zmid=0,
        colorscale="RdBu_r",  # Reversed: red=positive, blue=negative
        colorbar=dict(title="Correlation"),
        text=np.round(M.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=800,
        height=600,
        margin=dict(l=100, r=20, t=70, b=100)
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)

def top_pairs(C: pd.DataFrame, k=30) -> pd.DataFrame:
    """Extract top correlated pairs with better formatting"""
    if C.empty:
        return pd.DataFrame()
    
    # Get upper triangle of correlation matrix
    pairs = []
    for i in range(len(C.index)):
        for j in range(i + 1, len(C.columns)):
            corr_val = C.iloc[i, j]
            if np.isfinite(corr_val) and abs(corr_val) > 0.01:  # Filter out near-zero correlations
                pairs.append({
                    "Asset A": C.index[i],
                    "Asset B": C.columns[j],
                    "Correlation": round(corr_val, 4),
                    "Abs Correlation": round(abs(corr_val), 4)
                })
    
    if not pairs:
        return pd.DataFrame()
    
    df_pairs = pd.DataFrame(pairs)
    return df_pairs.nlargest(k, "Abs Correlation")

# --- Main Application ---
st.title("🔄 Crypto Correlation Analysis Dashboard")
st.caption("Analyze price and volatility correlations for liquidity pool optimization using CoinCap v3 API")

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("📊 Data Settings")
    hist_days = st.select_slider(
        "Historical data (days)",
        options=[90, 180, 270, 365],
        value=365,
        help="Days of historical data to fetch"
    )
    
    corr_win_s = st.selectbox(
        "Correlation window",
        ["7D", "30D", "90D"],
        index=1,
        help="Time window for correlation calculation"
    )
    
    vol_roll_s = st.selectbox(
        "Volatility window",
        ["7", "30", "90"],
        index=1,
        help="Rolling window for volatility"
    )
    
    min_cov_pct = st.slider(
        "Min coverage %",
        40, 100, 60,
        step=5,
        help="Minimum data coverage to use"
    )
    
    topn = st.slider(
        "Top pairs to display",
        10, 50, 20,
        step=5
    )
    
    st.divider()
    
    if st.button("🔄 Clear Cache & Refresh", type="primary"):
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()
    
    st.divider()
    st.caption(f"API Endpoint: {st.session_state.api_base}")
    if API_KEY:
        st.success(f"✅ API Key: ...{API_KEY[-4:]}")

# --- Data Loading ---
st.header("📥 Data Loading Progress")

# Fetch universe
with st.spinner("Fetching crypto universe..."):
    ids, syms, raw_data = fetch_universe()

if not ids:
    st.error("❌ Failed to fetch assets. Please check your API key and connection.")
    st.stop()

st.success(f"✅ Found {len(ids)} eligible non-stable assets")

# Create progress tracking
col1, col2, col3 = st.columns(3)
with col1:
    progress_bar = st.progress(0)
with col2:
    status_text = st.empty()
with col3:
    success_counter = st.empty()

# Fetch historical data
series = {}
failed = []

for idx, (asset_id, symbol) in enumerate(zip(ids, syms)):
    progress = (idx + 1) / len(ids)
    progress_bar.progress(progress)
    status_text.text(f"Loading {symbol}...")
    
    s = fetch_hist_daily(asset_id, days=hist_days)
    
    if s is not None:
        series[asset_id] = s
        success_counter.text(f"✅ {len(series)}/{idx+1}")
    else:
        failed.append(symbol)
    
    # Rate limiting
    if (idx + 1) % 10 == 0:
        time.sleep(0.5)
    else:
        time.sleep(0.05)

# Clear progress indicators
progress_bar.empty()
status_text.empty()
success_counter.empty()

if not series:
    st.error("❌ No historical data could be fetched. The API might be down or your key might be invalid.")
    st.stop()

st.info(f"📊 Successfully loaded: {len(series)}/{len(ids)} assets")
if failed:
    with st.expander(f"Failed assets ({len(failed)})"):
        st.write(", ".join(failed[:20]) + ("..." if len(failed) > 20 else ""))

# --- Create Price DataFrame ---
st.header("📈 Data Processing")

# Align all series to common date range
all_dates = pd.date_range(
    start=min(s.index.min() for s in series.values()),
    end=max(s.index.max() for s in series.values()),
    freq="D"
)

# Create aligned DataFrame
aligned_data = {}
for asset_id, s in series.items():
    # Reindex to common dates
    aligned = s.reindex(all_dates)
    # Forward fill up to 7 days
    aligned = aligned.fillna(method='ffill', limit=7)
    aligned_data[asset_id] = aligned

prices = pd.DataFrame(aligned_data)

# Map to symbols
id2sym = dict(zip(ids, syms))
prices.columns = [id2sym.get(c, c).upper() for c in prices.columns]

# Clean data
prices = prices.replace([np.inf, -np.inf], np.nan)
prices = prices[prices > 0]

st.success(f"✅ Price matrix created: {prices.shape[1]} assets × {prices.shape[0]} days")

# --- Find Best End Date ---
st.subheader("📅 Selecting Analysis Window")

# Calculate coverage
coverage = prices.notna().mean(axis=1)
threshold = min_cov_pct / 100.0

# Find best end date
eligible_dates = coverage[coverage >= threshold]

if len(eligible_dates) > 0:
    end_date = eligible_dates.index[-1]  # Most recent date with good coverage
else:
    end_date = coverage.idxmax()  # Date with best coverage overall

coverage_at_end = coverage.loc[end_date]

col1, col2 = st.columns(2)
with col1:
    st.metric("Selected End Date", end_date.strftime("%Y-%m-%d"))
with col2:
    st.metric("Data Coverage", f"{coverage_at_end:.1%}")

# --- Calculate Returns and Volatility ---
st.header("📊 Correlation Analysis")

# Parse window parameters
corr_window_days = {"7D": 7, "30D": 30, "90D": 90}[corr_win_s]
vol_window_days = int(vol_roll_s)

# Calculate log returns
with st.spinner("Calculating returns..."):
    returns = np.log(prices).diff()
    
    # Remove outliers (more conservative approach)
    for col in returns.columns:
        col_data = returns[col]
        q_low = col_data.quantile(0.001)
        q_high = col_data.quantile(0.999)
        returns[col] = col_data.clip(lower=q_low, upper=q_high)

# Select window for analysis
window_start = end_date - pd.Timedelta(days=corr_window_days)
returns_window = returns.loc[window_start:end_date]

# Filter assets with sufficient data IN THE WINDOW
min_required = max(3, corr_window_days // 3)
valid_counts = returns_window.count()
valid_assets = valid_counts[valid_counts >= min_required].index

returns_window = returns_window[valid_assets]

# Remove zero-variance assets
variance = returns_window.var()
non_zero_var = variance[variance > 1e-10].index
returns_window = returns_window[non_zero_var]

st.info(f"📊 Using {len(returns_window.columns)} assets with sufficient data in {corr_win_s} window")

# Calculate volatility
with st.spinner("Calculating volatility..."):
    volatility = returns.rolling(
        window=vol_window_days,
        min_periods=max(2, vol_window_days // 2)
    ).std() * np.sqrt(365)
    
    volatility_window = volatility.loc[window_start:end_date]
    volatility_window = volatility_window[valid_assets]
    
    # Filter volatility
    vol_variance = volatility_window.var()
    valid_vol = vol_variance[vol_variance > 1e-10].index
    volatility_window = volatility_window[valid_vol]

# Calculate correlations
with st.spinner("Computing correlation matrices..."):
    min_overlap = max(3, corr_window_days // 6)
    
    corr_price = pairwise_corr(returns_window, min_overlap=min_overlap)
    corr_vol = pairwise_corr(volatility_window, min_overlap=min_overlap)

# --- Display Results ---
st.header("🎯 Results")

tab1, tab2, tab3 = st.tabs(["📈 Price Correlations", "📊 Volatility Correlations", "📋 Summary"])

with tab1:
    if not corr_price.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_heatmap(corr_price, f"Price Correlations ({corr_win_s})", "heatmap_price")
        
        with col2:
            st.subheader("🏆 Top Correlated Pairs")
            top_price = top_pairs(corr_price, k=topn)
            
            if not top_price.empty:
                st.dataframe(
                    top_price[["Asset A", "Asset B", "Correlation"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                csv = corr_price.to_csv()
                st.download_button(
                    "📥 Download Matrix",
                    csv.encode(),
                    "price_correlations.csv",
                    "text/csv"
                )
            else:
                st.warning("No significant correlations found")
    else:
        st.error("Unable to calculate price correlations")

with tab2:
    if not corr_vol.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_heatmap(corr_vol, f"Volatility Correlations ({corr_win_s}, σ={vol_roll_s}d)", "heatmap_vol")
        
        with col2:
            st.subheader("🏆 Top Correlated Pairs")
            top_vol = top_pairs(corr_vol, k=topn)
            
            if not top_vol.empty:
                st.dataframe(
                    top_vol[["Asset A", "Asset B", "Correlation"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                csv = corr_vol.to_csv()
                st.download_button(
                    "📥 Download Matrix",
                    csv.encode(),
                    "volatility_correlations.csv",
                    "text/csv"
                )
            else:
                st.warning("No significant correlations found")
    else:
        st.error("Unable to calculate volatility correlations")

with tab3:
    st.subheader("📊 Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Assets Analyzed", len(series))
        st.metric("Assets in Correlation Window", len(returns_window.columns))
        st.metric("Analysis Window", corr_win_s)
        st.metric("Volatility Period", f"{vol_roll_s} days")
    
    with col2:
        st.metric("Date Range", f"{prices.index.min().date()} to {end_date.date()}")
        st.metric("Total Days", len(prices))
        st.metric("Min Overlap for Correlation", min_overlap)
        st.metric("Coverage at End Date", f"{coverage_at_end:.1%}")
    
    st.divider()
    
    st.subheader("💡 Liquidity Pool Insights")
    
    if not top_price.empty:
        highest_corr = top_price.iloc[0]
        st.info(f"""
        **Highest Price Correlation:** {highest_corr['Asset A']} ↔ {highest_corr['Asset B']} 
        (ρ = {highest_corr['Correlation']:.3f})
        
        ➡️ High positive correlation suggests lower impermanent loss risk but potentially lower fees.
        """)
    
    if not top_vol.empty:
        highest_vol_corr = top_vol.iloc[0]
        st.info(f"""
        **Highest Volatility Correlation:** {highest_vol_corr['Asset A']} ↔ {highest_vol_corr['Asset B']}
        (ρ = {highest_vol_corr['Correlation']:.3f})
        
        ➡️ Similar volatility patterns may indicate correlated market risks.
        """)

# --- Debug Information ---
with st.expander("🔍 Debug Information"):
    st.json({
        "API Endpoint": st.session_state.api_base,
        "API Key Present": bool(API_KEY),
        "Assets Fetched": len(ids),
        "Assets with Data": len(series),
        "Failed Assets": len(failed),
        "Date Range": f"{prices.index.min().date()} to {prices.index.max().date()}",
        "Window Start": window_start.date(),
        "Window End": end_date.date(),
        "Correlation Window": f"{corr_window_days} days",
        "Volatility Window": f"{vol_window_days} days",
        "Min Overlap": min_overlap,
        "Price Matrix Shape": str(prices.shape),
        "Returns Window Shape": str(returns_window.shape),
        "Valid Price Correlations": int(np.isfinite(corr_price.values).sum()) if not corr_price.empty else 0,
        "Valid Vol Correlations": int(np.isfinite(corr_vol.values).sum()) if not corr_vol.empty else 0
    })

st.divider()
st.caption("""
**📚 Guide for Liquidity Providers:**
- **High Correlation (>0.7):** Lower IL risk, stable pools, good for risk-averse LPs
- **Medium Correlation (0.3-0.7):** Balanced risk/reward
- **Low/Negative Correlation (<0.3):** Higher IL risk but potentially higher trading fees
- **Consider both price AND volatility correlations for complete risk assessment**
""")

# streamlit_stock_dashboard.py
# Enhanced Streamlit Stock Dashboard with improved prediction performance:
# - Cached data fetching with validation and auto-adjust.
# - Expanded indicators (ATR, Bollinger Bands, ADX) with tunable periods.
# - Weighted rule-based signals with ADX trend filter and ATR-normalized volume.
# - Confidence-based recommendations with SPY correlation and signal thresholds.
# - Advanced Monte Carlo (lognormal, bootstrapped, rolling volatility).
# - Batch watchlist summaries with fundamentals (P/E).
# - Placeholder for sentiment (requires nltk, vaderSentiment).
# - Fixed: Added OHLCV column validation in get_data to prevent KeyError on missing 'Low'/'High'.
# - Added: CNN Fear & Greed Index for market sentiment (with fallback to yesterday's date and User-Agent).

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')
import requests
import json

# For sentiment (optional: pip install nltk vaderSentiment)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    st.warning("Install nltk and vaderSentiment for news sentiment analysis.")

st.set_page_config(layout='wide', page_title='Enhanced Stock Watch & Signal Dashboard')

# ------------------------- Helper functions -------------------------

@st.cache_data
def get_data(ticker, period='1y', interval='1d'):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
        # Validate full OHLCV structure
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = set(required_cols) - set(hist.columns)
        if hist.empty or len(missing_cols) > 0:
            raise ValueError(f"Incomplete OHLCV data: missing columns {missing_cols}")
        if len(hist) < 50:
            raise ValueError("Insufficient historical data (need at least 50 periods)")
        info = tk.info if hasattr(tk, 'info') else {}
        # Optional: Polygon fallback (requires polygon-io client; pip install polygon-api-client)
        # from polygon import RESTClient
        # try:
        #     client = RESTClient(api_key="YOUR_KEY")
        #     hist = pd.DataFrame(client.get_aggs(ticker, 1, "day", from_="2024-01-01", to=datetime.now().strftime("%Y-%m-%d")).results)
        #     hist['Datetime'] = pd.to_datetime(hist['timestamp'], unit='ms')
        #     hist.set_index('Datetime', inplace=True)
        # except:
        #     pass  # Fall back to yfinance
        return hist, info
    except Exception as e:
        st.error(f'Error fetching data for {ticker}: {e}')
        return pd.DataFrame(), {}

def calc_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, sma_short=20, sma_long=50, bb_period=20, atr_period=14, adx_period=14):
    df = df.copy()
    # SMAs
    df['SMA_short'] = df['Close'].rolling(sma_short).mean()
    df['SMA_long'] = df['Close'].rolling(sma_long).mean()
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period-1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    # Bollinger Bands
    bb_mid = df['Close'].rolling(bb_period).mean()
    bb_std = df['Close'].rolling(bb_period).std()
    df['BB_upper'] = bb_mid + (bb_std * 2)
    df['BB_lower'] = bb_mid - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_mid
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = tr.ewm(span=atr_period, adjust=False).mean()  # EMA for smoothness
    # ADX (simplified directional movement)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / df['ATR'])
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / df['ATR'])
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(span=adx_period).mean()
    return df

def rule_based_signal(df, rsi_oversold=30, rsi_overbought=70):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    weights = {'RSI': 2.0, 'MACD': 1.5, 'SMA': 1.0, 'BB': 1.0, 'Volume': 0.5, 'ADX': 1.0}

    # RSI
    if latest['RSI'] < rsi_oversold:
        signals.append(('RSI oversold -> BUY', weights['RSI']))
    elif latest['RSI'] > rsi_overbought:
        signals.append(('RSI overbought -> SELL', weights['RSI']))

    # MACD
    if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
        signals.append(('MACD bullish crossover -> BUY', weights['MACD']))
    elif (prev['MACD'] > prev['MACD_signal']) and (latest['MACD'] < latest['MACD_signal']):
        signals.append(('MACD bearish crossover -> SELL', weights['MACD']))

    # SMA
    if latest['Close'] > latest['SMA_long']:
        signals.append(('Price above long SMA -> BULLISH', weights['SMA']))
    else:
        signals.append(('Price below long SMA -> BEARISH', weights['SMA']))

    # Bollinger Bands
    if latest['Close'] < latest['BB_lower']:
        signals.append(('Price below BB lower -> BUY', weights['BB']))
    elif latest['Close'] > latest['BB_upper']:
        signals.append(('Price above BB upper -> SELL', weights['BB']))

    # Volume normalized by ATR (relative volume)
    vol_avg20 = df['Volume'].rolling(20).mean().iloc[-1]
    atr_norm_factor = (1 + (latest['ATR'] / latest['Close']) if not np.isnan(latest['ATR']) else 1)
    vol_atr_norm = latest['Volume'] / (vol_avg20 * atr_norm_factor)
    if vol_atr_norm > 1.5:
        signals.append(('Normalized volume spike -> CONFIRMS recent move', weights['Volume']))

    # ADX trend strength (skip if NaN)
    if not np.isnan(latest['ADX']) and latest['ADX'] > 25:
        signals.append(('Strong trend (ADX>25) -> AMPLIFY signals', weights['ADX']))
    elif not np.isnan(latest['ADX']):
        signals.append(('Weak trend (ADX<25) -> CAUTION', -weights['ADX'] * 0.5))  # Mild bearish weight

    # Weighted votes
    buy_votes = sum(w for s, w in signals if 'BUY' in s or 'BULLISH' in s or 'AMPLIFY' in s)
    sell_votes = sum(w for s, w in signals if 'SELL' in s or 'BEARISH' in s)
    total_weight = sum(abs(w) for s, w in signals)
    confidence = max(-100, min(100, (buy_votes - sell_votes) / total_weight * 100)) if total_weight > 0 else 0

    # Recommendation with threshold
    aligned_signals = sum(1 for s, _ in signals if 'BUY' in s or 'SELL' in s or 'BULLISH' in s or 'BEARISH' in s)
    if buy_votes > sell_votes and aligned_signals >= 3:
        recommendation = 'STRONG BUY'
    elif buy_votes > sell_votes:
        recommendation = 'BUY'
    elif sell_votes > buy_votes and aligned_signals >= 3:
        recommendation = 'STRONG SELL'
    elif sell_votes > buy_votes:
        recommendation = 'SELL'
    else:
        recommendation = 'HOLD'

    return recommendation, signals, confidence

def get_correlation(ticker_df, spy_ticker='SPY'):
    try:
        spy_hist, _ = get_data(spy_ticker, period='1y', interval='1d')
        if not spy_hist.empty and len(spy_hist) == len(ticker_df):
            corr = ticker_df['Close'].corr(spy_hist['Close'])
            return corr if not np.isnan(corr) else 0
    except:
        pass
    return 0

def get_sentiment_score(info):
    if not SENTIMENT_AVAILABLE or 'news' not in info:
        return None
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(article.get('title', '') + ' ' + article.get('publisher', ''))['compound']
              for article in info['news'][:5]]  # Top 5 recent
    return np.mean(scores) if scores else None

def estimate_days_to_target(df, current_price, target_return, sims=10000, max_days=365, bootstrap=True):
    returns = df['Close'].pct_change().dropna()
    if len(returns) < 50:
        return {'probability': 0, 'median_days': None, '90pct_days': None}
    
    # Rolling volatility (last 252 days ~1y)
    rolling_sigma = returns.rolling(252, min_periods=50).std().iloc[-1]
    sigma = rolling_sigma if not np.isnan(rolling_sigma) else returns.std()
    mu = returns.mean()
    
    days_to_target = []
    for _ in range(sims):
        if bootstrap:
            sampled_returns = np.random.choice(returns, size=min(max_days, len(returns)), replace=True)
            price = current_price
            day = 0
            for r in sampled_returns:
                day += 1
                price *= (1 + r)
                if (price / current_price - 1) >= target_return:
                    days_to_target.append(day)
                    break
                if day >= max_days:
                    break
            if day < max_days:
                days_to_target.append(np.nan)
        else:
            # Lognormal fallback
            log_mu = np.log(1 + mu) - 0.5 * sigma**2
            for day in range(1, max_days + 1):
                r = np.random.lognormal(mean=log_mu, sigma=sigma)
                price = current_price * r
                if (price / current_price - 1) >= target_return:
                    days_to_target.append(day)
                    break
            else:
                days_to_target.append(np.nan)
    
    days_arr = np.array(days_to_target, dtype=float)
    prob_reach = np.sum(~np.isnan(days_arr)) / sims
    valid_days = days_arr[~np.isnan(days_arr)]
    median_days = np.median(valid_days) if len(valid_days) > 0 else np.nan
    perc90 = np.percentile(valid_days, 90) if len(valid_days) > 0 else np.nan
    return {'probability': prob_reach, 'median_days': float(median_days) if not np.isnan(median_days) else None, '90pct_days': float(perc90) if not np.isnan(perc90) else None}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_fear_greed_index():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    dates_to_try = [
        date.today().isoformat(),
        (date.today() - timedelta(days=1)).isoformat(),
        (date.today() - timedelta(days=2)).isoformat()
    ]
    
    for d in dates_to_try:
        try:
            url = base_url + d
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            fear_greed = data['fear_and_greed']
            score = fear_greed['score']
            rating = fear_greed['rating']
            
            # Color logic
            if score < 25:
                color = '🟥 Extreme Fear'
            elif score < 45:
                color = '🔴 Fear'
            elif score < 55:
                color = '🟡 Neutral'
            elif score < 75:
                color = '🟢 Greed'
            else:
                color = '🟩 Extreme Greed'
            
            return score, rating, color
        except Exception as e:
            continue  # Try next date
    
    st.warning("Could not fetch Fear & Greed data. Using fallback.")
    return None, 'N/A', 'N/A'

# ------------------------- Streamlit UI -------------------------

st.title('Enhanced Stock Watch + Indicator Dashboard')

# Fear & Greed Index (market-wide)
st.markdown("---")
st.subheader("Market Sentiment: Fear & Greed Index")
score, rating, color_emoji = get_fear_greed_index()
if score is not None:
    col_fg1, col_fg2, col_fg3 = st.columns([1, 3, 1])
    with col_fg1:
        st.metric("Score", score, delta=None)
    with col_fg2:
        st.write(f"**{rating}** {color_emoji}")
    with col_fg3:
        st.progress(score / 100)  # Visual gauge
else:
    st.info("Fear & Greed data unavailable.")
st.markdown("---")

# Static groups (updated approximate caps as of Oct 2025)
big_cap = 'AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, JNJ, V'
medium_cap = 'ADBE, AMD, LULU, SQ, SNOW, PYPL, RIVN, DOCU, DASH, ZM'
small_cap = 'SOFI, HOOD, RKT, BB, GPRO'  # Adjusted for actual small caps

group_selection = st.sidebar.radio('Select Market Cap Group', ['Big Cap (>$10B)', 'Medium Cap ($1B–$10B)', 'Small Cap (<$1B)'])

if group_selection.startswith('Big'):
    default_tickers = big_cap
elif group_selection.startswith('Medium'):
    default_tickers = medium_cap
else:
    default_tickers = small_cap

with st.sidebar:
    st.header('Watchlist')
    tickers_input = st.text_area('Enter tickers (comma separated)', value=default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    st.markdown('---')
    st.header('Settings')
    lookback = st.selectbox('Historical lookback', options=['6mo','1y','2y','5y'], index=1)
    interval = st.selectbox('Data interval', options=['1d','1wk'], index=0)
    # Tunable parameters
    st.header('Indicator Tunings')
    rsi_period = st.slider('RSI Period', 10, 20, 14)
    rsi_oversold = st.slider('RSI Oversold Threshold', 20, 40, 30)
    rsi_overbought = st.slider('RSI Overbought Threshold', 60, 80, 70)
    # ... (add more sliders as needed)
    run_button = st.button('Refresh data')

if not tickers:
    st.info('Add some tickers in the left sidebar to start.')
    st.stop()

# Batch watchlist summary
st.subheader('Watchlist Summary')
if st.button('Generate Batch Summary'):
    batch_results = []
    for ticker in tickers[:10]:  # Limit to 10 for performance
        hist, info = get_data(ticker, period=lookback, interval=interval)
        if hist.empty:
            continue
        df = calc_indicators(hist, rsi_period=rsi_period)
        rec, _, conf = rule_based_signal(df, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought)
        pe = info.get('forwardPE', 'N/A')
        corr = get_correlation(df)
        sentiment = get_sentiment_score(info)
        batch_results.append({
            'Ticker': ticker,
            'Price': f"${df['Close'].iloc[-1]:.2f}",
            'Rec': rec,
            'Conf (%)': f"{conf:.1f}",
            'SPY Corr': f"{corr:.2f}",
            'P/E': pe,
            'Sentiment': f"{sentiment:.2f}" if sentiment is not None else 'N/A'
        })
    st.table(pd.DataFrame(batch_results))

col1, col2 = st.columns([1,2])

with col1:
    st.subheader('Monitored stocks')
    selected = st.selectbox('Select a stock', options=tickers)
    st.write('Selected:', selected)

with col2:
    st.subheader('Quick summary (latest)')
    hist, info = get_data(selected, period=lookback, interval=interval)
    if hist.empty:
        st.error('No data. Try another ticker or interval.')
        st.stop()
    df = calc_indicators(hist, rsi_period=rsi_period)
    latest = df.iloc[-1]

    price = f"${latest['Close']:.2f}"
    volume_m = latest['Volume'] / 1_000_000
    vol_str = f"{volume_m:.2f}M pcs"
    market_cap = info.get('marketCap', None)
    if market_cap:
        market_cap_b = market_cap / 1_000_000
        mc_str = f"${market_cap_b:.0f}B"
    else:
        mc_str = 'N/A'
    pe = info.get('forwardPE', 'N/A')
    pe_str = f"{pe:.1f}x" if isinstance(pe, (int, float)) else pe

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Price', price)
    c2.metric('Volume', vol_str)
    c3.metric('Market Cap', mc_str)
    c4.metric('Fwd P/E', pe_str)

    corr = get_correlation(df)
    st.metric('SPY Correlation', f"{corr:.2f}")

    sentiment = get_sentiment_score(info)
    if sentiment is not None:
        st.metric('News Sentiment', f"{sentiment:.2f}")

st.markdown('---')

st.subheader('Price chart & Indicators')
price_col, ind_col = st.columns([3,1])
with price_col:
    st.line_chart(df['Close'])
    st.write('SMA short/long:')
    st.line_chart(df[['SMA_short','SMA_long']].dropna())
    st.write('Bollinger Bands:')
    bb_df = df[['BB_upper', 'BB_lower', 'Close']].dropna()
    st.line_chart(bb_df)
with ind_col:
    st.write('Latest indicators')
    ind_latest = pd.DataFrame({
        'Value': [
            f"{latest['RSI']:.2f}",
            f"{latest['MACD']:.4f}",
            f"{latest['MACD_signal']:.4f}",
            f"{latest['ATR']:.4f}",
            f"{latest['ADX']:.2f}",
            f"{latest['BB_width']:.4f}"
        ]
    }, index=['RSI','MACD','MACD_signal','ATR','ADX','BB_width'])
    st.write(ind_latest)

st.markdown('---')

st.subheader('Recommendation & Estimated holding periods')
recommendation, signals, confidence = rule_based_signal(df)
st.write('Algorithmic signals:')
for s, _ in signals:
    st.write('- ' + s)
color = 'success' if 'BUY' in recommendation else 'error' if 'SELL' in recommendation else 'warning'
st.markdown(f"<h4 style='color: {'green' if 'BUY' in recommendation else 'red' if 'SELL' in recommendation else 'orange'}'>{recommendation} (Confidence: {confidence:.1f}%)</h4>", unsafe_allow_html=True)

targets = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00]
results = []
current_price = latest['Close']
for t in targets:
    res = estimate_days_to_target(df, current_price, target_return=t, sims=10000, bootstrap=True)
    results.append({
        'Target (%)': int(t*100),
        'Prob. to reach target within 1 year (%)': round(res['probability']*100,2),
        'Median Days': res['median_days'],
        '90th Percentile Days': res['90pct_days']
    })

st.table(pd.DataFrame(results))

st.markdown('---')

st.write('Notes and limitations:')
st.write('- Enhanced with weighted signals, ADX, ATR/BB, SPY corr, sentiment (if enabled), and bootstrapped MC sims.')
st.write('- Added CNN Fear & Greed Index for market sentiment (fetches latest available date with User-Agent).')
st.write('- Backtest rules for your portfolio; predictions are probabilistic.')
st.write('- Tune via sidebar; for production, add backtesting (e.g., backtrader).')

st.write('To run:')
st.code("""
# Install dependencies
pip install streamlit yfinance pandas numpy nltk vaderSentiment requests
# Run
streamlit run streamlit_stock_dashboard.py
""")

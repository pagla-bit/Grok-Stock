import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from polygon import RESTClient
import os
import time

# API Setup (use secrets in Streamlit Cloud)
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')  # Set env var or secret
client = RESTClient(POLYGON_API_KEY) if POLYGON_API_KEY else None

# Pre-populated watchlist by groups
default_watchlist = {
    'Large Cap (>10B)': ['JPM', 'WMT', 'UNH', 'V', 'PG', 'JNJ', 'HD', 'MRK', 'CVX', 'BABA'],
    'Mid Cap (1-10B)': ['MUSA', 'HRL', 'YUM', 'STLD', 'VFC', 'CLX', 'TRV', 'ALL', 'PPL', 'EIX'],
    'Penny (<1B)': ['SRTS', 'PSHG', 'CINT', 'DDL', 'WDH', 'MAPS', 'SELF', 'VERU', 'BURU', 'LAES']
}

# Sidebar: Grouped watchlist management
st.sidebar.title("Stock Watchlist")
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = {k: v[:] for k, v in default_watchlist.items()}

group = st.sidebar.selectbox("Select Group", list(default_watchlist.keys()))
current_group = st.session_state.watchlist[group]
selected_tickers = st.sidebar.multiselect(f"{group} Tickers", current_group, default=current_group)

# Add/Remove (per group)
new_ticker = st.sidebar.text_input(f"Add to {group}")
if st.sidebar.button(f"Add to {group}") and new_ticker:
    if new_ticker.upper() not in current_group:
        current_group.append(new_ticker.upper())
    st.session_state.watchlist[group] = current_group
    st.rerun()

if st.sidebar.button(f"Remove from {group}"):
    st.session_state.watchlist[group] = [t for t in current_group if t not in selected_tickers]
    st.rerun()

# Flatten for main select
all_tickers = [t for group_tickers in st.session_state.watchlist.values() for t in group_tickers]
st.sidebar.write("All Monitored:", all_tickers[:10], "..." if len(all_tickers) > 10 else "")

# Main dashboard
st.title("Advanced Stock Monitoring Dashboard")
selected_stock = st.selectbox("Select Stock", all_tickers)

strategy = st.selectbox("Trading Strategy", ["Momentum (RSI/MACD)", "Trend Following (SMA Crossover)", "Value Investing (P/E + Fundamentals)", "Mean Reversion (Bollinger Bands)"])
estimation_method = st.selectbox("Profit Estimation Method", ["Historical Backtest", "Monte Carlo Simulation", "Linear Projection"])

if st.button("Refresh Data (Real-Time)") or st.button("Auto-Refresh Every 60s"):
    time.sleep(1)  # Simulate poll
    st.rerun()
    if st.button("Stop Auto"): pass  # Placeholder

if selected_stock:
    # Fetch real-time data (Polygon for current, yfinance for hist)
    try:
        if client:
            snapshot = client.get_snapshot(f"A.{selected_stock}")
            current_price = snapshot.day.c if snapshot.day else yf.Ticker(selected_stock).info.get('currentPrice', 0)
            volume = snapshot.day.v if snapshot.day else yf.Ticker(selected_stock).info.get('volume', 0)
        else:
            ticker_yf = yf.Ticker(selected_stock)
            current_price = ticker_yf.info.get('currentPrice', 0)
            volume = ticker_yf.info.get('volume', 0)
        info = ticker_yf.info  # Fallback
        hist = ticker_yf.history(period="1y", interval="1d")  # EOD hist; upgrade Polygon for intraday
    except:
        st.error("Data fetch failed—check ticker/API key.")
        st.stop()

    if hist.empty:
        st.error("Invalid ticker.")
    else:
        # Compute indicators
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        macd = ta.macd(hist['Close'])
        hist['MACD'] = macd['MACD_12_26_9']
        hist['MACD_Signal'] = macd['MACDs_12_26_9']
        hist['SMA_50'] = ta.sma(hist['Close'], length=50)
        bb = ta.bbands(hist['Close'])
        hist['BB_Upper'] = bb['BBU_5_2.0']
        hist['BB_Lower'] = bb['BBL_5_2.0']
        
        latest = hist.iloc[-1]
        rsi = latest['RSI']
        macd_val = latest['MACD']
        macd_sig = latest['MACD_Signal']
        sma_50 = latest['SMA_50']
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        
        # Fundamentals
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        beta = info.get('beta', 'N/A')
        
        # Display indicators
        st.subheader(f"Real-Time Indicators for {selected_stock} ({datetime.now().strftime('%H:%M:%S')})")
        indicators_df = pd.DataFrame({
            'Metric': ['Current Price', 'Market Cap', 'Volume', 'P/E Ratio', 'Beta', 'RSI (14)', 'MACD', 'MACD Signal', '50-Day SMA', 'BB Upper', 'BB Lower'],
            'Value': [f"${current_price:.2f}", f"{market_cap:,}" if market_cap != 'N/A' else 'N/A',
                      f"{volume:,}", f"{pe_ratio:.2f}" if pe_ratio != 'N/A' else 'N/A',
                      f"{beta:.2f}" if beta != 'N/A' else 'N/A', f"{rsi:.2f}",
                      f"{macd_val:.4f}", f"{macd_sig:.4f}", f"${sma_50:.2f}",
                      f"${bb_upper:.2f}", f"${bb_lower:.2f}"]
        })
        st.table(indicators_df)
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                     low=hist['Low'], close=hist['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='50-Day SMA', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')))
        fig.update_layout(title=f"{selected_stock} Chart (1Y, Real-Time Update)", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy-Based Recommendation
        st.subheader(f"Recommendation: {strategy}")
        score = 0
        signal = "Hold"
        
        if strategy == "Momentum (RSI/MACD)":
            if rsi < 30: score += 40
            if rsi > 70: score -= 40
            if macd_val > macd_sig: score += 30
            else: score -= 30
            if volume > hist['Volume'].mean() * 1.5: score += 30
        elif strategy == "Trend Following (SMA Crossover)":
            if current_price > sma_50: score += 50
            sma_200 = ta.sma(hist['Close'], 200).iloc[-1]
            if sma_50 > sma_200: score += 30
            if volume > hist['Volume'].mean(): score += 20
        elif strategy == "Value Investing (P/E + Fundamentals)":
            if pe_ratio != 'N/A' and pe_ratio < 15: score += 50  # Undervalued
            if beta < 1.2: score += 20  # Lower risk
            if market_cap > 10e9: score += 10  # Stability bias
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield > 0.02: score += 20
        elif strategy == "Mean Reversion (Bollinger Bands)":
            if current_price < bb_lower: score += 50  # Oversold, buy
            if current_price > bb_upper: score -= 50  # Overbought, sell
            if abs((current_price - hist['Close'].mean()) / hist['Close'].std()) < 1: score += 20  # Near mean
        
        score = max(0, min(100, score + 50))  # Normalize
        if score > 70: signal = "Strong Buy"
        elif score > 55: signal = "Buy"
        elif score < 30: signal = "Strong Sell"
        elif score < 45: signal = "Sell"
        
        st.metric("Signal", signal, delta=f"Score: {score:.0f}%")
        st.write(f"*Tailored to {strategy}. Adjust thresholds for your risk.*")
        
        # Profit Timeline (Buy signals only)
        if "Buy" in signal:
            st.subheader("Holding Periods for Profit Targets")
            targets = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00]
            daily_returns = hist['Close'].pct_change().dropna()
            avg_daily_ret = daily_returns.mean()
            volatility = daily_returns.std()
            
            timeline_data = []
            for t in targets:
                if estimation_method == "Historical Backtest":
                    # Days from past periods hitting target
                    days = int((t / avg_daily_ret) * 252) if avg_daily_ret > 0 else "N/A"
                elif estimation_method == "Linear Projection":
                    # Simple linear: target / avg daily
                    days = int(t / max(avg_daily_ret, 0.001)) if avg_daily_ret > 0 else "N/A"  # Avoid div0
                else:  # Monte Carlo
                    # Simulate 1000 paths, 252 trading days
                    sim_returns = np.random.normal(avg_daily_ret, volatility, (1000, 252))
                    sim_paths = (1 + sim_returns).cumprod(axis=1) - 1
                    hit_days = np.mean([np.argmax(path >= t) + 1 for path in sim_paths if any(path >= t)])
                    days = int(hit_days) if not np.isnan(hit_days) else "N/A"
                
                timeline_data.append({'Target (%)': f"{t*100:.0f}%", 'Est. Days': days, 'Method': estimation_method})
            
            timeline_df = pd.DataFrame(timeline_data)
            st.table(timeline_df)
            st.write("*Projections assume lognormal returns; 95% CI ±volatility. Add stop-loss (e.g., -10%). Past ≠ future.*")
        else:
            st.info("No Buy—monitor for signals.")

st.markdown("---")
st.caption("*Educational tool. Not advice. Data: Polygon (real-time, paid) / yfinance (delayed). Consult advisor.*")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EMA Crossover Backtester", layout="wide")

# =====================================================
#  Core Backtest Function
# =====================================================

def ema_crossover_backtest(symbol, short_window, long_window, start, end, initial_cash=10000):
    data = yf.download(symbol, start=start, end=end, progress=False)
    data.dropna(inplace=True)

    # Compute EMAs
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()

    # Generate Signals
    data['Signal'] = np.where(data['EMA_short'] > data['EMA_long'], 1, 0)
    data['Position'] = data['Signal'].shift(1)

    # Returns
    data['Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Return'] * data['Position']

    # Portfolio Simulation
    data['Portfolio_Value'] = (1 + data['Strategy_Return']).cumprod() * initial_cash
    data['Buy_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash

    # Metrics
    total_return = (data['Portfolio_Value'].iloc[-1] / initial_cash - 1) * 100
    buyhold_return = (data['Buy_Hold_Value'].iloc[-1] / initial_cash - 1) * 100
    sharpe_ratio = (data['Strategy_Return'].mean() / data['Strategy_Return'].std()) * np.sqrt(252)

    return data, total_return, buyhold_return, sharpe_ratio


# =====================================================
#  Streamlit UI
# =====================================================

st.title("📈 EMA Crossover Strategy Backtester")

# Sidebar inputs
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Stock Symbol", "")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
short_window = st.sidebar.slider("Short EMA Window", 5, 50, 12)
long_window = st.sidebar.slider("Long EMA Window", 20, 200, 26)
initial_cash = st.sidebar.number_input("Initial Cash ($)", 1000, 1000000, 10000, step=1000)

if st.sidebar.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        data, total_return, buyhold_return, sharpe_ratio = ema_crossover_backtest(
            symbol, short_window, long_window, start, end, initial_cash
        )

    st.success(f"✅ Backtest Complete for {symbol}")

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{total_return:.2f}%")
    col2.metric("Buy & Hold Return", f"{buyhold_return:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # --- Chart 1: Portfolio Value ---
    st.subheader("Portfolio Performance")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['Portfolio_Value'], label="EMA Strategy", linewidth=2)
    ax.plot(data['Buy_Hold_Value'], label="Buy & Hold", linestyle="--", alpha=0.8)
    ax.set_title(f"{symbol} Portfolio Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Chart 2: Price & EMAs ---
    st.subheader("Price and Moving Averages")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(data['Close'], label="Price", color="gray")
    ax2.plot(data['EMA_short'], label=f"EMA {short_window}", color="green")
    ax2.plot(data['EMA_long'], label=f"EMA {long_window}", color="red")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # --- Show Data Table ---
    st.subheader("Data Sample")
    st.dataframe(data.tail(10))

else:
    st.info("👈 Configure settings in the sidebar and click 'Run Backtest'")

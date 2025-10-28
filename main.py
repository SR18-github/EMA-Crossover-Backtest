import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
#  EMA Crossover Backtester with On-Screen Display
# =====================================================

def ema_crossover_backtest(symbol, short_window=12, long_window=26,
                            start='2020-01-01', end='2025-01-01',
                            initial_cash=10000):
    # --- Load Data ---
    data = yf.download(symbol, start=start, end=end, progress=False)
    data.dropna(inplace=True)

    # --- Compute EMAs ---
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()

    # --- Generate Signals ---
    data['Signal'] = np.where(data['EMA_short'] > data['EMA_long'], 1, 0)
    data['Position'] = data['Signal'].shift(1)

    # --- Returns ---
    data['Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Return'] * data['Position']

    # --- Portfolio Simulation ---
    data['Portfolio_Value'] = (1 + data['Strategy_Return']).cumprod() * initial_cash
    data['Buy_Hold_Value'] = (1 + data['Return']).cumprod() * initial_cash

    # --- Trade Analysis ---
    data['Trade'] = data['Position'].diff()
    entries = data[data['Trade'] == 1].index
    exits = data[data['Trade'] == -1].index
    if len(entries) > len(exits):
        exits = exits.append(pd.Index([data.index[-1]]))

    trades = pd.DataFrame({'Entry': entries, 'Exit': exits})
    if not trades.empty:
        trades['Entry_Price'] = data.loc[trades['Entry'], 'Close'].values
        trades['Exit_Price'] = data.loc[trades['Exit'], 'Close'].values
        trades['Return_%'] = (trades['Exit_Price'] / trades['Entry_Price'] - 1) * 100
        win_rate = (trades['Return_%'] > 0).mean() * 100
        avg_trade = trades['Return_%'].mean()
    else:
        win_rate = avg_trade = 0

    # --- Performance Metrics ---
    total_return = (data['Portfolio_Value'].iloc[-1] / initial_cash - 1) * 100
    buyhold_return = (data['Buy_Hold_Value'].iloc[-1] / initial_cash - 1) * 100
    sharpe_ratio = (data['Strategy_Return'].mean() / data['Strategy_Return'].std()) * np.sqrt(252)

    # --- Visualization ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"{symbol} EMA Crossover Backtest", fontsize=16, weight='bold')

    # ---- Chart 1: Portfolio Performance ----
    ax1.plot(data['Portfolio_Value'], label='EMA Strategy', linewidth=2)
    ax1.plot(data['Buy_Hold_Value'], label='Buy & Hold', linestyle='--', alpha=0.8)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True)

    # ---- Chart 2: Price and EMAs ----
    ax2.plot(data['Close'], label='Price', color='gray', alpha=0.7)
    ax2.plot(data['EMA_short'], label=f'Short EMA ({short_window})', color='green')
    ax2.plot(data['EMA_long'], label=f'Long EMA ({long_window})', color='red')
    ax2.set_ylabel("Price ($)")
    ax2.legend()

    # ---- Add Metrics Box ----
    metrics_text = (
        f"Total Return: {total_return:.2f}%\n"
        f"Buy & Hold: {buyhold_return:.2f}%\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"Trades: {len(trades)}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Avg Trade: {avg_trade:.2f}%"
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax1.text(1.02, 0.5, metrics_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='center', bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.show()

    # --- Show Trade Table ---
    if not trades.empty:
        print("\nRecent Trades:")
        print(trades.tail(10).to_string(index=False))

    return data


# =====================================================
#  Optimization Function
# =====================================================

def optimize_ema(symbol, start='2020-01-01', end='2025-01-01'):
    best_return = -np.inf
    best_params = (0, 0)

    print(f"\n🔎 Optimizing {symbol} EMA Parameters...\n")

    for short in range(5, 21, 5):
        for long in range(20, 61, 10):
            if short >= long:
                continue

            data = yf.download(symbol, start=start, end=end, progress=False)
            data['EMA_short'] = data['Close'].ewm(span=short, adjust=False).mean()
            data['EMA_long'] = data['Close'].ewm(span=long, adjust=False).mean()
            data['Signal'] = np.where(data['EMA_short'] > data['EMA_long'], 1, 0)
            data['Position'] = data['Signal'].shift(1)
            data['Return'] = data['Close'].pct_change()
            data['Strategy_Return'] = data['Return'] * data['Position']

            total_return = (1 + data['Strategy_Return']).prod() - 1

            print(f"Tested Short={short}, Long={long} → Return={total_return*100:.2f}%")

            if total_return > best_return:
                best_return = total_return
                best_params = (short, long)

    print(f"\n🔥 Best Parameters for {symbol}: Short EMA={best_params[0]}, Long EMA={best_params[1]} | Return={best_return*100:.2f}%")
    return best_params


# =====================================================
#  Run the Backtest
# =====================================================

if __name__ == "__main__":
    symbol = "IRBT"
    best_short, best_long = optimize_ema(symbol)
    ema_crossover_backtest(symbol, short_window=best_short, long_window=best_long)

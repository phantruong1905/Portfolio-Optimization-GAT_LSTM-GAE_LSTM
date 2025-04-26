import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
plot_path = os.getenv("PLOT_PATH", "visualization")
plot_path = os.path.join(plot_path, "Efficient_Frontier.png")

def run_efficient_frontier_strategy(df, rf_train=0.00, rf_test=0.00, gamma=1.0, train_split=0.9):
    """
    Run Efficient Frontier Max Sharpe Ratio strategy with backtesting.

    Parameters:
        df (DataFrame): Input DataFrame with ['Date', 'Symbol', 'Adj Close'] and other features.
        rf_train (float): Risk-free rate used in training.
        rf_test (float): Risk-free rate used for out-of-sample Sharpe ratio.
        gamma (float): L2 regularization strength.
        train_split (float): Proportion of data to use for training.
    """
    # Step 1: Pivot to get stock price matrix
    price_df = df.pivot(index="Date", columns="Symbol", values="Adj Close").dropna()
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.drop(columns="VNINDEX")

    # Step 2: Train-Test Split
    split_idx = int(len(price_df) * train_split)
    train_prices = price_df.iloc[:split_idx]
    test_prices = price_df.iloc[split_idx:]

    # Step 3: Log returns
    train_returns = np.log(train_prices / train_prices.shift(1)).dropna()
    test_returns = np.log(test_prices / test_prices.shift(1)).dropna()

    # Step 4: Annualized mean & cov
    mu = train_returns.mean() * 252
    Sigma = train_returns.cov() * 252

    # Step 5: Optimize using Efficient Frontier
    ef = EfficientFrontier(mu, Sigma)
    ef.add_objective(objective_functions.L2_reg, gamma=gamma)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("Optimal Portfolio Weights:", cleaned_weights)
    ef.portfolio_performance(verbose=True, risk_free_rate=rf_train)

    # Step 6: Backtest
    weights = list(cleaned_weights.values())
    portfolio_returns = test_returns @ weights

    realized_return = np.mean(portfolio_returns) * 252
    realized_volatility = np.std(portfolio_returns) * np.sqrt(252)
    realized_sharpe = (realized_return - rf_test) / realized_volatility

    print("\nOut-of-Sample Performance:")
    print(f"Annualized Return: {realized_return:.4f}")
    print(f"Annualized Volatility: {realized_volatility:.4f}")
    print(f"Realized Sharpe Ratio: {realized_sharpe:.4f}")

    # Step 7: Plot Cumulative Returns
    sharpe_cumulative = (1 + portfolio_returns).cumprod() - 1
    sharpe_cumulative.index = pd.to_datetime(sharpe_cumulative.index)

    cumulative_return = sharpe_cumulative.iloc[-1]
    print(f"Cumulative Return: {cumulative_return:.4%}")

    return cleaned_weights, sharpe_cumulative


load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data")

stock_data_path = os.path.join(DATA_PATH, "train.pkl")
df = pd.read_pickle(stock_data_path)


run_efficient_frontier_strategy(df)
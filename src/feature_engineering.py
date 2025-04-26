import os
import logging
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ta import add_all_ta_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_features(data_path):
    stock_data_path = os.path.join(data_path, "stock_data.pkl")
    if not os.path.exists(stock_data_path):
        logging.error(f"Stock data not found at {stock_data_path}")
        return

    logging.info("Loading stock data...")
    df = pd.read_pickle(stock_data_path)

    # Add TA features per symbol
    def gen_feature(data):
        return add_all_ta_features(
            data, open="Open", high="High", low="Low", close="Adj Close", volume="Volume"
        )

    logging.info("Generating TA features...")
    df = df.groupby("Symbol").apply(gen_feature).reset_index(drop=True)

    # Select TA + raw price/volume features
    ta_cols = [
        "volatility_atr", "trend_macd", "trend_adx", "trend_sma_fast", "momentum_rsi"
    ]
    raw_cols = ["Open", "High", "Low", "Adj Close", "Volume"]
    selected_cols = ["Date", "Symbol"] + ta_cols + raw_cols

    # Drop rows with any missing TA or raw values
    df = df[selected_cols].dropna()

    # --- Split per symbol, time-based ---
    train_list, val_list, test_list = [], [], []

    for symbol, group in df.groupby("Symbol"):
        group_sorted = group.sort_values("Date").reset_index(drop=True)

        n = len(group_sorted)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train = group_sorted.iloc[:n_train]
        val = group_sorted.iloc[n_train:n_train + n_val]
        test = group_sorted.iloc[n_train + n_val:]

        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    print(train_df.columns)

    # Save final sets
    train_df.to_pickle(os.path.join(data_path, "train.pkl"))
    val_df.to_pickle(os.path.join(data_path, "val.pkl"))
    test_df.to_pickle(os.path.join(data_path, "test.pkl"))

    logging.info("Saved train, val, and test DataFrames with raw + TA features.")

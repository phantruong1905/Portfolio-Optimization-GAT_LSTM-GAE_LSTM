import os
import logging
import pandas as pd
from dotenv import load_dotenv
from vnstock import Vnstock


def fetch_stock_data(data_path, stocks, start_date, end_date):
    """Fetches historical stock data and saves it as a pickle file."""

    vnstock = Vnstock()
    data = []

    for symbol in stocks:
        try:
            logging.info(f"Fetching data for {symbol}...")
            stock = vnstock.stock(symbol=symbol, source='VCI')
            df = stock.quote.history(start=start_date, end=end_date, interval='1D')

            if df.empty:
                logging.warning(f"No data found for {symbol}, skipping...")
                continue

            # Rename columns for consistency
            df.rename(columns={
                "time": "Date",
                "close": "Adj Close",
                "high": "High",
                "low": "Low",
                "open": "Open",
                "volume": "Volume"
            }, inplace=True)

            df["Symbol"] = symbol
            df = df[["Date", "Symbol", "Adj Close", "High", "Low", "Open", "Volume"]]

            # Convert Date to datetime format
            df["Date"] = pd.to_datetime(df["Date"])

            logging.info(
                f"{symbol} | Shape: {df.shape} | From {df['Date'].min().date()} to {df['Date'].max().date()}")

            logging.info(f"Loaded {df.shape[0]} rows for {symbol}")

            data.append(df)

        except Exception as e:
            logging.error(f" Error fetching data for {symbol}: {str(e)}")

    # Merge all stock data into one DataFrame
    if data:
        final_df = pd.concat(data, ignore_index=True)

        final_df.set_index(["Symbol", "Date"], inplace=True)

        # Keep daily data (no resampling)
        final_df = final_df.dropna().reset_index()

        save_path = os.path.join(data_path, "stock_data.pkl")
        final_df.to_pickle(save_path)
        print(final_df.head())

        logging.info(f" Final dataset saved to: {save_path}")
        logging.info(f"Final dataset shape: {final_df.shape}")
    else:
        logging.error("No data was collected!")


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import os
from datetime import datetime
from dotenv import load_dotenv


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Define the StockLSTM and PortfolioLSTM models
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim]
        # Use the last hidden state
        return h_n[-1]  # [batch_size, hidden_dim]


class PortfolioLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_assets, dropout):
        super(PortfolioLSTM, self).__init__()
        self.num_assets = num_assets
        self.stock_lstm = StockLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.fc = nn.Linear(hidden_dim * num_assets, num_assets)

    def forward(self, x, mask):
        # x: [batch_size, num_assets, seq_len, input_dim]
        # mask: [batch_size, num_assets]
        batch_size = x.size(0)
        lstm_outputs = []

        for i in range(self.num_assets):
            stock_input = x[:, i, :, :]  # [batch_size, seq_len, input_dim]
            stock_output = self.stock_lstm(stock_input)  # [batch_size, hidden_dim]
            lstm_outputs.append(stock_output)

        combined = torch.cat(lstm_outputs, dim=1)  # [batch_size, hidden_dim * num_assets]
        weights = self.fc(combined)  # [batch_size, num_assets]
        weights = F.softmax(weights, dim=1)  # [batch_size, num_assets]

        # Apply mask to zero out weights for unavailable stocks
        weights = weights * mask  # [batch_size, num_assets]

        # Re-normalize weights to sum to 1 for available stocks (avoid division by zero)
        weights_sum = torch.sum(weights, dim=1, keepdim=True) + 1e-8
        weights = weights / weights_sum  # [batch_size, num_assets]

        return weights

class PortfolioMSELoss(nn.Module):
    def __init__(self):
        super(PortfolioMSELoss, self).__init__()

    def forward(self, pred_weights, actual_returns, mask):
        # Apply mask to weights and returns
        masked_weights = pred_weights * mask
        portfolio_pred_return = torch.sum(masked_weights * actual_returns, dim=1)  # [batch_size]

        # Target: average return with 1/N allocation (could use other benchmarks here too)
        actual_portfolio_return = torch.sum(mask * actual_returns, dim=1) / (mask.sum(dim=1) + 1e-8)

        return F.mse_loss(portfolio_pred_return, actual_portfolio_return)

# Sharpe Ratio Loss
class SharpeRatioLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0, annualization_factor=252):
        super(SharpeRatioLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def forward(self, pred_weights, actual_returns, mask):
        # pred_weights: [batch_size, num_assets]
        # actual_returns: [batch_size, num_assets]
        # mask: [batch_size, num_assets]

        # Apply mask to weights and returns
        masked_weights = pred_weights * mask  # [batch_size, num_assets]
        portfolio_returns = torch.sum(masked_weights * actual_returns, dim=1)  # [batch_size]

        # Compute mean and std only for batches with at least one valid return
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std(unbiased=False) + 1e-6  # Avoid division by zero

        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
        sharpe_ratio *= torch.sqrt(torch.tensor(self.annualization_factor, device=pred_weights.device))

        return -sharpe_ratio  # Negative for minimization


class SimpleStockDataset(Dataset):
    def __init__(self, embeddings_df, price_df, seq_len, symbols):
        """
        Simplified dataset for portfolio optimization with embeddings and log returns.

        Args:
            embeddings_df: DataFrame with embeddings features
            price_df: DataFrame with price data to calculate log returns
            seq_len: Length of sequences for LSTM input
            symbols: List of stock symbols to include
        """
        self.seq_len = seq_len
        self.symbols = sorted(symbols)
        self.num_assets = len(symbols)

        # Ensure dates are datetime
        embeddings_df['Date'] = pd.to_datetime(embeddings_df['Date'])
        price_df['Date'] = pd.to_datetime(price_df['Date'])

        # Filter to only include the symbols we want
        embeddings_df = embeddings_df[embeddings_df['Symbol'].isin(self.symbols)].copy()
        price_df = price_df[price_df['Symbol'].isin(self.symbols)].copy()

        # Calculate log returns here instead of in a separate function
        logger.info("Calculating log returns...")
        price_df = price_df.sort_values(['Symbol', 'Date'])
        price_df['log_return'] = price_df.groupby('Symbol')['Adj Close'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # Fill NaN values with 0
        price_df['log_return'] = price_df['log_return'].fillna(0)

        # Log return statistics
        logger.info(f"Log return stats: mean={price_df['log_return'].mean():.6f}, "
                    f"std={price_df['log_return'].std():.6f}")
        logger.info(f"Zero values in log_return: {(price_df['log_return'] == 0).sum()}")
        logger.info(f"Total log_return values: {len(price_df['log_return'])}")

        # Extract embedding columns
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('dim_')]

        # Create date range for embeddings
        all_embed_dates = sorted(embeddings_df['Date'].unique())

        # Create pivot tables for embeddings
        logger.info("Creating pivot tables for embeddings...")
        embeddings_pivot = {}
        for col in embedding_cols:
            # Pivot to get dates as index, symbols as columns
            pivot = embeddings_df.pivot(index='Date', columns='Symbol', values=col)
            embeddings_pivot[col] = pivot

        # Create pivot for returns (will align with embeddings)
        logger.info("Creating pivot tables for returns...")
        returns_pivot = price_df.pivot(index='Date', columns='Symbol', values='log_return')

        # Shift returns for next-day prediction
        # Shift -1 means the target is the next day's return
        target_returns = returns_pivot.shift(-1)

        # Fill NaN values with 0 after shifting
        target_returns = target_returns.fillna(0)

        logger.info(f"Zero values in target returns: {(target_returns == 0).sum().sum()}")

        # Store dates from embeddings pivot
        self.dates = np.array(embeddings_pivot[embedding_cols[0]].index)

        # Convert embeddings to 3D array [dates, symbols, embedding_dim]
        embeddings_array = np.stack([
            embeddings_pivot[col][self.symbols].values
            for col in embedding_cols
        ], axis=-1)

        # Get returns array for these dates and symbols
        returns_array = target_returns[self.symbols].values

        # Create simplified mask (all True since we want to use all data)
        masks_array = np.ones_like(returns_array)

        # Debug - check array shapes
        logger.info(f"Embeddings array shape: {embeddings_array.shape}")
        logger.info(f"Returns array shape: {returns_array.shape}")

        # Create sequences for LSTM
        self.features = []
        self.targets = []
        self.masks = []

        for i in range(len(self.dates) - seq_len):
            # Get sequence of embeddings
            seq_embeddings = embeddings_array[i:i + seq_len]
            # Get next day's returns as target
            next_returns = returns_array[i + seq_len - 1]
            # Create mask (all 1s for simplicity)
            seq_mask = masks_array[i + seq_len - 1]

            self.features.append(seq_embeddings)
            self.targets.append(next_returns)
            self.masks.append(seq_mask)

        if not self.features:
            raise ValueError("No valid sequences found")

        # Convert to numpy arrays
        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        self.masks = np.array(self.masks)

        # Debug - final dataset stats
        logger.info(f"Created dataset with {len(self.features)} sequences")
        logger.info(f"Features shape: {self.features.shape}")
        logger.info(f"Targets shape: {self.targets.shape}")
        logger.info(f"Non-zero targets: {np.count_nonzero(self.targets)}")
        logger.info(f"Zero targets: {np.size(self.targets) - np.count_nonzero(self.targets)}")
        logger.info(
            f"Zero target percentage: {(np.size(self.targets) - np.count_nonzero(self.targets)) / np.size(self.targets) * 100:.2f}%")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # LSTM expects [num_assets, seq_len, features]
        features = self.features[idx].transpose(1, 0, 2)
        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(self.targets[idx], dtype=torch.float),
            torch.tensor(self.masks[idx], dtype=torch.float)
        )


# Separate verify function that doesn't assume log_return exists yet
def verify_log_returns_calculation(df, symbol=None):
    """
    Verify log returns calculation for debugging purposes.

    Args:
        df: DataFrame with Adj Close column
        symbol: Optional symbol to filter on
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    if symbol:
        df_copy = df_copy[df_copy['Symbol'] == symbol].copy()

    # Sort by date
    df_copy = df_copy.sort_values(['Symbol', 'Date'])

    # Calculate log returns
    df_copy['calc_log_return'] = df_copy.groupby('Symbol')['Adj Close'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Display original price and calculated returns
    print(f"Data for symbol {symbol if symbol else 'all symbols'}")
    print(df_copy[['Date', 'Symbol', 'Adj Close', 'calc_log_return']].head(10))

    # Check statistics
    print(f"\nCalculated log return stats:")
    print(f"Mean: {df_copy['calc_log_return'].mean():.6f}")
    print(f"Std: {df_copy['calc_log_return'].std():.6f}")
    print(f"Min: {df_copy['calc_log_return'].min():.6f}")
    print(f"Max: {df_copy['calc_log_return'].max():.6f}")
    print(f"Zero values: {(df_copy['calc_log_return'] == 0).sum()} out of {len(df_copy)}")
    print(f"NaN values: {df_copy['calc_log_return'].isna().sum()}")

    return df_copy

# Training Loop
def train_portfolio_lstm(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    best_val_sharpe = -float('inf')
    best_model_path = "best_portfolio_lstm.pth"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_embeddings, batch_returns, batch_masks in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_returns = batch_returns.to(device)
            batch_masks = batch_masks.to(device)

            optimizer.zero_grad()
            weights = model(batch_embeddings, batch_masks)
            loss = criterion(weights, batch_returns, batch_masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        model.eval()
        val_loss = 0.0
        val_sharpe = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_embeddings, batch_returns, batch_masks in val_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_returns = batch_returns.to(device)
                batch_masks = batch_masks.to(device)

                weights = model(batch_embeddings, batch_masks)
                loss = criterion(weights, batch_returns, batch_masks)

                val_loss += loss.item()
                portfolio_returns = torch.sum(batch_returns * weights, dim=1)
                mean_returns = torch.mean(portfolio_returns)
                std_returns = torch.std(portfolio_returns, unbiased=False)
                sharpe = (mean_returns / (std_returns + 1e-8)).item()
                val_sharpe += sharpe
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        avg_val_sharpe = val_sharpe / val_batches if val_batches > 0 else -float('inf')

        if avg_val_sharpe > best_val_sharpe:
            best_val_sharpe = avg_val_sharpe
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Sharpe: {avg_val_sharpe:.4f}")

    return best_model_path


def evaluate_portfolio(model, test_loader, criterion, device):
    model.eval()
    test_weights = []
    test_portfolio_returns = []

    with torch.no_grad():
        for batch_embeddings, batch_returns, batch_masks in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_returns = batch_returns.to(device)
            batch_masks = batch_masks.to(device)

            weights = model(batch_embeddings, batch_masks)
            portfolio_returns = torch.sum(weights * batch_returns, dim=1)

            test_weights.append(weights.cpu().numpy())
            test_portfolio_returns.append(portfolio_returns.cpu().numpy())

    test_weights = np.concatenate(test_weights, axis=0)
    test_portfolio_returns = np.concatenate(test_portfolio_returns, axis=0)

    mean_return = np.mean(test_portfolio_returns)
    std_return = np.std(test_portfolio_returns)
    sharpe_ratio = (mean_return / (std_return + 1e-8)) * np.sqrt(252)

    print(f"Test Sharpe Ratio: {sharpe_ratio:.4f}")
    print("Final Portfolio Weights (first 5 samples):")
    for i in range(min(5, test_weights.shape[0])):
        weights_str = ", ".join([f"{w:.4f}" for w in test_weights[i]])
        print(f"Sample {i + 1}: [{weights_str}]")
    print(f"Mean Weights per Asset: {', '.join([f'{w:.4f}' for w in np.mean(test_weights, axis=0)])}")

    return test_weights, test_portfolio_returns

def evaluate_train_weights(model, train_loader, device):
    model.eval()
    train_weights = []
    train_dates = []

    with torch.no_grad():
        for batch_embeddings, batch_returns, batch_masks in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_masks = batch_masks.to(device)
            weights = model(batch_embeddings, batch_masks)
            train_weights.append(weights.cpu().numpy())

    train_weights = np.concatenate(train_weights, axis=0)
    return train_weights


def main():
    embeddings_train = pd.read_csv("embeddings_train_after.csv")
    embeddings_val = pd.read_csv("embeddings_val_after.csv")
    embeddings_test = pd.read_csv("embeddings_test_after.csv")

    load_dotenv()
    DATA_PATH = os.getenv("DATA_PATH", "data")
    logger.info(f"Using DATA_PATH: {DATA_PATH}")

    # 2️⃣ Load Processed Data
    train_df = pd.read_pickle(os.path.join(DATA_PATH, "train.pkl"))
    val_df = pd.read_pickle(os.path.join(DATA_PATH, "val.pkl"))
    test_df = pd.read_pickle(os.path.join(DATA_PATH, "test.pkl"))

    # Debug step - check log returns calculation in train_df
    print("Checking a sample of log returns calculation:")
    sample_symbol = train_df['Symbol'].iloc[0]
    verify_df = verify_log_returns_calculation(train_df, sample_symbol)

    # Filter out VNINDEX
    embeddings_train = embeddings_train[embeddings_train['Symbol'] != 'VNINDEX']
    embeddings_val = embeddings_val[embeddings_val['Symbol'] != 'VNINDEX']
    embeddings_test = embeddings_test[embeddings_test['Symbol'] != 'VNINDEX']
    train_df = train_df[train_df['Symbol'] != 'VNINDEX']
    val_df = val_df[val_df['Symbol'] != 'VNINDEX']
    test_df = test_df[test_df['Symbol'] != 'VNINDEX']

    # Convert dates to datetime (if not already done)
    embeddings_train['Date'] = pd.to_datetime(embeddings_train['Date'])
    embeddings_val['Date'] = pd.to_datetime(embeddings_val['Date'])
    embeddings_test['Date'] = pd.to_datetime(embeddings_test['Date'])
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    # We don't need to pre-compute log returns here anymore
    # as the SimpleStockDataset will handle that internally

    symbols = sorted(embeddings_train['Symbol'].unique())
    num_assets = len(symbols)

    # Create datasets using the simplified SimpleStockDataset
    seq_len = 20
    input_dim = 4
    hidden_dim = 64
    num_layers = 2
    dropout = 0.5
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0005
    weight_decay = 1e-5

    train_dataset = SimpleStockDataset(embeddings_train, train_df, seq_len, symbols)
    val_dataset = SimpleStockDataset(embeddings_val, val_df, seq_len, symbols)
    test_dataset = SimpleStockDataset(embeddings_test, test_df, seq_len, symbols)

    print("Train Dataset:")
    print(f"Features Shape: {train_dataset.features.shape}, Zero Count: {(train_dataset.features == 0).sum()}")
    print(f"Targets Shape: {train_dataset.targets.shape}, Zero Count: {(train_dataset.targets == 0).sum()}")
    print(f"Masks Shape: {train_dataset.masks.shape}, Zero Count: {(train_dataset.masks == 0).sum()}")

    print("Validation Dataset:")
    print(f"Features Shape: {val_dataset.features.shape}, Zero Count: {(val_dataset.features == 0).sum()}")
    print(f"Targets Shape: {val_dataset.targets.shape}, Zero Count: {(val_dataset.targets == 0).sum()}")
    print(f"Masks Shape: {val_dataset.masks.shape}, Zero Count: {(val_dataset.masks == 0).sum()}")

    print("Test Dataset:")
    print(f"Features Shape: {test_dataset.features.shape}, Zero Count: {(test_dataset.features == 0).sum()}")
    print(f"Targets Shape: {test_dataset.targets.shape}, Zero Count: {(test_dataset.targets == 0).sum()}")
    print(f"Masks Shape: {test_dataset.masks.shape}, Zero Count: {(test_dataset.masks == 0).sum()}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader_no_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # For weights in order

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PortfolioLSTM(input_dim, hidden_dim, num_layers, num_assets, dropout).to(device)
    criterion = SharpeRatioLoss(risk_free_rate=0.0, annualization_factor=252).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Starting training...")
    best_model_path = train_portfolio_lstm(train_loader, val_loader, model, criterion, optimizer, num_epochs, device)

    print("Evaluating on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_weights, test_portfolio_returns = evaluate_portfolio(model, test_loader, criterion, device)

    # Print weights allocation over train dataset
    print("Evaluating weights on train dataset...")
    train_weights = evaluate_train_weights(model, train_loader_no_shuffle, device)
    train_dates = train_dataset.dates[seq_len:]
    print("Train Weights Allocation (all samples):")
    for i, (date, weights) in enumerate(zip(train_dates, train_weights)):
        weights_str = ", ".join([f"{w:.4f}" for w in weights])
        converted_date = pd.to_datetime(date)
        print(f"Date {converted_date.strftime('%Y-%m-%d')}: [{weights_str}]")
    print(f"Mean Train Weights per Asset: {', '.join([f'{w:.4f}' for w in np.mean(train_weights, axis=0)])}")

    # Save results
    test_dates = test_dataset.dates[seq_len:]
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Portfolio_Return': test_portfolio_returns
    })
    weights_df = pd.DataFrame(test_weights, columns=symbols, index=test_dates)

    results_df.to_csv("portfolio_returns.csv", index=False)
    weights_df.to_csv("portfolio_weights.csv")

    print("Results saved to 'portfolio_returns.csv' and 'portfolio_weights.csv'")


if __name__ == "__main__":
    main()
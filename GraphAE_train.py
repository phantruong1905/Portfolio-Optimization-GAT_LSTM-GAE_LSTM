import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from GraphAE import prepare_gnn_lstm_data, create_corr_edges_with_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#-----------------------------------------------------------------------------------------------------------------------#

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch
import torch.optim as optim
from torch_geometric.nn import GAE
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit

from utils.loss import SharpeRatioLoss, LogReturnLoss
from utils.graph_dataset import GNNLSTMDataset
from torch.utils.data import Dataset, DataLoader

from models.GAT_LSTM import GraphAutoencoder

from src.data_loader import fetch_stock_data
from src.feature_engineering import generate_features




load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data")
stock_data_path = os.path.join(DATA_PATH, "features.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Fetching stock data...")
fetch_stock_data(data_path=DATA_PATH, stocks=['VNINDEX', 'VCI', 'HCM', 'CTG', 'HPG', 'STB', 'VCB', 'SZC', 'NLG', 'HDC', 'SIP', 'PHR', 'DRI', 'DPR', 'DCM', 'DPM'],
                       start_date="2020-01-10", end_date="2025-03-20")
#
generate_features(data_path=DATA_PATH)
# Updated create_window_graph to handle empty graphs
def create_window_graph(df, symbols, feature_cols, corr_threshold, is_train=True):
    # Compute node features for this window
    n_symbols = len(symbols)
    x_stocks = torch.zeros((n_symbols, len(feature_cols)), dtype=torch.float)
    symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}

    # Log data availability per stock
    for sym in symbols:
        stock_data = df[df['Symbol'] == sym]
        logger.info(f"Window data for {sym}: {len(stock_data)} rows")

    for i, sym in enumerate(symbols):
        stock_data = df[df['Symbol'] == sym][feature_cols]
        if not stock_data.empty:
            x_stocks[i] = torch.tensor(stock_data.mean().values, dtype=torch.float)
        else:
            x_stocks[i] = torch.zeros(len(feature_cols), dtype=torch.float)

    # Log zero counts in node features before imputation
    zero_count = (x_stocks == 0).sum().item()
    total_count = x_stocks.numel()

    # Impute zeros in node features
    for col in range(x_stocks.shape[1]):
        non_zero_mask = x_stocks[:, col] != 0
        if non_zero_mask.sum() > 0:
            mean_non_zero = x_stocks[non_zero_mask, col].mean()
            x_stocks[~non_zero_mask, col] = mean_non_zero
        else:
            x_stocks[:, col] = torch.where(x_stocks[:, col] == 0, torch.tensor(1e-6), x_stocks[:, col])

    # Log zero counts after imputation
    zero_count = (x_stocks == 0).sum().item()
    logger.info(f"Node features (x) after imputation: {zero_count} zeros out of {total_count} values ({zero_count / total_count:.2%})")

    # Create graph for this window
    edge_index, edge_attr, _ = create_corr_edges_with_weights(df, threshold=corr_threshold)

    # If no edges, create a fully connected graph (excluding VNINDEX-to-VNINDEX)
    if edge_index.size(1) == 0:
        logger.warning("No edges in this window's graph. Creating a fully connected graph.")
        edge_index = []
        edge_attr = []
        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                if symbols[i] == "VNINDEX" and symbols[j] == "VNINDEX":
                    continue
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append(1.0)
                edge_attr.append(1.0)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

    # Log the number of edges
    num_edges = edge_index.size(1) // 2
    logger.info(f"Number of edges: {num_edges}")

    # Create PyTorch Geometric Data object
    data = Data(x=x_stocks, edge_index=edge_index, edge_attr=edge_attr)

    if is_train:
        # Split edges for training the GAE within this window
        transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False)
        train_data, val_data, test_data = transform(data)
        return train_data, val_data, test_data, x_stocks, symbol_to_idx
    else:
        # Return the full graph for inference (val/test sets)
        return data, x_stocks, symbol_to_idx

def train_gae(model, train_data, val_data, epochs, lr, patience=20, device="cuda"):
    model = model.to(device)

    # Log zero counts in data.x (which now comes from embeddings_before_*.csv)
    for data, set_name in [(train_data, "train"), (val_data, "val")]:
        x_np = data.x.cpu().numpy()
        zero_count = (x_np == 0).sum()
        total_count = x_np.size
        print(
            f"{set_name} data.x (from embeddings): {zero_count} zeros out of {total_count} values ({zero_count / total_count:.2%})")

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Encode training data
        z = model.encode(train_data.x, train_data.edge_index, train_data.edge_attr)

        # Reconstruction loss
        pred_weights = model.decode(z, train_data.edge_index)
        loss = loss_fn(pred_weights, train_data.edge_attr)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # Log training loss and zero counts in embeddings
            z_np = z.cpu().detach().numpy()
            zero_count = (z_np == 0).sum()
            total_count = z_np.size
            print(f"Epoch {epoch}: Train Loss: {loss.item():.4f}")

            # Validation
            model.eval()
            with torch.no_grad():
                z_val = model.encode(val_data.x, val_data.edge_index, val_data.edge_attr)
                val_pred_weights = model.decode(z_val, val_data.edge_index)
                val_loss = loss_fn(val_pred_weights, val_data.edge_attr)

                # Log validation loss and zero counts
                z_val_np = z_val.cpu().numpy()
                zero_count_val = (z_val_np == 0).sum()
                total_count_val = z_val_np.size
                print(f"Epoch {epoch}: Val Loss: {val_loss.item():.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model = model.state_dict()
                else:
                    epochs_no_improve += 10
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        model.load_state_dict(best_model)
                        break

        model.train()

    # Load the best model
    model.load_state_dict(best_model)

    # Get final embeddings for training data
    final_z = get_node_embeddings(model, train_data, device)
    return model, final_z


def get_node_embeddings(model, data, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    return z


from models.GAT_LSTM import GraphAutoencoder, GAE_Encoder


# ✅ Setup Logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 1️⃣ Load Environment Variables
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data")
logger.info(f"Using DATA_PATH: {DATA_PATH}")

# 2️⃣ Load Processed Data
train_df = pd.read_pickle(os.path.join(DATA_PATH, "train.pkl"))
val_df = pd.read_pickle(os.path.join(DATA_PATH, "val.pkl"))
test_df = pd.read_pickle(os.path.join(DATA_PATH, "test.pkl"))
logger.info(f"Loaded train_df: {train_df.shape}, val_df: {val_df.shape}, test_df: {test_df.shape}")



# Prepare data for GNN + LSTM
seq_len = 20
corr_threshold = 0.5  # Lowered to avoid empty graphs

train_dataset, val_dataset, test_dataset, feature_cols, scaler = prepare_gnn_lstm_data(
    train_df, val_df, test_df
)

corr_threshold = 0.5
embedding_dim = 4
hidden_dim = 16
in_dim = len(feature_cols)

# Ensure Date is in datetime format
train_dataset['Date'] = pd.to_datetime(train_dataset['Date'])
val_dataset['Date'] = pd.to_datetime(val_dataset['Date'])
test_dataset['Date'] = pd.to_datetime(test_dataset['Date'])

# Filter train_dataset to include only data up to 2022-02
train_end_date = pd.to_datetime('2022-02-28')
train_dataset = train_dataset[train_dataset['Date'] <= train_end_date]

# Add YearMonth column for monthly grouping
train_dataset['YearMonth'] = train_dataset['Date'].dt.to_period('M')
val_dataset['YearMonth'] = val_dataset['Date'].dt.to_period('M')
test_dataset['YearMonth'] = test_dataset['Date'].dt.to_period('M')

# Get unique months and symbols from training data
train_months = sorted(train_dataset['YearMonth'].unique())
symbols = sorted(train_dataset['Symbol'].unique())
n_symbols = len(symbols)
symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}


# Compute edges for each month using only the training data
monthly_edges = {}
logger.info("Computing edges for each month based on log return correlations (using training data only)...")
for month in train_months:
    month_df = train_dataset[train_dataset['YearMonth'] == month]
    edge_index, edge_attr = create_corr_edges_with_weights(
        month_df, symbols, threshold=corr_threshold, price_col='Adj Close'
    )
    monthly_edges[month] = (edge_index, edge_attr)
    num_edges = edge_index.size(1) // 2 if edge_index.numel() > 0 else 0
    logger.info(f"Month {month}: Created {num_edges} edges")

# Get the last training month to use its edges and GAE for val and test
last_train_month = train_months[-1]
last_train_edge_index, last_train_edge_attr = monthly_edges[last_train_month]
logger.info(f"Last training month: {last_train_month}, will use its edges and GAE for val and test periods")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


# Function to create daily graph data
def create_daily_graph(df, symbols, feature_cols, edge_index, edge_attr, date):
    """
    Create a graph for a specific day with the given edge structure.

    Args:
        df (pd.DataFrame): DataFrame with daily data
        symbols (list): List of stock symbols
        feature_cols (list): List of feature columns
        edge_index (torch.Tensor): Edge indices for the month
        edge_attr (torch.Tensor): Edge weights for the month
        date (pd.Timestamp): Date for which to create the graph

    Returns:
        data (Data): PyTorch Geometric Data object
        x_stocks (torch.Tensor): Node features
    """
    x_stocks = torch.zeros((len(symbols), len(feature_cols)), dtype=torch.float)

    # Filter data for the specific date
    day_df = df[df['Date'] == date]

    # Log data availability
    for sym in symbols:
        stock_data = day_df[day_df['Symbol'] == sym]
        logger.debug(f"Data for {sym} on {date}: {len(stock_data)} rows")

    # Compute node features for this day
    for i, sym in enumerate(symbols):
        stock_data = day_df[day_df['Symbol'] == sym][feature_cols]
        if not stock_data.empty:
            x_stocks[i] = torch.tensor(stock_data.iloc[0].values, dtype=torch.float)
        else:
            x_stocks[i] = torch.zeros(len(feature_cols), dtype=torch.float)

    # Impute zeros in node features with added noise
    zero_count = (x_stocks == 0).sum().item()
    total_count = x_stocks.numel()
    logger.debug(
        f"Node features (x) before imputation on {date}: {zero_count} zeros out of {total_count} values ({zero_count / total_count:.2%})")

    for col in range(x_stocks.shape[1]):
        non_zero_mask = x_stocks[:, col] != 0
        if non_zero_mask.sum() > 0:
            mean_non_zero = x_stocks[non_zero_mask, col].mean()
            noise = torch.randn(sum(~non_zero_mask)) * 0.01
            x_stocks[~non_zero_mask, col] = mean_non_zero + noise
        else:
            x_stocks[:, col] = torch.where(x_stocks[:, col] == 0, torch.tensor(1e-6), x_stocks[:, col])

    zero_count = (x_stocks == 0).sum().item()
    logger.debug(
        f"Node features (x) after imputation on {date}: {zero_count} zeros out of {total_count} values ({zero_count / total_count:.2%})")

    # Create PyTorch Geometric Data object
    data = Data(x=x_stocks, edge_index=edge_index, edge_attr=edge_attr)
    return data, x_stocks


# Train a separate GAE for each month in the training period
monthly_gaes = {}
for month in train_months:
    logger.info(f"Training GAE for month {month}...")
    # Get the dates for this month
    month_dates = sorted(train_dataset[train_dataset['YearMonth'] == month]['Date'].unique())
    if not month_dates:
        logger.warning(f"No data for month {month}. Skipping GAE training.")
        continue

    # Prepare daily graphs for this month
    month_data_list = []
    for date in month_dates:
        edge_index, edge_attr = monthly_edges[month]
        data, x_stocks = create_daily_graph(train_dataset, symbols, feature_cols, edge_index, edge_attr, date)
        month_data_list.append(data)

    # Batch the daily graphs for this month
    if not month_data_list:
        logger.warning(f"No graphs created for month {month}. Skipping GAE training.")
        continue
    month_batch = Batch.from_data_list(month_data_list)
    month_batch = month_batch.to(device)

    # Split edges for training the GAE
    transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False)
    train_data, val_data, test_data = transform(month_batch)

    # Train the GAE for this month
    model = GraphAutoencoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=embedding_dim).to(device)
    trained_gae, _ = train_gae(model, train_data, val_data, epochs=200, lr=0.0005, patience=20, device=device)
    monthly_gaes[month] = trained_gae

# Ensure we have a GAE for the last training month
if last_train_month not in monthly_gaes:
    raise ValueError(f"No GAE trained for the last training month {last_train_month}. Cannot proceed with inference.")

# Generate daily embeddings for each dataset separately
embedding_history_train_before = []
embedding_history_train_after = []
embedding_dates_train = []

embedding_history_val_before = []
embedding_history_val_after = []
embedding_dates_val = []

embedding_history_test_before = []
embedding_history_test_after = []
embedding_dates_test = []


# Function to generate embeddings for a given dataset
def generate_embeddings(dataset, dates, monthly_gaes, monthly_edges, last_train_month, last_train_edge_index,
                        last_train_edge_attr, symbols, feature_cols, device):
    embedding_history_before = []
    embedding_history_after = []
    embedding_dates = []

    logger.info(f"Generating embeddings for dataset with {len(dates)} dates...")
    for date in dates:
        year_month = pd.to_datetime(date).to_period('M')
        # Determine which GAE and edges to use
        if year_month in monthly_gaes:
            gae_model = monthly_gaes[year_month]
            edge_index, edge_attr = monthly_edges[year_month]
        else:
            gae_model = monthly_gaes[last_train_month]
            edge_index, edge_attr = last_train_edge_index, last_train_edge_attr
            logger.debug(f"Using last training month's GAE and edges for date {date} (month {year_month})")

        data, x_stocks = create_daily_graph(dataset, symbols, feature_cols, edge_index, edge_attr, date)
        data = data.to(device)

        z_before = x_stocks
        z_after = get_node_embeddings(gae_model, data, device=device)
        z_before_np = z_before.cpu().detach().numpy()
        z_after_np = z_after.cpu().detach().numpy()

        embedding_history_before.append(z_before_np)
        embedding_history_after.append(z_after_np)
        embedding_dates.append(date)

    return embedding_history_before, embedding_history_after, embedding_dates


# Generate embeddings for train, val, and test datasets
train_dates = sorted(train_dataset['Date'].unique())
val_dates = sorted(val_dataset['Date'].unique())
test_dates = sorted(test_dataset['Date'].unique())

# Train embeddings
embedding_history_train_before, embedding_history_train_after, embedding_dates_train = generate_embeddings(
    train_dataset, train_dates, monthly_gaes, monthly_edges, last_train_month, last_train_edge_index,
    last_train_edge_attr, symbols, feature_cols, device
)

# Val embeddings
embedding_history_val_before, embedding_history_val_after, embedding_dates_val = generate_embeddings(
    val_dataset, val_dates, monthly_gaes, monthly_edges, last_train_month, last_train_edge_index, last_train_edge_attr,
    symbols, feature_cols, device
)

# Test embeddings
embedding_history_test_before, embedding_history_test_after, embedding_dates_test = generate_embeddings(
    test_dataset, test_dates, monthly_gaes, monthly_edges, last_train_month, last_train_edge_index,
    last_train_edge_attr, symbols, feature_cols, device
)

# Convert embeddings to arrays
embeddings_train_before = np.stack(embedding_history_train_before)  # Shape: [n_train_days, n_symbols, in_dim]
embeddings_train_after = np.stack(embedding_history_train_after)  # Shape: [n_train_days, n_symbols, out_dim]
embedding_dates_train = np.array(embedding_dates_train)

embeddings_val_before = np.stack(embedding_history_val_before)  # Shape: [n_val_days, n_symbols, in_dim]
embeddings_val_after = np.stack(embedding_history_val_after)  # Shape: [n_val_days, n_symbols, out_dim]
embedding_dates_val = np.array(embedding_dates_val)

embeddings_test_before = np.stack(embedding_history_test_before)  # Shape: [n_test_days, n_symbols, in_dim]
embeddings_test_after = np.stack(embedding_history_test_after)  # Shape: [n_test_days, n_symbols, out_dim]
embedding_dates_test = np.array(embedding_dates_test)


# Function to create DataFrame from embeddings
def create_embedding_df(embeddings, dates, dim_count, symbols):
    embedding_dfs = []
    for t, date in enumerate(dates):
        df_t = pd.DataFrame(
            embeddings[t],
            columns=[f"dim_{i}" for i in range(dim_count)],
            index=symbols
        )
        df_t['Date'] = date
        embedding_dfs.append(df_t.reset_index().rename(columns={'index': 'Symbol'}))
    return pd.concat(embedding_dfs, ignore_index=True)


# Create DataFrames and save for each split
df_train_before = create_embedding_df(embeddings_train_before, embedding_dates_train, embeddings_train_before.shape[2],
                                      symbols)
df_train_after = create_embedding_df(embeddings_train_after, embedding_dates_train, embeddings_train_after.shape[2],
                                     symbols)
df_val_before = create_embedding_df(embeddings_val_before, embedding_dates_val, embeddings_val_before.shape[2], symbols)
df_val_after = create_embedding_df(embeddings_val_after, embedding_dates_val, embeddings_val_after.shape[2], symbols)
df_test_before = create_embedding_df(embeddings_test_before, embedding_dates_test, embeddings_test_before.shape[2],
                                     symbols)
df_test_after = create_embedding_df(embeddings_test_after, embedding_dates_test, embeddings_test_after.shape[2],
                                    symbols)

# Save to separate CSV files
df_train_before.to_csv("embeddings_train_before.csv", index=False)
df_train_after.to_csv("embeddings_train_after.csv", index=False)
df_val_before.to_csv("embeddings_val_before.csv", index=False)
df_val_after.to_csv("embeddings_val_after.csv", index=False)
df_test_before.to_csv("embeddings_test_before.csv", index=False)
df_test_after.to_csv("embeddings_test_after.csv", index=False)

# PLOTTING
# Define date ranges for train, val, and test
train_start = pd.to_datetime('2014-03-11')
train_end = pd.to_datetime('2022-02-28')
val_start = pd.to_datetime('2022-03-01')
val_end = pd.to_datetime('2024-01-25')
test_start = pd.to_datetime('2024-01-26')
test_end = pd.to_datetime('2025-03-31')

# Select a stock to plot
stock_name = symbols[0]  # e.g., 'CTG'

# Filter data for the selected stock from each dataset
df_stock_train = train_dataset[train_dataset['Symbol'] == stock_name].copy()
df_stock_val = val_dataset[val_dataset['Symbol'] == stock_name].copy()
df_stock_test = test_dataset[test_dataset['Symbol'] == stock_name].copy()

df_train_before_stock = df_train_before[df_train_before['Symbol'] == stock_name].copy()
df_train_after_stock = df_train_after[df_train_after['Symbol'] == stock_name].copy()
df_val_before_stock = df_val_before[df_val_before['Symbol'] == stock_name].copy()
df_val_after_stock = df_val_after[df_val_after['Symbol'] == stock_name].copy()
df_test_before_stock = df_test_before[df_test_before['Symbol'] == stock_name].copy()
df_test_after_stock = df_test_after[df_test_after['Symbol'] == stock_name].copy()

# Create just 3 subplots - train, val, test data
fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

# Plot Train data
n_dims_to_plot = min(6, embedding_dim)
for i in range(n_dims_to_plot):
    axes[0].plot(df_train_after_stock["Date"], df_train_after_stock[f"dim_{i}"],
                 label=f"Dim {i+1}", alpha=0.7)
axes[0].set_title(f"Train Period Embeddings: {stock_name}")
axes[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
axes[0].set_xlim(train_start, test_end)

# Plot Validation data
for i in range(n_dims_to_plot):
    axes[1].plot(df_val_after_stock["Date"], df_val_after_stock[f"dim_{i}"],
                 label=f"Dim {i+1}", alpha=0.7)
axes[1].set_title(f"Validation Period Embeddings: {stock_name}")
axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)

# Plot Test data
for i in range(n_dims_to_plot):
    axes[2].plot(df_test_after_stock["Date"], df_test_after_stock[f"dim_{i}"],
                 label=f"Dim {i+1}", alpha=0.7)
axes[2].set_title(f"Test Period Embeddings: {stock_name}")
axes[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)

plt.tight_layout()
plt.show()
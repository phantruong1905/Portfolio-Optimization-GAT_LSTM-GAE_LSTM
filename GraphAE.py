import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def compute_log_return_split(train_df, val_df, test_df):
    """
    Compute log returns separately for each dataset to avoid data leakage.

    Args:
        train_df, val_df, test_df: DataFrames to process

    Returns:
        DataFrames with added log_return column
    """
    def compute_log_return(df):
        df = df.sort_values(['Symbol', 'Date']).copy()
        df['log_return'] = df.groupby('Symbol')['Adj Close'].transform(lambda x: np.log(x / x.shift(1)))
        return df.dropna(subset=['log_return'])  # Drop NaN values which occur due to shift

    train_df = compute_log_return(train_df)
    val_df = compute_log_return(val_df)
    test_df = compute_log_return(test_df)

    return train_df, val_df, test_df





def create_corr_edges_with_weights(df, symbols, threshold=0.3, price_col='Adj Close'):
    """
    Create edges based on the correlation of log returns within the given DataFrame (e.g., a month's data).

    Args:
        df (pd.DataFrame): DataFrame with columns ['Date', 'Symbol', price_col, ...]
        symbols (list): List of stock symbols
        threshold (float): Correlation threshold for creating edges
        price_col (str): Column name for the price data (default: 'Adj Close')

    Returns:
        edge_index (torch.Tensor): Edge indices [2, num_edges]
        edge_attr (torch.Tensor): Edge weights [num_edges, 1]
    """
    # Ensure Date is in datetime format
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Remove duplicates based on Symbol and Date
    df = df.drop_duplicates(subset=['Symbol', 'Date'], keep='last')

    # Ensure Adj Close is numeric
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

    # Sort by Symbol and Date to ensure correct shift
    df = df.sort_values(['Symbol', 'Date'])

    # Compute log returns using transform to preserve the index
    df['Log Return'] = df.groupby('Symbol')[price_col].transform(lambda x: np.log(x / x.shift(1)))

    # Drop NaN values (first row of each stock will have NaN log return)
    df = df.dropna(subset=['Log Return'])

    # Get the symbols present in this month's data
    available_symbols = sorted(df['Symbol'].unique())

    # If there’s not enough data to compute correlations, create a fully connected graph for available symbols
    if len(available_symbols) < 2 or df['Log Return'].std() < 1e-6:
        edge_index = []
        edge_attr = []
        symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i >= j:
                    continue
                if sym1 not in available_symbols or sym2 not in available_symbols:
                    continue
                if sym1 == 'VNINDEX' and sym2 == 'VNINDEX':
                    continue
                edge_index.append([symbol_to_idx[sym1], symbol_to_idx[sym2]])
                edge_index.append([symbol_to_idx[sym2], symbol_to_idx[sym1]])
                edge_attr.append(1.0)
                edge_attr.append(1.0)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        return edge_index, edge_attr

    # Pivot to get a matrix of daily log returns [n_days, n_symbols]
    pivot = df.pivot(index='Date', columns='Symbol', values='Log Return')

    # Reindex to include all symbols, filling missing ones with 0
    pivot = pivot.reindex(columns=symbols, fill_value=0)

    # Compute correlation matrix
    corr_matrix = pivot.corr()
    corr_matrix_np = corr_matrix.to_numpy()

    # Create edges based on correlation threshold
    edge_index = []
    edge_attr = []
    symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}

    for i, sym1 in enumerate(symbols):
        for j, sym2 in enumerate(symbols):
            if i >= j:  # Avoid duplicate pairs
                continue
            if sym1 not in available_symbols or sym2 not in available_symbols:
                continue
            if sym2 == 'VNINDEX':  # Exclude VNINDEX as a destination
                continue
            corr = corr_matrix_np[i, j]
            if abs(corr) >= threshold:
                edge_index.append([symbol_to_idx[sym1], symbol_to_idx[sym2]])
                edge_index.append([symbol_to_idx[sym2], symbol_to_idx[sym1]])
                edge_attr.append(corr)
                edge_attr.append(corr)

    # If no edges were created, create a fully connected graph (excluding VNINDEX as a destination)
    if not edge_index:
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i >= j:
                    continue
                if sym1 not in available_symbols or sym2 not in available_symbols:
                    continue
                if sym2 == 'VNINDEX':  # Exclude VNINDEX as a destination
                    continue
                edge_index.append([symbol_to_idx[sym1], symbol_to_idx[sym2]])
                edge_index.append([symbol_to_idx[sym2], symbol_to_idx[sym1]])
                edge_attr.append(1.0)
                edge_attr.append(1.0)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

    return edge_index, edge_attr


def normalize_splits(train_df, val_df, test_df):
    """
    Normalize only 'log_return' while keeping other features unchanged.
    """
    normalized_col = ['log_return', "volatility_atr", "trend_macd", "trend_adx", "trend_sma_fast", "momentum_rsi", 'Open', 'High', 'Low',
       'Adj Close', 'Volume']  # Only normalize log_return
    feature_cols = normalized_col  # All features

    # Apply StandardScaler ONLY to log_return
    scaler = StandardScaler()
    train_df[normalized_col] = scaler.fit_transform(train_df[normalized_col])
    val_df[normalized_col] = scaler.transform(val_df[normalized_col])
    test_df[normalized_col] = scaler.transform(test_df[normalized_col])

    return train_df, val_df, test_df, feature_cols, scaler  # Ensure feature_cols contains all 7 features



def prepare_gnn_lstm_data(train_df, val_df, test_df):
    print(f"Data loaded: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)} rows")
    train_df, val_df, test_df = compute_log_return_split(train_df, val_df, test_df)
    train_df, val_df, test_df, feature_cols, scaler = normalize_splits(train_df, val_df, test_df)
    print(f"Features normalized: {len(feature_cols)} features")
    return train_df, val_df, test_df, feature_cols, scaler




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

from utils.loss import SharpeRatioLoss, LogReturnLoss
from utils.graph_dataset import GNNLSTMDataset
from torch.utils.data import Dataset, DataLoader

from models.GAT_LSTM import GraphAutoencoder

from src.data_loader import fetch_stock_data
from src.feature_engineering import generate_features
from torch_geometric.transforms import RandomLinkSplit


def create_train_graph(train_dataset, edge_index, edge_attr, feature_cols):
    train_data = Data(
        x=torch.tensor(train_dataset[feature_cols].values, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    transform = RandomLinkSplit(
        num_val=0.05, num_test=0.1,  # Adjust train/val/test split as needed
        is_undirected=True,
        add_negative_train_samples=False
    )

    train_data, val_data, test_data = transform(train_data)  # ✅ Now includes `train_pos_edge_index`

    return train_data, val_data, test_data


def train_gae(model, train_data, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # Binary classification loss for edge prediction

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 🔥 Get embeddings (z) and predict adjacency matrix
        z = model.encoder(train_data.x, train_data.edge_index)  # Use `edge_index`
        adj_pred = model.decode(z, train_data.edge_label_index)  # Use `edge_label_index`

        # Create ground truth adjacency (1 for existing edges, 0 for others)
        target_adj = torch.ones_like(adj_pred)

        loss = loss_fn(adj_pred, target_adj)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {loss.item():.4f}")

    return model


def get_node_embeddings(model, train_data):
    """Extracts node embeddings from the trained GAE model."""
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.train_pos_edge_index)
    return z  # Final embeddings of stocks







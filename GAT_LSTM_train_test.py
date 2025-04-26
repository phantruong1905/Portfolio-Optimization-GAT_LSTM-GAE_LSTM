import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


# --------------------------
# 🚀 Fix Date-Based Splitting
# --------------------------
def split_by_date(df, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataframe into train, validation, and test sets based on chronological order.

    Args:
        df: DataFrame with 'Date' column
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation

    Returns:
        train_df, val_df, test_df: Chronologically split DataFrames
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by Date to ensure chronological order
    df = df.sort_values('Date')

    # Compute split indices based on row count, not date range
    total_rows = len(df)
    train_size = int(total_rows * train_ratio)
    val_size = int(total_rows * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    print(f"✅ Train: {len(train_df)} rows ({train_df['Date'].min()} → {train_df['Date'].max()})")
    print(f"✅ Val: {len(val_df)} rows ({val_df['Date'].min()} → {val_df['Date'].max()})")
    print(f"✅ Test: {len(test_df)} rows ({test_df['Date'].min()} → {test_df['Date'].max()})")

    return train_df, val_df, test_df


# --------------------------
# 🛠 Compute Log Returns Correctly
# --------------------------
def compute_log_return_split(train_df, val_df, test_df):
    """
    Compute log returns while maintaining split integrity.

    Args:
        train_df, val_df, test_df: DataFrames to process

    Returns:
        DataFrames with added log_return column
    """
    full_df = pd.concat([train_df, val_df, test_df]).sort_values(['Symbol', 'Date'])

    # Compute log returns before filtering
    full_df['log_return'] = full_df.groupby('Symbol')['Adj Close'].transform(lambda x: np.log(x / x.shift(1)))

    # Drop rows with NaNs (first row of each symbol)
    full_df = full_df.dropna(subset=['log_return'])

    # Re-split based on the original date ranges
    train_df = full_df[full_df['Date'].isin(train_df['Date'])]
    val_df = full_df[full_df['Date'].isin(val_df['Date'])]
    test_df = full_df[full_df['Date'].isin(test_df['Date'])]

    return train_df, val_df, test_df


# --------------------------
# 🔧 Fix Edge Creation Function
# --------------------------
def create_corr_edges_with_weights(df, threshold=0.3):
    df = df.reset_index(drop=True)
    price_df = df.pivot_table(index="Date", columns="Symbol", values="Adj Close")
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    corr_matrix = log_returns.corr()

    edge_list = []
    edge_weight_list = []

    symbols = corr_matrix.columns.tolist()
    symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}

    # Ensure VNINDEX exists in the dataset
    vnindex_idx = symbol_to_idx.get("VNINDEX", None)

    for i, sym_i in enumerate(symbols):
        if sym_i == "VNINDEX":
            continue
        for j, sym_j in enumerate(symbols):
            if sym_j == "VNINDEX" or i >= j:
                continue
            corr_value = corr_matrix.loc[sym_i, sym_j]
            if abs(corr_value) >= threshold:
                edge_list.append((i, j))
                edge_list.append((j, i))
                edge_weight_list.append(corr_value)
                edge_weight_list.append(corr_value)

    # VNINDEX → each stock (only if VNINDEX exists)
    if vnindex_idx is not None:
        for sym in symbols:
            if sym == "VNINDEX":
                continue
            j = symbol_to_idx[sym]
            corr_value = corr_matrix.loc["VNINDEX", sym]
            if abs(corr_value) >= threshold:
                edge_list.append((vnindex_idx, j))
                edge_weight_list.append(corr_value)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weight_list, dtype=torch.float).view(-1, 1)

    return edge_index, edge_attr, symbols


# --------------------------
# 🚀 Normalize Train, Val, Test Separately
# --------------------------
def normalize_splits(train_df, val_df, test_df):
    """
    Normalize features across train, validation, and test sets.

    Returns:
        Normalized DataFrames, feature column names, and fitted scaler
    """
    exclude_cols = ['Symbol', 'Date', 'Adj Close']

    feature_cols = [col for col in train_df.columns if col.startswith("LSTM_AE_feat") or col == "log_return"]

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, val_df, test_df, feature_cols


# --------------------------
# 🔥 Final Data Preparation Pipeline
# --------------------------
def prepare_gnn_lstm_data(data_path, seq_len=10, corr_threshold=0.3):
    """
    Complete pipeline to prepare data for GAT-LSTM training.

    Args:
        data_path: Path to raw data pickle file
        seq_len: LSTM sequence length
        corr_threshold: Correlation threshold for edge creation

    Returns:
        Processed datasets and metadata
    """
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_pickle(data_path)

    print(f"✅ Data Loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")

    # Split Data by Date
    train_df, val_df, test_df = split_by_date(df)

    # Compute log returns (before normalization)
    train_df, val_df, test_df = compute_log_return_split(train_df, val_df, test_df)

    # Create correlation-based edges
    edge_index, edge_attr, symbols = create_corr_edges_with_weights(train_df, threshold=corr_threshold)

    # Normalize feature columns
    train_df, val_df, test_df, feature_cols = normalize_splits(train_df, val_df, test_df)

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df, val_df, test_df, edge_index, edge_attr, feature_cols, seq_len
    )

    print(f"✅ Datasets: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'feature_cols': feature_cols,
        'symbols': symbols,
        'num_features': len(feature_cols),
        'num_assets': len(symbols)
    }

#-----------------------------------------------------------------------------------------------------------------------#

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch

from utils.loss import SharpeRatioLoss, LogReturnLoss
from utils.graph_dataset import GNNLSTMDataset
from models.GAT_LSTM import GAT_Encoder, LSTM_Allocator, GAT_LSTM, MLP_Allocator

from src.data_loader import fetch_stock_data
from src.feature_engineering import generate_features


def collate_fn(batch):
    from torch_geometric.data import Batch

    data_seq, target_returns = zip(*batch)  # Unzip batch into graphs & targets

    # 🛠 Ensure `data_seq` is a list, not a tuple of lists
    if isinstance(data_seq[0], list):
        data_seq = list(data_seq[0])  # Extract the list inside the tuple

    # 🔄 Batch graphs properly
    batched_graph = Batch.from_data_list(data_seq)

    print(f"Type: {type(target_returns)}, Length: {len(target_returns)}, Content: {target_returns}")

    # Convert target returns to a tensor
    target_returns = torch.stack(target_returns, dim=0).unsqueeze(-1)





    return batched_graph, target_returns


def train_gat_lstm(
        config,
        train_df,
        val_df,
        test_df,
        edge_index,
        edge_weight,
        feature_cols,
        symbols,
        seq_len,
        batch_size,
        num_epochs,
        learning_rate,
        weight_decay,
        patience,
        early_stopping_patience,
        model_save_path="checkpoint/gat_lstm_model.pt"
):
    """
    Train the GAT-LSTM portfolio optimization model with early stopping and model checkpointing

    Args:
        train_df, val_df, test_df: Data splits
        edge_index: Tensor of shape [2, num_edges]
        edge_weight: Tensor of shape [num_edges, 1]
        feature_cols: List of feature column names
        symbols: List of stock symbols
        seq_len: Sequence length for LSTM
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs to train
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization strength
        patience: Patience for learning rate scheduler
        early_stopping_patience: Patience for early stopping
        model_save_path: Path to save the best model

    Returns:
        Best model and training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()  # Get the current GPU index (usually 0)
        torch.cuda.set_per_process_memory_fraction(0.8, device=gpu_index)
    print(f"Using device: {device}")

    # Ensure directory for model saving exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Build datasets
    train_dataset = GNNLSTMDataset(train_df, seq_len, edge_index, edge_weight, feature_cols)
    val_dataset = GNNLSTMDataset(val_df, seq_len, edge_index, edge_weight, feature_cols)
    test_dataset = GNNLSTMDataset(test_df, seq_len, edge_index, edge_weight, feature_cols)

    print(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=False)

    # Model setup
    input_dim = len(feature_cols)
    num_assets = len([s for s in symbols if s != "VNINDEX"])
    edge_dim = 1  # Correlation weights

    print(f"Model parameters - Input dim: {input_dim}, Edge dim: {edge_dim}, Num assets: {num_assets}")

    # Initialize model components
    gat_encoder = GAT_Encoder(
        input_dim=config["input_dim"],
        hid_dim=config["hid_dim"],  # Ensure hid_dim is defined in config
        edge_dim=config["edge_dim"],
        gnn_embed_dim=config["gnn_embed_dim"],
        heads=config.get("gat_heads", 4)  # Default to 4 heads if not specified
    )

    lstm_allocator = LSTM_Allocator(
        input_dim=config["gnn_embed_dim"],  # Input matches GAT embedding size
        hidden_dim=config["lstm_hidden_dim"],
        num_layers=config["lstm_layers"],
        dropout=config["dropout"]
    )

    mlp_allocator = MLP_Allocator(
        input_dim=config["gnn_embed_dim"],  # Input is the same as the LSTM output size
        hidden_dim=config["mlp_hidden_dim"],
        output_dim=num_assets
    )

    # Create full GAT-LSTM model
    model = GAT_LSTM(
        gat_encoder=gat_encoder,
        lstm_allocator=lstm_allocator,
        mlp_allocator=mlp_allocator,
        device=device
    ).to(device)

    # Print total model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Define Log Return Loss
    criterion = SharpeRatioLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # Since Sharpe ratio is maximized
        factor=0.5,
        patience=config.get("patience", 5),
        verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'lr': []
    }

    # Early stopping setup
    best_val_loss = float("inf")
    early_stopping_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]") as train_bar:
            for batch in train_bar:
                print(f"Batch type: {type(batch)}")
                print(f"Batch content: {batch}")
                data_seq, target_returns = batch

                # Move data to device
                # Flatten and move each graph to device
                print("Data sequence content:", data_seq)

                data_seq = [[g.to(device) if hasattr(g, 'to') else g for g in seq] for seq in data_seq]


                target_returns = target_returns.to(device)

                # Handle VNINDEX adjustment to ensure dimensions match
                if model.vnindex_idx >= 0:
                    target_returns_adjusted = torch.index_select(
                        target_returns, dim=1,
                        index=torch.tensor(
                            [i for i in range(target_returns.shape[1]) if i != model.vnindex_idx],
                            device=target_returns.device
                        )
                    )
                else:
                    target_returns_adjusted = target_returns

                print(f"Type of data_seq: {type(data_seq)}")
                if isinstance(data_seq, list):
                    print(f"Type of first element: {type(data_seq[0])}")

                pred_weights = model(data_seq)


                # # Handle batch size mismatch (shouldn't happen but just in case)
                # if pred_weights.shape[0] != target_returns_adjusted.shape[0]:
                #     min_batch = min(pred_weights.shape[0], target_returns_adjusted.shape[0])
                #     pred_weights = pred_weights[:min_batch]
                #     target_returns_adjusted = target_returns_adjusted[:min_batch]

                # Compute loss
                loss = criterion(pred_weights, target_returns_adjusted)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        print(f"Length of val_loader: {len(val_loader)}")

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]") as val_bar:
                for batch in val_bar:
                    data_seq, target_returns = batch

                    # Move data to device
                    data_seq = [[g.to(device) for g in seq] for seq in data_seq]
                    target_returns = target_returns.to(device)

                    # Handle VNINDEX adjustment (same as training)
                    if model.vnindex_idx >= 0:
                        target_returns_adjusted = torch.index_select(
                            target_returns, dim=1,
                            index=torch.tensor(
                                [i for i in range(target_returns.shape[1]) if i != model.vnindex_idx],
                                device=target_returns.device
                            )
                        )
                    else:
                        target_returns_adjusted = target_returns  # No change needed if VNINDEX is not in data

                    # Forward pass
                    pred_weights = model(data_seq)

                    # Handle batch size mismatch (same as training)
                    if pred_weights.shape[0] != target_returns_adjusted.shape[0]:
                        min_batch = min(pred_weights.shape[0], target_returns_adjusted.shape[0])
                        pred_weights = pred_weights[:min_batch]
                        target_returns_adjusted = target_returns_adjusted[:min_batch]

                    # Compute loss
                    loss = criterion(pred_weights, target_returns_adjusted)

                    val_losses.append(loss.item())
                    val_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Compute average validation loss
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(avg_val_loss)

        print(
            f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'gat_config': {
                    'input_dim': input_dim,
                    'edge_dim': edge_dim,
                    'gnn_embed_dim': config["gnn_embed_dim"]
                },
                'lstm_config': {
                    'gnn_embed_dim': config["gnn_embed_dim"],
                    'lstm_hidden_dim': config["lstm_hidden_dim"],
                    'num_assets': num_assets
                }
            }, model_save_path)
            print(f"✅ Model saved with val loss: {best_val_loss:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training complete.")

    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_losses = []
    all_pred_weights = []
    all_target_returns = []

    with torch.no_grad():
        with tqdm(test_loader, desc=f"Testing") as test_bar:
            for batch in test_bar:
                data_seq, target_returns = batch

                # Move data to device
                data_seq = [[g.to(device) for g in seq] for seq in data_seq]
                target_returns = target_returns.to(device)

                # Handle VNINDEX adjustment (same as validation)
                if model.vnindex_idx >= 0:
                    target_returns_adjusted = torch.index_select(
                        target_returns, dim=1,
                        index=torch.tensor(
                            [i for i in range(target_returns.shape[1]) if i != model.vnindex_idx],
                            device=target_returns.device
                        )
                    )
                else:
                    target_returns_adjusted = target_returns  # No change if VNINDEX is not in data

                # Forward pass
                pred_weights = model(data_seq)
                print(f"Model output shape: {pred_weights.shape}")

                # # Handle batch size mismatch (same as validation)
                # if pred_weights.shape[0] != target_returns_adjusted.shape[0]:
                #     min_batch = min(pred_weights.shape[0], target_returns_adjusted.shape[0])
                #     pred_weights = pred_weights[:min_batch]
                #     target_returns_adjusted = target_returns_adjusted[:min_batch]

                print(f"pred_weights shape: {pred_weights.shape}")  # Expected: [batch_size, num_assets]
                print(
                    f"target_returns_adjusted shape: {target_returns_adjusted.shape}")  # Expected: [batch_size, num_assets]


                # Compute loss
                loss = criterion(pred_weights, target_returns_adjusted)
                test_losses.append(loss.item())

                # Store predictions and targets
                all_pred_weights.append(pred_weights.cpu().numpy())
                all_target_returns.append(target_returns_adjusted.cpu().numpy())

                test_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Compute average test loss
    avg_test_loss = np.mean(test_losses)

    # Stack all predictions and target returns
    all_pred_weights = np.vstack(all_pred_weights)
    all_target_returns = np.vstack(all_target_returns)

    # Compute portfolio performance
    portfolio_returns = np.sum(all_pred_weights * all_target_returns, axis=1)
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    sharpe_ratio = portfolio_returns.mean() / (portfolio_returns.std() + 1e-6) * np.sqrt(252)

    # Store weights in DataFrame (without VNINDEX)
    symbols_no_vnindex = [sym for sym in symbols if sym != "VNINDEX"]
    weights_df = pd.DataFrame(all_pred_weights, columns=symbols_no_vnindex)

    # Print results
    print(weights_df.tail())
    print(f"Test Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Cumulative Return: {cumulative_returns[-1] * 100:.2f}%")
    print(f"Test Loss: {avg_test_loss:.4f}")

    import matplotlib.pyplot as plt

    # --- 1. Loss curves ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')

    if 'test_loss' in history and any(history['test_loss']):
        test_epochs = [i * 1 for i in range(len(history['test_loss']))]
        plt.plot(test_epochs, history['test_loss'], label='Test Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training / Validation / Test Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/loss_plot.png')
    plt.show()

    # --- 2. Cumulative returns ---
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_returns, label=f'Cumulative Return (Sharpe: {sharpe_ratio:.2f})')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.title('Backtest Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/cumulative_returns.png')
    plt.show()

    # --- 3. Portfolio weights over time ---
    plt.figure(figsize=(12, 6))
    plt.stackplot(range(len(all_pred_weights)), all_pred_weights.T, labels=symbols_no_vnindex, alpha=0.8)
    plt.xlabel('Trading Days')
    plt.ylabel('Asset Weights')
    plt.title('Portfolio Allocation Over Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig('results/portfolio_allocation.png')
    plt.show()

    return model, history, {
         'sharpe_ratio': sharpe_ratio,
         'cumulative_return': cumulative_returns[-1],
         'pred_weights': all_pred_weights,
         'target_returns': all_target_returns,
         'portfolio_returns': portfolio_returns
     }


def main(stock_list, config):
    load_dotenv()
    DATA_PATH = os.getenv("DATA_PATH", "data")
    stock_data_path = os.path.join(DATA_PATH, "features.pkl")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Fetching stock data...")
    fetch_stock_data(data_path=DATA_PATH, stocks=stock_list,
                     start_date="2024-01-10", end_date="2025-03-17")

    logging.info("Generating features...")
    generate_features(data_path=DATA_PATH)

    # Create directories for results
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    logging.info("Preparing data...")
    data = prepare_gnn_lstm_data(data_path=stock_data_path, seq_len=config["seq_len"], corr_threshold=config["corr_threshold"])

    logging.info("Data loaded and prepared.")
    logging.info(f"Number of assets: {data['num_assets']}")
    logging.info(f"Number of features: {data['num_features']}")

    # Print dataset sample
    train_dataset = data['train_dataset']
    data_seq, target_returns = train_dataset[0]
    logging.info(f"Sample sequence length: {len(data_seq)}")
    logging.info(f"Sample graph data: Nodes={data_seq[0].x.shape}, Edges={data_seq[0].edge_index.shape}")
    logging.info(f"Target returns shape: {target_returns.shape}")

    # Train model
    logging.info("Training model...")
    model, history, metrics = train_gat_lstm(
        config=config,
        train_df=data['train_dataset'].data,
        val_df=data['val_dataset'].data,
        test_df=data['test_dataset'].data,
        edge_index=data['edge_index'],
        edge_weight=data['edge_attr'],
        feature_cols=data['feature_cols'],
        symbols=data['symbols'],
        seq_len=config["seq_len"],
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        patience=config["patience"],
        early_stopping_patience=config["early_stopping_patience"],
        model_save_path=config["model_save_path"]
    )

    logging.info("=== Final Performance Metrics ===")
    logging.info(f" Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logging.info(f"Cumulative Return: {metrics['cumulative_return'] * 100:.2f}%")

    # Save portfolio weights
    symbols_no_vnindex = [sym for sym in data['symbols'] if sym != "VNINDEX"]

    weights_df = pd.DataFrame(
        metrics['pred_weights'],
        columns=symbols_no_vnindex
    )

    logging.info(f"Optimal Training Portfolio: {weights_df.iloc[-1].to_dict()}")

    weights_path = 'results/portfolio_weights.csv'
    weights_df.to_csv(weights_path, index=False)
    logging.info(f"Portfolio weights saved to {weights_path}")

    # Save returns
    returns_df = pd.DataFrame({
        'portfolio_return': metrics['portfolio_returns'],
        'cumulative_return': np.cumprod(1 + metrics['portfolio_returns']) - 1
    })
    returns_path = 'results/portfolio_returns.csv'
    returns_df.to_csv(returns_path, index=False)
    logging.info(f"Returns saved to {returns_path}")

    logging.info("All results successfully saved.")


if __name__ == "__main__":
    stock_list = ["VNINDEX", "CTG", "HCM"]

    config = {
        "seq_len": 15,  # Reduce sequence length to lower memory usage
        "batch_size": 32,  # Reduce batch size to fit GPU memory
        "num_epochs": 1,  # Allow enough training time
        "learning_rate": 1e-3,  # Keep learning rate stable
        "weight_decay": 1e-4,  # Moderate regularization
        "patience": 7,
        "early_stopping_patience": 10,
        "input_dim": 7,
        "hid_dim": 32,  # Reduce GAT hidden dim to cut params (was 128)
        "edge_dim": 1,
        "gnn_embed_dim": 32,  # Reduce GNN embedding size
        "lstm_hidden_dim": 32,  # Reduce LSTM size to lower param count
        "lstm_layers": 1,  # Use a single LSTM layer to save memory
        "dropout": 0.2,
        "mlp_hidden_dim": 32,  # Reduce MLP size
        "gat_heads": 2,  # Reduce attention heads (was 8)
        "model_save_path": "checkpoint/gat_lstm_portfolio.pt",
        "corr_threshold": 0.6
    }

    main(stock_list, config)
#
# import itertools
# from copy import deepcopy
#
# if __name__ == "__main__":
#     stock_list = ["VNINDEX", "CTG", "STB", "HCM", "VCI"]
#
#     base_config = {
#         "seq_len": 15,
#         "batch_size": 32,
#         "num_epochs": 1,
#         "learning_rate": 1e-4,
#         "weight_decay": 1e-5,
#         "patience": 1,
#         "early_stopping_patience": 10,
#         "gnn_hid_dim": 16,
#         "gnn_embed_dim": 32,
#         "lstm_hidden_dim": 128,
#         "lstm_num_layers": 2,
#         "dropout": 0.5,
#         "model_save_path": "checkpoint/gat_lstm_portfolio.pt",
#         "corr_threshold": 0.4
#     }
#
#     grid = {
#         "seq_len": [10, 15],
#         "learning_rate": [1e-4, 3e-4],
#         "dropout": [0.3, 0.5, 0.7],
#         "corr_threshold": [0.4, 0.5, 0.6]
#     }
#
#     keys, values = zip(*grid.items())
#     for i, combo in enumerate(itertools.product(*values)):
#         config = deepcopy(base_config)
#         config.update(dict(zip(keys, combo)))
#         print(f"\n[Run {i+1}] Config: {config}")
#         main(stock_list, config)


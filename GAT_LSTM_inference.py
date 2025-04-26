import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm

from utils.loss import SharpeRatioLoss
from utils.graph_dataset import GNNLSTMDataset
from models.GAT_LSTM import GAT_Encoder, LSTM_Allocator, GAT_LSTM

from src.data_loader import fetch_stock_data
from src.feature_engineering import generate_features

def collate_fn(batch):
    # batch is a list of tuples: (data_seq, target)
    data_seqs, targets = zip(*batch)
    return list(data_seqs), torch.stack(targets)


def train_gat_lstm(
        config,
        train_df,
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
    print(f"Using device: {device}")

    # Ensure directory for model saving exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Build datasets
    train_dataset = GNNLSTMDataset(train_df, seq_len, edge_index, edge_weight, feature_cols)


    print(f"Datasets created - Train: {len(train_dataset)}")

    # Create data loaders
    train_loader = GraphDataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # Model setup
    input_dim = len(feature_cols)
    num_assets = len([s for s in symbols if s != "VNINDEX"])
    edge_dim = 1  # Correlation weights

    print(f"Model parameters - Input dim: {input_dim}, Edge dim: {edge_dim}, Num assets: {num_assets}")

    # Initialize model components
    gat_encoder = GAT_Encoder(
        input_dim=input_dim,
        hid_dim=config["gnn_hid_dim"],
        edge_dim=edge_dim,
        gnn_embed_dim=config["gnn_embed_dim"],
        dropout=config["dropout"]
    )

    lstm_model = LSTM_Allocator(
        gnn_embed_dim=config["gnn_embed_dim"],
        lstm_hidden_dim=config["lstm_hidden_dim"],
        num_lstm_layers=config["lstm_num_layers"],
        num_assets=num_assets,
        dropout=config["dropout"]
    )

    model = GAT_LSTM(gat_encoder, lstm_model, symbols).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = SharpeRatioLoss(risk_free_rate=0.0, annualization_factor=252)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=patience,
        verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
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
                optimizer.zero_grad()
                data_seq, target_returns = batch

                # Move data to device
                data_seq = [g.to(device) for g in data_seq]
                target_returns = target_returns.to(device)

                # Handle VNINDEX adjustment to ensure dimensions match
                if model.vnindex_idx >= 0 and target_returns.shape[1] != len(model.symbols) - 1:
                    # Remove VNINDEX from target returns if it exists
                    target_returns_adjusted = torch.cat([
                        target_returns[:, :model.vnindex_idx],
                        target_returns[:, model.vnindex_idx:] if model.vnindex_idx == target_returns.shape[
                            1] else target_returns[:, model.vnindex_idx + 1:]
                    ], dim=1)
                else:
                    target_returns_adjusted = target_returns

                # Forward pass
                pred_weights = model(data_seq)

                # Handle batch size mismatch (shouldn't happen but just in case)
                if pred_weights.shape[0] != target_returns_adjusted.shape[0]:
                    min_batch = min(pred_weights.shape[0], target_returns_adjusted.shape[0])
                    pred_weights = pred_weights[:min_batch]
                    target_returns_adjusted = target_returns_adjusted[:min_batch]

                # Compute loss
                loss = criterion(pred_weights, target_returns_adjusted)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

    # After training, evaluate on the entire training set
    model.eval()
    all_pred_weights = []
    all_target_returns = []
    train_eval_losses = []

    with torch.no_grad():
        for batch in train_loader:  # Use train_loader to get 100% of data
            data_seq, target_returns = batch
            data_seq = [g.to(device) for g in data_seq]
            target_returns = target_returns.to(device)

            # Handle VNINDEX adjustment
            vnindex_idx = symbols.index("VNINDEX") if "VNINDEX" in symbols else -1
            if vnindex_idx >= 0:
                target_returns_adjusted = torch.cat([
                    target_returns[:, :vnindex_idx],
                    target_returns[:, vnindex_idx + 1:]
                ], dim=1)
            else:
                target_returns_adjusted = target_returns

            pred_weights = model(data_seq)

            # Handle batch size mismatch
            if pred_weights.shape[0] != target_returns_adjusted.shape[0]:
                min_batch = min(pred_weights.shape[0], target_returns_adjusted.shape[0])
                pred_weights = pred_weights[:min_batch]
                target_returns_adjusted = target_returns_adjusted[:min_batch]

            # Compute loss
            loss = criterion(pred_weights, target_returns_adjusted)
            train_eval_losses.append(loss.item())

            all_pred_weights.append(pred_weights.cpu().numpy())
            all_target_returns.append(target_returns_adjusted.cpu().numpy())

    # Calculate performance metrics
    avg_eval_loss = np.mean(train_eval_losses)
    all_pred_weights = np.vstack(all_pred_weights)
    all_target_returns = np.vstack(all_target_returns)

    # Calculate portfolio returns and Sharpe ratio
    portfolio_returns = np.sum(all_pred_weights * all_target_returns, axis=1)
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

    # Create DataFrame for the weights
    symbols_no_vnindex = [sym for sym in symbols if sym != "VNINDEX"]
    weights_df = pd.DataFrame(all_pred_weights, columns=symbols_no_vnindex)

    # Print the results
    print(weights_df.tail())
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Cumulative Return: {cumulative_returns[-1] * 100:.2f}%")
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")

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
                     start_date="2017-07-10", end_date="2025-03-17")

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
    logging.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
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
    stock_list = ["VNINDEX", "CTG", "HCM", "STB"]

    config = {
        "seq_len": 5,  #
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "patience": 1,
        "early_stopping_patience": 10,
        "gnn_hid_dim": 8,
        "gnn_embed_dim": 16,
        "lstm_hidden_dim": 128,  #
        "lstm_num_layers": 2,
        "dropout": 0.7,  #
        "model_save_path": "checkpoint/gat_lstm_portfolio.pt",
        "corr_threshold": 0.6  #
    }
    main(stock_list, config)



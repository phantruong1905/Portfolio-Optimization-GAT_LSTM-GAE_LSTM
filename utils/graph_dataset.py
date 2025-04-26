import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data


class GNNLSTMDataset(Dataset):
    def __init__(self, df, seq_len, edge_index, edge_weight, feature_cols):
        super().__init__()
        self.seq_len = seq_len
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.data = df[['Date', 'Symbol', 'log_return_raw'] + feature_cols].copy()

        # Get unique symbols for mapping
        self.symbols = sorted(self.data['Symbol'].unique())
        self.symbol_to_idx = {sym: i for i, sym in enumerate(self.symbols)}
        self.num_assets = len(self.symbols)

        # Pivot each feature separately, then stack along feature axis
        ts_list = []
        for feat in feature_cols:
            feat_df = self.data.pivot(index='Date', columns='Symbol', values=feat)
            ts_list.append(feat_df)

        # Result shape: [num_dates, num_symbols, num_features]
        self.time_series = pd.concat(ts_list, axis=1, keys=feature_cols).dropna()

        # Also pivot log_return to get targets
        self.returns = self.data.pivot(index='Date', columns='Symbol', values='log_return_raw').dropna()

        # Align dates just in case
        common_dates = self.time_series.index.intersection(self.returns.index)
        self.time_series = self.time_series.loc[common_dates]
        self.returns = self.returns.loc[common_dates]

        self.dates = self.time_series.index.tolist()
        self.feature_cols = feature_cols
        self.num_features = len(feature_cols)

    def __len__(self):
        # Return number of possible sequences
        return len(self.dates) - self.seq_len

    def __getitem__(self, idx):
        """
        Returns a sequence of length seq_len of PyG Data objects,
        along with the next period returns as target
        """
        # Get sequence of dates
        seq_dates = self.dates[idx:idx + self.seq_len]
        target_date = self.dates[idx + self.seq_len]  # Next day return

        # Create sequence of graph data objects
        data_seq = []
        for date in seq_dates:
            # Get features for all assets on this date
            features = []
            for feat in self.feature_cols:
                feat_values = self.time_series.loc[date, (feat, slice(None))].values
                features.append(feat_values)

            # Stack features for all assets: [num_assets, num_features]
            node_features = np.column_stack(features)

            # Create PyG Data object
            data = Data(
                x=torch.FloatTensor(node_features),
                edge_index=self.edge_index.clone(),  # Use the provided edge index
                edge_attr=self.edge_weight.clone(),  # Use the provided edge weights
                # Add batch indicator for global pooling (all nodes are in one graph)
                batch=torch.zeros(len(node_features), dtype=torch.long)
            )

            data_seq.append(data)

        # Get returns for the target date (next period returns)
        target_returns = torch.FloatTensor(self.returns.loc[target_date].values)

        return data_seq, target_returns

    def get_symbols(self):
        """Return the list of symbols in the dataset"""
        return self.symbols



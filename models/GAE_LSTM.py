import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Model definitions
class GAE_Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super(GAE_Encoder, self).__init__()
        self.conv1 = GATConv(in_channels=in_dim, out_channels=hidden_dim, heads=heads, dropout=0.1)
        self.conv2 = GATConv(in_channels=hidden_dim * heads, out_channels=out_dim, heads=1, dropout=0.1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GAE_Encoder(in_dim, hidden_dim, out_dim, heads)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single continuous value (correlation)
        )

    def encode(self, x, edge_index, edge_attr=None):
        return self.encoder(x, edge_index, edge_attr)

    def decode(self, z, edge_index):
        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        return self.decoder_mlp(edge_features)  # Predict correlation value

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        edge_pred = self.decode(z, edge_index)
        return edge_pred, z


class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim]
        # Use the last hidden state
        return h_n[-1]  # [batch_size, hidden_dim]


class PortfolioLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_assets, dropout=0.2):
        super(PortfolioLSTM, self).__init__()
        self.num_assets = num_assets
        self.stock_lstm = StockLSTM(input_dim, hidden_dim, num_layers, dropout)
        self.fc = nn.Linear(hidden_dim * num_assets, num_assets)  # Combine and predict weights

    def forward(self, x):
        # x: [batch_size, num_assets, seq_len, input_dim]
        batch_size = x.size(0)
        lstm_outputs = []

        # Process each stock individually
        for i in range(self.num_assets):
            stock_input = x[:, i, :, :]  # [batch_size, seq_len, input_dim]
            stock_output = self.stock_lstm(stock_input)  # [batch_size, hidden_dim]
            lstm_outputs.append(stock_output)

        # Combine LSTM outputs
        combined = torch.cat(lstm_outputs, dim=1)  # [batch_size, hidden_dim * num_assets]

        # Predict portfolio weights
        weights = self.fc(combined)  # [batch_size, num_assets]
        weights = F.softmax(weights, dim=1)  # Ensure weights sum to 1
        return weights




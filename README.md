# Portfolio Optimization using Graph-Based Deep Learning

This project focuses on building a portfolio optimization model for the Vietnam stock market by leveraging graph neural networks and deep learning architectures.

## Model Architecture

- **Graph Autoencoder (GAE)** to learn latent relationships between stocks based on historical returns and technical indicators.
- Encoded features are then passed into an **LSTM** network to model temporal dependencies over time.
- Additionally experimented with **Graph Attention Networks (GAT)** combined with LSTM for improved performance.
- A **custom loss function** was designed based on the **Sharpe Ratio** to directly optimize for risk-adjusted returns.

## Highlights

- Outperformed traditional methods such as **Efficient Frontier** and **Equal Weights Allocation** during backtesting.
- Custom loss function (Sharpe-based) significantly enhanced stability and profitability of generated portfolios.

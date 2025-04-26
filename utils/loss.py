import torch
import torch.nn as nn

class SharpeRatioLoss(nn.Module):
    def __init__(self, risk_free_rate=0, annualization_factor=252):
        super(SharpeRatioLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def forward(self, pred_weights, actual_returns):
        """
        Args:
            pred_weights: [batch_size, num_assets] - predicted portfolio weights (softmaxed)
            actual_returns: [batch_size, num_assets] - actual next-day asset log returns

        Returns:
            Negative Sharpe ratio (for minimization)
        """
        if actual_returns.shape[1] > pred_weights.shape[1]:  # Handle extra index (e.g., VNINDEX)
            actual_returns = actual_returns[:, :pred_weights.shape[1]]

        portfolio_returns = torch.sum(pred_weights * actual_returns, dim=1)  # [batch_size]
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std() + 1e-6  # Avoid division by zero

        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
        sharpe_ratio *= torch.sqrt(torch.tensor(self.annualization_factor, device=pred_weights.device))

        return -sharpe_ratio  # Negative for minimization

class LogReturnLoss(nn.Module):
    def __init__(self):
        super(LogReturnLoss, self).__init__()

    def forward(self, pred_weights, actual_returns):
        """
        Args:
            pred_weights: [batch_size, num_assets] - predicted portfolio weights (already softmaxed)
            actual_returns: [batch_size, num_assets] - actual next-day asset log returns

        Returns:
            Negative mean log return (for minimization)
        """
        if actual_returns.shape[1] > pred_weights.shape[1]:  # VNINDEX is still there
            actual_returns = actual_returns[:, :pred_weights.shape[1]]

        portfolio_returns = torch.sum(pred_weights * actual_returns, dim=1)  # [batch_size]

        return -portfolio_returns.mean()

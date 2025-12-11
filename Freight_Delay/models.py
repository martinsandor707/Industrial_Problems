import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3, hidden_layers=None):
        """
        Args:
            input_dim (int): number of numerical features in the input
            hidden_dim (int): (Deprecated/Default) size of first hidden layer. Used if hidden_layers is None.
            dropout (float): dropout rate to prevent overfitting
            hidden_layers (list): List of hidden layer sizes. If provided, overrides hidden_dim logic.
        """
        super(BinaryClassifier, self).__init__()

        if hidden_layers is None:
            # Default structure similar to original but using the new sequential builder
            # Original: input -> hidden_dim -> hidden_dim/2 -> 1
            hidden_layers = [hidden_dim, hidden_dim // 2]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path="checkpoint.pt"):
        """
        patience: how many epochs to wait without improvement
        delta: minimum change in validation loss to be considered improvement
        path: file to save the best model
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)  # save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

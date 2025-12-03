import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        """
        Args:
            input_dim (int): number of numerical features in the input
            hidden_dim (int): size of hidden layer
            dropout (float): dropout rate to prevent overfitting
        """
        super(BinaryClassifier, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # output layer

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        # x = torch.sigmoid(self.fc3(x))  # binary output between 0–1
        return x


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
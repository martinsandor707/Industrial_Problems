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
        x = torch.sigmoid(self.fc3(x))  # binary output between 0–1
        return x

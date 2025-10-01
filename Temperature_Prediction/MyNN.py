import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TemperatureRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TemperatureRegressor, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  # Input -> Hidden
        self.ln = nn.LayerNorm(hidden_size)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 1)  # Hidden -> Output (1 value)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.activation(x)
        x = self.fc2(x)  # No activation here, pytorch loss functions have activation by default
        return x.squeeze(-1)

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


class TwoInputNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # first input: a single important feature
        self.fc1 = nn.Linear(input_size, input_size )

        # second input: 50 less important features
        self.fc2 = nn.Linear(51, 3)

        # after concatenation
        self.fc_combined = nn.Linear(input_size+3, hidden_size)

        self.activation = nn.Tanh()
        self.lin = nn.Linear()

    def forward(self, x1, x2):
        # x1 shape: (batch, 1)
        # x2 shape: (batch, 50)

        h1 = self.activation(self.fc1(x1))
        h2 = self.activation(self.fc2(x2))

        # concatenate along feature dimension
        h = torch.cat([h1, h2], dim=1)

        out = self.fc_combined(h)
        return out
    
class ProbabilisticRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()
        self.mean_head = nn.Linear(hidden_size, 1)
        self.std_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.activation(self.fc(x))

        mean = self.mean_head(h)  # linear output
        std = F.softplus(self.std_head(h))  # strictly positive

        return [mean, std]
    

##### Claude's version of ProbabilisticRegressor #####
class EnsembleTemperatureForecastNN(nn.Module):
    """
    Neural Network for temperature forecasting from ensemble predictions.
    Outputs parameters of a Gaussian distribution (mean and std).
    
    Args:
        n_ensemble: Number of ensemble members (default: 50)
        hidden_sizes: List of hidden layer sizes (default: [128, 64, 32])
        dropout_rate: Dropout probability (default: 0.2)
    """
    def __init__(self, n_ensemble=50, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(EnsembleTemperatureForecastNN, self).__init__()
        
        self.n_ensemble = n_ensemble
        
        # Build hidden layers
        layers = []
        input_size = n_ensemble
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer: 2 outputs (mean and log_std)
        # We predict log(std) to ensure std is always positive
        self.output_layer = nn.Linear(hidden_sizes[-1], 2)
        
        # Minimum std to avoid numerical instability
        self.min_std = 1e-6
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, n_ensemble) containing ensemble forecasts
            
        Returns:
            mean: Predicted mean temperature, shape (batch_size, 1)
            std: Predicted standard deviation, shape (batch_size, 1)
        """
        # Pass through hidden layers
        h = self.hidden_layers(x)
        
        # Get output parameters
        output = self.output_layer(h)
        
        # Split into mean and log_std
        mean = output[:, 0:1]
        log_std = output[:, 1:2]
        
        # Convert log_std to std with minimum bound
        std = torch.exp(log_std) + self.min_std
        
        return torch.cat([mean, std], dim=1)
    
    def predict_distribution(self, x):
        """
        Returns a torch.distributions.Normal object for the predicted distribution.
        
        Args:
            x: Tensor of shape (batch_size, n_ensemble)
            
        Returns:
            Normal distribution object
        """
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)
    
    def sample(self, x, n_samples=1):
        """
        Sample from the predicted distribution.
        
        Args:
            x: Tensor of shape (batch_size, n_ensemble)
            n_samples: Number of samples to draw per input
            
        Returns:
            Samples of shape (batch_size, n_samples)
        """
        dist = self.predict_distribution(x)
        return dist.sample((n_samples,)).squeeze(-1).transpose(0, 1)
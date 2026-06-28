import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """
    LSTM for behavioral prediction with multi-scale input.
    LSTMs maintain a hidden state across the sequence, allowing them
    to learn long-term patterns (minutes to hours).
    """
    def __init__(
        self,
        n_features: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        # Input projection (optional, helps training)
        self.input_proj = nn.Linear(n_features, n_features)
        
        # LSTM: maintains hidden state across sequence
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(n_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_features),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        returns: (batch, n_features) — predicted next frame
        """
        # Project input
        x = self.input_proj(x)
        
        # LSTM: returns (output, (h_n, c_n))
        # output: (batch, seq_len, n_hidden)
        # h_n: final hidden state
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state and predict
        last_hidden = h_n[-1]  # (batch, n_hidden)
        y_pred = self.output(last_hidden)
        
        return y_pred
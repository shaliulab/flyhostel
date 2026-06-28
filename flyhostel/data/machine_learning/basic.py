import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from .dataset import BehaviorSequenceDataset


class SequenceTransformer(nn.Module):
    """
    Transformer for predicting future behavior from past sequences.
    
    Input: (batch, seq_len, n_features)
    Output: (batch, n_features) — predicts one frame ahead
    """
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        d_ff: int = 256,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = self._build_positional_encoding(seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection (predict next frame)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, n_features),
        )
    
    def _build_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)
        
        Returns
        -------
        (batch, n_features) — predicted next frame
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        device = x.device
        pe = self.pos_encoder.to(device)
        x = x + pe[:, :x.shape[1], :]
        
        # Transformer
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Take last frame and project to output
        x = x[:, -1, :]  # (batch, d_model)
        x = self.output_proj(x)  # (batch, n_features)
        
        return x


def train_transformer(
    X: np.ndarray,
    seq_len: int = 10,
    n_epochs: int = 100,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
) -> dict:
    """
    Train a transformer to predict next frame from sequence.
    
    Parameters
    ----------
    X            : (n_frames, n_features)
    seq_len      : lookback window in frames (default 10 ≈ 0.4 sec at 25 fps)
    n_epochs     : number of training epochs
    batch_size   : batch size
    device       : 'cuda' or 'cpu'
    verbose      : print progress
    
    Returns
    -------
    {
        'model': trained model,
        'scaler': fitted StandardScaler for normalization,
        'history': {'train_loss': [...], 'val_loss': [...]},
    }
    """
    # Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # Create dataset
    dataset = BehaviorSequenceDataset(X_norm, seq_len=seq_len, stride=1)
    
    # Temporal split: 80% train, 20% validation
    n_train = int(0.8 * len(dataset))
    train_indices = np.arange(0, n_train)
    val_indices = np.arange(n_train, len(dataset))
    
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # Model
    model = SequenceTransformer(
        n_features=X.shape[1],
        seq_len=seq_len,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ).to(device)
    
    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = np.inf
    patience = 20
    no_improve_count = 0
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x_seq, y_true in train_loader:
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(x_seq)
        
        train_loss /= len(train_set)
        history['train_loss'].append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, y_true in val_loader:
                x_seq, y_true = x_seq.to(device), y_true.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item() * len(x_seq)
        
        val_loss /= len(val_set)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_count += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if no_improve_count >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'device': device,
    }


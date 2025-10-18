"""TimesNet model training and evaluation"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path

from bitcoin.modeling.models import TimesNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tqdm.auto import tqdm


class TimeSeriesDataset(Dataset):
    """Time series dataset for sequence-to-sequence prediction"""
    def __init__(self, data, seq_len=30, pred_len=1, label_len=0, features='MS'):
        if hasattr(data, 'columns') and 'date' in data.columns:
            data = data.drop(columns=['date'])

        self.data = data.values if hasattr(data, 'values') else data

        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features

        # Data format: [target, feature1, feature2, ...]
        # Column 0: target variable
        # Columns 1+: input features
        self.data_x = self.data[:, 1:]  # Features only (exclude target column 0)
        self.data_y = self.data[:, :1]  # Target only (column 0)

        self.n_features = self.data_x.shape[1]  # Number of input features

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len


        seq_x = self.data_x[s_begin:s_end]  # [seq_len, n_features]

        # Output: Future data (all features or just target, depending on model)
        seq_y = self.data_y[r_begin:r_end]  # [label_len+pred_len, n_features]

        # Time marks (zeros for now, can add real dates later)
        # For daily frequency (freq='d'), need 3 features: [month, day, weekday]
        seq_x_mark = np.zeros((self.seq_len, 3))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 3))

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.FloatTensor(seq_x_mark),
            torch.FloatTensor(seq_y_mark)
        )


class Config:
    """Configuration object for TimesNet model"""
    def __init__(self,
                 seq_len=30,           # Input sequence length
                 pred_len=1,           # Prediction length
                 label_len=0,          # Label length (not used for forecasting)
                 enc_in=63,            # Number of input features
                 c_out=1,              # Number of output features
                 d_model=64,           # Model dimension
                 d_ff=128,             # Feed-forward dimension
                 e_layers=2,           # Number of encoder layers
                 top_k=3,              # Top-k frequencies for TimesBlock
                 num_kernels=6,        # Number of inception kernels
                 dropout=0.1,          # Dropout rate
                 embed='timeF',        # Time features encoding
                 freq='d'):            # Frequency (d=daily)

        self.task_name = 'short_term_forecast'
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_in = enc_in
        self.dec_in = c_out
        self.c_out = c_out
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.dropout = dropout
        self.embed = embed
        self.freq = freq


class TimesNetRegressor:
    """
    TimesNet regressor with scikit-learn style API

    Why TimesNet?
    -------------
    1. State-of-the-art for short-term forecasting (ranked #1 in Time-Series-Library)
    2. Multi-periodicity modeling via FFT discovers Bitcoin's market cycles
    3. 2D convolution captures complex temporal patterns
    4. Built-in normalization handles Bitcoin's non-stationary price dynamics
    5. Efficient for 1-day ahead prediction with moderate sequence length

    Architecture:
    ------------
    - TimesBlock: FFT-based period discovery + 2D Inception convolution
    - Multi-scale temporal modeling (top-k periods)
    - Adaptive aggregation of period representations
    - Non-stationary normalization (critical for Bitcoin)

    API follows scikit-learn conventions:
        - fit(train_data, val_data): Train the model
        - predict(data): Make predictions
        - score(data): Return R² score
        - get_params(): Get model parameters
        - set_params(**params): Update parameters
    """

    def __init__(self,
                 seq_len=30,
                 pred_len=1,
                 n_features=63,
                 d_model=64,
                 d_ff=128,
                 e_layers=2,
                 top_k=3,
                 num_kernels=6,
                 dropout=0.1,
                 learning_rate=0.001,
                 batch_size=32,
                 device='auto',
                 verbose=True):

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.dropout = dropout

        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Initialize model config
        self.config = Config(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_features,
            c_out=1,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            top_k=top_k,
            num_kernels=num_kernels,
            dropout=dropout
        )

        # Initialize model
        self.model = TimesNet(self.config).float().to(self.device)

        # Initialize optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.is_fitted = False

    def create_dataloader(self, data, shuffle=False):
        """
        Create PyTorch DataLoader following official Time-Series-Library format

        Args:
            data: Unified DataFrame/array with ALL features (including target)
            shuffle: Whether to shuffle data
        """
        dataset = TimeSeriesDataset(data, seq_len=self.seq_len, pred_len=self.pred_len)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # Decoder input (zeros for forecasting)
            dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Compute loss
            loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])

            # Backward pass
            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())

        return np.mean(train_loss)

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_loss = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Compute loss
                loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])
                val_loss.append(loss.item())

        return np.mean(val_loss)

    def fit(self, train_data, val_data, epochs=50, patience=10):
        """
        Train TimesNet model with early stopping

        Args:
            train_data: Training DataFrame with ALL features (including target)
            val_data: Validation DataFrame with ALL features (including target)
            epochs: Maximum training epochs
            patience: Early stopping patience

        Returns:
            self: Returns self for method chaining
        """
        train_loader = self.create_dataloader(train_data, shuffle=True)
        val_loader = self.create_dataloader(val_data, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Print header (only if verbose)
        if self.verbose:
            print(f"\nTraining TimesNet | Device: {self.device} | Params: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        # Training loop with tqdm progress bar
        if self.verbose:
            pbar = tqdm(range(epochs), desc='Training', ncols=100, colour='green')
            epochs_iter = pbar
        else:
            epochs_iter = range(epochs)

        for epoch in epochs_iter:
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start

            # Early stopping
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                is_best = True
            else:
                patience_counter += 1

            # Update progress bar
            if self.verbose:
                status = f"★ BEST" if is_best else f"Patience: {patience_counter}/{patience}"
                pbar.set_postfix({
                    'Train Loss': f'{train_loss:.6f}',
                    'Val Loss': f'{val_loss:.6f}',
                    'Status': status,
                    'Time': f'{epoch_time:.1f}s'
                })

            if patience_counter >= patience:
                if self.verbose:
                    pbar.close()
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True
        total_time = time.time() - start_time

        if self.verbose:
            print(f"Training completed: {len(self.train_losses)} epochs | {total_time:.1f}s | Best Val Loss: {best_val_loss:.6f}\n")

        return self

    def predict(self, data):
        """
        Make predictions on new data - Following official format

        Args:
            data: DataFrame with ALL features (including target for creating sequences)

        Returns:
            predictions: Array of predicted values
        """
        self.model.eval()
        predictions = []

        # Create dataset (uses unified data format)
        dataset = TimeSeriesDataset(data, seq_len=self.seq_len, pred_len=self.pred_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Decoder input (zeros for forecasting)
                dec_inp = torch.zeros((batch_x.shape[0], self.pred_len, self.n_features)).float().to(self.device)

                # Concatenate with label_len if needed
                if self.config.label_len > 0:
                    label_inp = torch.zeros((batch_x.shape[0], self.config.label_len, self.n_features)).float().to(self.device)
                    dec_inp = torch.cat([label_inp, dec_inp], dim=1)

                dec_y_mark = torch.zeros((batch_x.shape[0], self.config.label_len + self.pred_len, 3)).float().to(self.device)

                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, dec_y_mark)

                # Extract predictions (output is [batch, pred_len, c_out])
                # We want only the target column prediction
                predictions.append(outputs[:, :, 0].cpu().numpy())  # Get first channel (target)

        return np.concatenate(predictions, axis=0)

    def evaluate(self, data, target_col='next_day_high', set_name='Test'):
        """
        Evaluate model and compute metrics

        Returns:
            dict: Metrics (MAE, RMSE, MAPE, R�)
        """
        # Make predictions
        y_pred = self.predict(data)

        # Extract true target values
        if isinstance(data, pd.DataFrame):
            y_true = data[target_col].values
        else:
            # Assume first column is target if array
            y_true = data[:, 0]

        # Align predictions with targets (account for sequence length)
        y_true = y_true[self.seq_len:self.seq_len + len(y_pred)]
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        # Compute metrics (MAE, RMSE, R² valid in scaled space)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # MAPE is meaningless in scaled/log space (causes division by near-zero values)
        # Only compute MAPE in actual price space (see evaluate_test_set in predict.py)
        mape = None
        r2 = r2_score(y_true, y_pred)

        if mape is not None:
            print(f"  {set_name}: MAE={mae:.6f} | RMSE={rmse:.6f} | MAPE={mape:.2f}% | R²={r2:.4f}")
        else:
            print(f"  {set_name}: MAE={mae:.6f} | RMSE={rmse:.6f} | R²={r2:.4f}")

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'y_pred': y_pred,
            'y_true': y_true
        }

    def save_model(self, path, save_optimizer=False):
        """
        Save model weights for deployment/inference

        Args:
            path: Path to save model (use .pt extension for modern PyTorch)
            save_optimizer: If True, save optimizer state for resuming training

        Modern PyTorch Practice:
            - .pt is recommended (official PyTorch docs)
            - .pth is legacy but still widely used
            - Both work identically, .pt is cleaner
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
        }

        # Only save optimizer and training history if needed for resuming training
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['train_losses'] = self.train_losses
            checkpoint['val_losses'] = self.val_losses

        torch.save(checkpoint, path)

    def save_checkpoint(self, path):
        """
        Save full checkpoint for resuming training
        Includes optimizer state and training history
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"Checkpoint saved: {Path(path).name} ({size_mb:.1f}MB, can resume training)")

    def load_model(self, path):
        """
        Load model weights (and optionally optimizer state)

        Works with both:
        - Weights-only models (deployment)
        - Full checkpoints (resume training)
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Always load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state only if available (for resuming training)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            mode = "checkpoint"
        else:
            mode = "inference"

        # Load config if available
        if 'config' in checkpoint:
            # Config already set during initialization
            pass

    def score(self, data, target_col='next_day_high'):
        """
        Compute R² score on data (sklearn-style)

        Args:
            data: DataFrame with features and target
            target_col: Target column name

        Returns:
            float: R² score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling score()")

        # Make predictions
        y_pred = self.predict(data)

        # Extract true values
        if isinstance(data, pd.DataFrame):
            y_true = data[target_col].values
        else:
            y_true = data[:, 0]

        # Align predictions
        y_true = y_true[self.seq_len:self.seq_len + len(y_pred)]
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        return r2_score(y_true, y_pred)

    def get_params(self):
        """
        Get model parameters (sklearn-style)

        Returns:
            dict: Parameter dictionary
        """
        return {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'e_layers': self.e_layers,
            'top_k': self.top_k,
            'num_kernels': self.num_kernels,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """
        Set model parameters (sklearn-style)

        Note: Requires re-initialization of model if architecture params change

        Args:
            **params: Parameters to update

        Returns:
            self: Returns self for method chaining
        """
        # Update parameters
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

        # Re-initialize model if architecture changed
        if any(k in params for k in ['seq_len', 'pred_len', 'n_features',
                                      'd_model', 'd_ff', 'e_layers',
                                      'top_k', 'num_kernels', 'dropout']):
            self.config = Config(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                enc_in=self.n_features,
                c_out=1,
                d_model=self.d_model,
                d_ff=self.d_ff,
                e_layers=self.e_layers,
                top_k=self.top_k,
                num_kernels=self.num_kernels,
                dropout=self.dropout
            )
            self.model = TimesNet(self.config).float().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.is_fitted = False

        # Update optimizer learning rate if changed
        if 'learning_rate' in params:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        return self


# ============================================================================
# Model I/O Functions
# ============================================================================

def save_model(model, path):
    """
    Save model (auto-detects sklearn or PyTorch)

    Args:
        model: sklearn model or TimesNetRegressor
        path: Path to save model (.pkl for sklearn, .pt for PyTorch)
    """
    import os
    import pickle

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Detect model type
    if hasattr(model, 'save_model'):
        # TimesNetRegressor (PyTorch)
        model.save_model(path)
    else:
        # sklearn model
        with open(path, 'wb') as f:
            pickle.dump(model, f)


def load_model(path, model_type='auto', **kwargs):
    """
    Load model (auto-detects sklearn or PyTorch)

    Args:
        path: Path to saved model
        model_type: 'auto', 'sklearn', or 'pytorch' (default: 'auto')
        **kwargs: Additional arguments for PyTorch model loading
                 (n_features, seq_len, d_model, etc.)

    Returns:
        Loaded model
    """
    import pickle

    # Auto-detect by file extension
    if model_type == 'auto':
        if path.endswith('.pkl'):
            model_type = 'sklearn'
        elif path.endswith('.pt') or path.endswith('.pth'):
            model_type = 'pytorch'
        else:
            raise ValueError(f"Cannot auto-detect model type from extension: {path}")

    # Load based on type
    if model_type == 'sklearn':
        with open(path, 'rb') as f:
            return pickle.load(f)

    elif model_type == 'pytorch':
        # Load PyTorch model
        device = kwargs.get('device', 'auto')

        # Determine actual device for map_location
        if device == 'auto':
            actual_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            actual_device = device

        # Load checkpoint first to get saved configuration
        checkpoint = torch.load(path, map_location=actual_device, weights_only=False)

        # Build config: prioritize checkpoint, fallback to kwargs, then defaults
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            config = {
                'seq_len': saved_config.get('seq_len', kwargs.get('seq_len', 1)),
                'pred_len': 1,
                'n_features': saved_config.get('n_features', kwargs.get('n_features', 61)),
                'd_model': saved_config.get('d_model', kwargs.get('d_model', 64)),
                'd_ff': saved_config.get('d_ff', kwargs.get('d_ff', 128)),
                'e_layers': saved_config.get('e_layers', kwargs.get('e_layers', 2)),
                'num_kernels': saved_config.get('num_kernels', kwargs.get('num_kernels', 6)),
                'top_k': saved_config.get('top_k', kwargs.get('top_k', 1)),
                'dropout': saved_config.get('dropout', kwargs.get('dropout', 0.1)),
                'device': device,
                'verbose': False
            }
        else:
            # No saved config, use kwargs with defaults
            config = {
                'seq_len': kwargs.get('seq_len', 1),
                'pred_len': 1,
                'n_features': kwargs.get('n_features', 61),
                'd_model': kwargs.get('d_model', 64),
                'd_ff': kwargs.get('d_ff', 128),
                'e_layers': kwargs.get('e_layers', 2),
                'num_kernels': kwargs.get('num_kernels', 6),
                'top_k': kwargs.get('top_k', 1),
                'dropout': kwargs.get('dropout', 0.1),
                'device': device,
                'verbose': False
            }

        # Create model with determined configuration
        model = TimesNetRegressor(**config)

        # Load weights
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.is_fitted = True

        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

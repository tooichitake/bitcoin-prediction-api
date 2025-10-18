# PyTorch/Deep Learning Project Guidelines

## ⚠️ MANDATORY CODING STANDARDS ⚠️

**All code in this project MUST follow PyTorch and deep learning community standards.**

---

## 1. NAMING CONVENTIONS (PEP 8 + PyTorch Standards)

### Classes: PascalCase
```python
# ✅ CORRECT
class TimeSeriesDataset(Dataset):
class Config:
class Trainer:
class TimesNet(nn.Module):

# ❌ INCORRECT
class bitcoin_dataset:
class timesnet_config:
class BitcoinTimesNetTrainer:  # Too specific
```

**Standard Patterns:**
- Dataset classes: `{Purpose}Dataset` (e.g., `TimeSeriesDataset`, `ImageDataset`)
- Config classes: `Config` or `{Model}Config` (e.g., `BertConfig`)
- Trainer classes: `Trainer` (PyTorch Lightning standard)
- Model classes: `{ModelName}` (e.g., `TimesNet`, `ResNet`)

### Functions/Methods: snake_case
```python
# ✅ CORRECT
def train_model(...)
def load_checkpoint(...)
def compute_loss(...)
def forward(...)
def __init__(...)

# ❌ INCORRECT
def trainModel(...)
def LoadCheckpoint(...)
```

**PyTorch Standard Methods:**
```python
class Model(nn.Module):
    def __init__(self, ...)      # Constructor
    def forward(self, x)          # Forward pass (PyTorch convention)

class Trainer:
    def fit(self, ...)            # Keras/sklearn convention
    def train_epoch(self, ...)    # Training one epoch
    def validate(self, ...)       # Validation
    def predict(self, ...)        # Inference
    def evaluate(self, ...)       # Compute metrics
    def save_checkpoint(self, ...)
    def load_checkpoint(self, ...)
```

### Variables: snake_case
```python
# ✅ CORRECT
train_data = ...
val_data = ...
test_data = ...
learning_rate = 0.001
batch_size = 32
train_loader = DataLoader(...)
model = TimesNet(...)
optimizer = optim.Adam(...)
criterion = nn.MSELoss()

# ❌ INCORRECT
trainData = ...
TrainData = ...
TRAIN_DATA = ...  # Only for constants
```

### Constants: UPPER_SNAKE_CASE
```python
# ✅ CORRECT
MAX_SEQ_LEN = 1000
DEFAULT_LR = 0.001
NUM_CLASSES = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ❌ INCORRECT
max_seq_len = 1000  # Should be uppercase
MaxSeqLen = 1000
```

### Private Variables/Methods: _leading_underscore
```python
class Trainer:
    def __init__(self):
        self._cache = {}           # Private attribute
        self.model = None          # Public attribute

    def _compute_loss(self, ...):  # Private method
        pass

    def train(self, ...):          # Public method
        pass
```

---

## 2. HYPERPARAMETER NAMING (PyTorch/Transformer Standards)

### Configuration Object: Use `config` not `hyperparameters`
```python
# ✅ CORRECT (PyTorch/Hugging Face standard)
config = {
    # Sequence Configuration
    'seq_len': 30,              # Input sequence length
    'label_len': 0,             # Decoder start token length
    'pred_len': 1,              # Prediction length

    # Model Architecture (Transformer Standard)
    'd_model': 64,              # Model dimension (from "Attention is All You Need")
    'd_ff': 128,                # Feedforward dimension
    'n_heads': 8,               # Number of attention heads
    'e_layers': 2,              # Encoder layers
    'd_layers': 1,              # Decoder layers

    # Regularization
    'dropout': 0.1,             # Dropout probability

    # Optimization
    'lr': 0.001,                # Learning rate (or 'learning_rate')
    'batch_size': 32,

    # Training
    'train_epochs': 50,         # Number of epochs
    'patience': 10,             # Early stopping patience
}

# ❌ INCORRECT
hyperparameters = {
    'lookback_days': 30,        # Too domain-specific
    'hidden_dim': 64,           # Use d_model instead
    'feedforward_dim': 128,     # Use d_ff instead
    'num_layers': 2,            # Ambiguous (encoder or decoder?)
    'dropout_rate': 0.1,        # Use dropout instead
    'learning_rate': 0.001,     # Use lr (shorter)
    'max_epochs': 50,           # Use train_epochs
}
```

### PyTorch Parameter Name Mappings

| Our Code | PyTorch API | Source |
|----------|-------------|--------|
| `d_model` | `torch.nn.Transformer(d_model=...)` | "Attention is All You Need" |
| `d_ff` | `torch.nn.TransformerEncoder(dim_feedforward=...)` | Transformer standard |
| `n_heads` | `torch.nn.Transformer(nhead=...)` | PyTorch API |
| `e_layers` | `num_encoder_layers` | Time-Series-Library |
| `d_layers` | `num_decoder_layers` | Time-Series-Library |
| `dropout` | `nn.Dropout(p=dropout)` | PyTorch standard |
| `lr` | `optim.Adam(lr=...)` | PyTorch optimizer |
| `batch_size` | `DataLoader(batch_size=...)` | PyTorch DataLoader |

---

## 3. CODE STRUCTURE (PyTorch Project Layout)

### Standard Deep Learning Repository Structure
```
project_root/
├── data/                          # Data storage
│   ├── raw/                       # Original, immutable data
│   ├── interim/                   # Intermediate processed data
│   ├── processed/                 # Final, model-ready data
│   └── external/                  # External datasets
│
├── models/                        # Saved model checkpoints
│   └── {model_name}.pt           # Use .pt (modern) not .pth
│
├── notebooks/                     # Jupyter notebooks
│   └── experiment.ipynb
│
├── {project_name}/               # Source code (Python package)
│   ├── __init__.py
│   ├── data/                      # Data loading & processing
│   │   ├── __init__.py
│   │   └── dataset.py            # Dataset classes
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   └── timesnet.py
│   ├── training/                  # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── loss.py
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── config/                    # Configurations
│       ├── __init__.py
│       └── config.py
│
├── tests/                         # Unit tests
│   └── test_model.py
│
├── scripts/                       # Executable scripts
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── .gitignore
```

### Our Current Structure (Bitcoin Project)
```
36120-25SP-group4-AT3-experiment/
├── data/
│   ├── raw/                       # ✅ Original CSV files
│   ├── interim/                   # ✅ Intermediate data
│   └── processed/                 # ✅ Model-ready data
│
├── models/                        # ✅ Saved checkpoints (.pt)
│
├── figures/                       # ✅ Visualization outputs
│
├── notebooks/                     # ✅ Jupyter notebooks
│   └── 36120-25SP-AT3-group_4-25605217.ipynb
│
├── bitcoin/                       # ✅ Source package
│   ├── __init__.py
│   ├── features.py                # Data processing
│   ├── plots.py                   # Visualization
│   └── modeling/
│       ├── __init__.py
│       ├── models.py              # TimesNet architecture
│       ├── train.py               # Trainer class
│       └── predict.py             # Inference
│
├── requirements.txt
├── README.md
└── CLAUDE.md                      # ✅ This file
```

---

## 4. PYTORCH CODING PATTERNS

### Model Definition
```python
import torch
import torch.nn as nn

class TimesNet(nn.Module):
    def __init__(self, config):
        super().__init__()  # ✅ Modern Python 3 style
        self.config = config

        # Define layers
        self.encoder = nn.Linear(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # ✅ Use 'forward' not 'predict' for nn.Module
        x = self.encoder(x)
        x = self.dropout(x)
        return x
```

### Dataset Definition
```python
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # ✅ Required by PyTorch
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # ✅ Required by PyTorch
        # Return tensors, not numpy arrays
        return torch.FloatTensor(...)
```

### Training Loop
```python
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()  # ✅ Set to training mode
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()  # ✅ Clear gradients

        outputs = model(batch_x)  # ✅ Forward pass
        loss = criterion(outputs, batch_y)

        loss.backward()  # ✅ Backward pass
        optimizer.step()  # ✅ Update weights

        total_loss += loss.item()

    return total_loss / len(train_loader)
```

### Validation Loop
```python
@torch.no_grad()  # ✅ Disable gradient computation
def validate(model, val_loader, criterion, device):
    model.eval()  # ✅ Set to evaluation mode
    total_loss = 0

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item()

    return total_loss / len(val_loader)
```

### Saving/Loading Models
```python
# ✅ Save (modern .pt format)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, 'models/checkpoint.pt')

# ✅ Load
checkpoint = torch.load('models/checkpoint.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## 5. IMPORT ORGANIZATION (PEP 8)

```python
# Standard library imports
import os
import sys
import time
from pathlib import Path

# Third-party imports (alphabetical)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Local imports
from bitcoin.modeling.models import TimesNet
from bitcoin.utils.metrics import compute_metrics
```

**Order:**
1. Standard library
2. Third-party packages (alphabetical)
3. Local application imports

---

## 6. DOCUMENTATION STANDARDS

### Docstrings (Google Style)
```python
def train_model(train_data, val_data, config):
    """
    Train TimesNet model on time series data.

    Args:
        train_data (pd.DataFrame): Training dataset with all features
        val_data (pd.DataFrame): Validation dataset
        config (dict): Model configuration with hyperparameters

    Returns:
        Trainer: Trained model instance
        dict: Training history with losses and metrics

    Example:
        >>> config = {'d_model': 64, 'lr': 0.001}
        >>> trainer, results = train_model(train_df, val_df, config)
    """
    pass
```

### Inline Comments
```python
# ✅ GOOD: Explain WHY, not WHAT
# Use FFT to discover dominant periodicities (TimesNet paper approach)
freq_domain = torch.fft.rfft(x, dim=1)

# ❌ BAD: State the obvious
# Convert x to frequency domain using FFT
freq_domain = torch.fft.rfft(x, dim=1)
```

---

## 7. TYPE HINTS (Python 3.6+)

```python
from typing import Tuple, Dict, Optional
import torch
import pandas as pd

def train_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: Dict[str, any],
    device: str = 'cuda'
) -> Tuple[object, Dict[str, list]]:
    """Train model with type hints."""
    pass
```

---

## 8. ERROR HANDLING

```python
# ✅ Specific exceptions
try:
    checkpoint = torch.load(path)
except FileNotFoundError:
    raise FileNotFoundError(f"Checkpoint not found: {path}")
except RuntimeError as e:
    raise RuntimeError(f"Failed to load checkpoint: {e}")

# ❌ Bare except
try:
    checkpoint = torch.load(path)
except:  # BAD: Too broad
    pass
```

---

## 9. LOGGING (Not Print Statements)

For production code, use logging:
```python
import logging

logger = logging.getLogger(__name__)

# ✅ Use logging
logger.info(f"Training started with lr={config['lr']}")
logger.warning(f"Validation loss increased: {val_loss:.4f}")
logger.error(f"Failed to save checkpoint: {e}")

# ❌ Don't use print for important messages
print(f"Training started...")  # Only for notebooks/scripts
```

For our notebooks, print is acceptable.

---

## 10. CONFIGURATION MANAGEMENT

### Use Config Classes (Recommended for Large Projects)
```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    seq_len: int = 30
    pred_len: int = 1
    d_model: int = 64
    d_ff: int = 128
    e_layers: int = 2
    dropout: float = 0.1

@dataclass
class TrainConfig:
    lr: float = 0.001
    batch_size: int = 32
    train_epochs: int = 50
    patience: int = 10
```

### Or Use Dict (Simpler, What We Use)
```python
config = {
    'seq_len': 30,
    'd_model': 64,
    'lr': 0.001,
    'batch_size': 32,
}
```

---

## 11. TESTING STANDARDS

```python
import pytest
import torch

def test_model_forward_pass():
    """Test model forward pass with known input shape."""
    model = TimesNet(config)
    x = torch.randn(32, 30, 64)  # (batch, seq, features)

    output = model(x)

    assert output.shape == (32, 1, 1)  # (batch, pred_len, output_dim)
    assert not torch.isnan(output).any()
```

---

## 12. VERSION CONTROL

### .gitignore (Essential Patterns)
```
# Data
data/raw/*
data/interim/*
data/processed/*
!data/raw/.gitkeep

# Models
models/*.pt
models/*.pth

# Python
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/

# Environment
.env
venv/
.conda/

# IDE
.vscode/
.idea/
*.swp
```

---

## 13. DEPENDENCIES (requirements.txt)

```txt
# Deep Learning
torch>=2.0.0
numpy>=1.24.0

# Data Processing
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
```

**Pin versions for reproducibility:**
```txt
torch==2.0.1
numpy==1.24.3
pandas==2.0.3
```

---

## 14. CODE QUALITY TOOLS (Optional but Recommended)

### Black (Code Formatter)
```bash
pip install black
black bitcoin/
```

### isort (Import Sorter)
```bash
pip install isort
isort bitcoin/
```

### flake8 (Linter)
```bash
pip install flake8
flake8 bitcoin/
```

---

## 15. ACADEMIC WRITING STANDARDS

### Hyperparameter Explanations (Use Academic Language)
```python
"""
Hyperparameter selection balances model capacity and generalization:

**d_model=64 (Model Dimension)**
Following Transformer architecture conventions (Vaswani et al., 2017),
d_model determines the dimensionality of token embeddings and hidden states.
64 dimensions provide sufficient representational capacity for 61 input
features without excessive parameters that would cause overfitting on
2,071 training samples.

**Reference:**
Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
```

---

## 16. NOTEBOOK WRITING STANDARDS

### Cell Descriptions (Markdown Cells)

**MANDATORY RULES:**
- ✅ Write in English only
- ✅ Maximum 50 words per description
- ✅ Use flowing narrative prose (human-readable)
- ✅ NO bullet points or numbered lists in descriptions
- ✅ Explain the purpose and rationale

**Example - CORRECT:**
```markdown
This section analyzes the correlation between Bitcoin's technical indicators and next-day high price.
Strong correlations (>0.9) suggest these features provide predictive signals, while weaker correlations
indicate features that capture complementary market dynamics.
```

**Example - INCORRECT:**
```markdown
Correlation Analysis:
- Calculate correlations between features and target
- Identify strongly correlated features (>0.9)
- Remove weakly correlated features
- Visualize correlation matrix
```

### Code Comments

```python
# ✅ GOOD: Narrative explanation
# Calculate correlation matrix to identify which technical indicators
# have strong linear relationships with next-day high price
correlations = df.corr()['next_day_high'].sort_values(ascending=False)

# ❌ BAD: Bullet-point style
# - Get correlations
# - Sort by values
# - Display results
correlations = df.corr()['next_day_high'].sort_values(ascending=False)
```

### Section Descriptions

```markdown
✅ CORRECT:
The model achieved strong performance with R²=0.97 on the test set, indicating
it captures Bitcoin's price dynamics effectively. The low MAPE of 0.98% suggests
predictions are accurate enough for practical trading applications.

❌ INCORRECT:
Model Performance:
- R² = 0.97 (excellent)
- MAPE = 0.98%
- MAE = 0.009
- Predictions are accurate
```

---

## QUICK REFERENCE CHECKLIST

### Before Committing Code:
- [ ] Class names: PascalCase (e.g., `Trainer`, `Config`)
- [ ] Function names: snake_case (e.g., `train_model`)
- [ ] Variable names: snake_case (e.g., `train_data`)
- [ ] Use `config` not `hyperparameters`
- [ ] Use PyTorch standard parameters (`d_model`, `d_ff`, `lr`, etc.)
- [ ] Imports organized (stdlib → third-party → local)
- [ ] Docstrings for all public functions/classes
- [ ] No bare `except:` clauses
- [ ] Save models as `.pt` not `.pth`
- [ ] Type hints where appropriate
- [ ] Remove debug print statements (or use logging)

### File Structure:
- [ ] Code in modules, not in notebook cells
- [ ] Data in `data/{raw,interim,processed}/`
- [ ] Models in `models/`
- [ ] Notebooks in `notebooks/`
- [ ] Source code in `{project_name}/`

---

## REFERENCES

1. **PEP 8**: https://peps.python.org/pep-0008/
2. **PyTorch Documentation**: https://pytorch.org/docs/
3. **Hugging Face Transformers**: https://github.com/huggingface/transformers
4. **PyTorch Lightning**: https://lightning.ai/docs/pytorch/
5. **Time-Series-Library**: https://github.com/thuml/Time-Series-Library
6. **"Attention is All You Need"**: Vaswani et al., 2017

---

**Last Updated:** 2025-01-08
**Maintained By:** Claude Code Assistant
**Project:** Bitcoin Price Forecasting with TimesNet

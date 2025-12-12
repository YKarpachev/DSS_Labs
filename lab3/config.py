# config.py

import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "nvda_15m_clean.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "nvda_lstm_dss.pt"

# Create dirs if needed
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Device (your 5090 / 9950 will be picked automatically)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training / model hyperparameters
SEQ_LEN = 64          # past candles per sample
HORIZON = 5           # future closes to predict
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Loss mixing
CLS_WEIGHT = 0.5      # weight for classification loss vs regression
NUM_CLASSES = 3       # bear / neutral / bull

# Momentum thresholds for labeling
BEAR_THRESHOLD = -0.004   # -0.4%
BULL_THRESHOLD = 0.004    # +0.4%

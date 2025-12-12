# model.py

import torch.nn as nn

from config import HORIZON, NUM_CLASSES


class LSTMDSS(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 horizon=HORIZON, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]          # [B, H]
        reg_out = self.reg_head(h_last) # [B, horizon]
        cls_logits = self.cls_head(h_last)  # [B, num_classes]
        return reg_out, cls_logits

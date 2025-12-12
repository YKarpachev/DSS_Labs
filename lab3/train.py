# train.py

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import joblib

from config import (
    DEVICE, LR, EPOCHS, GRAD_CLIP, WEIGHT_DECAY,
    CLS_WEIGHT, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, MODEL_PATH
)
from data import prepare_data
from model import LSTMDSS


def train():
    (train_loader, val_loader, test_loader,
     scaler, feature_cols, df_full) = prepare_data()

    input_size = len(feature_cols)
    model = LSTMDSS(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    best_val_loss = np.inf
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # ----- Train -----
        model.train()
        total_loss = 0.0
        for X, y_reg, y_cls in train_loader:
            X = X.to(DEVICE)
            y_reg = y_reg.to(DEVICE)
            y_cls = y_cls.to(DEVICE)

            optimizer.zero_grad()
            pred_reg, logits = model(X)
            loss_reg = mse_loss(pred_reg, y_reg)
            loss_cls = ce_loss(logits, y_cls)
            loss = loss_reg + CLS_WEIGHT * loss_cls
            loss.backward()
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        correct_cls = 0
        total_cls = 0

        with torch.no_grad():
            for X, y_reg, y_cls in val_loader:
                X = X.to(DEVICE)
                y_reg = y_reg.to(DEVICE)
                y_cls = y_cls.to(DEVICE)

                pred_reg, logits = model(X)
                loss_reg = mse_loss(pred_reg, y_reg)
                loss_cls = ce_loss(logits, y_cls)
                loss = loss_reg + CLS_WEIGHT * loss_cls
                val_loss += loss.item() * X.size(0)

                preds_cls = logits.argmax(dim=1)
                correct_cls += (preds_cls == y_cls).sum().item()
                total_cls += y_cls.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_cls / total_cls if total_cls > 0 else 0.0

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"val_acc={val_acc:.3f}"
        )

        # Early stopping checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- Final test MSE -----
    model.eval()
    mse_accum = 0.0
    n_samples = 0
    with torch.no_grad():
        for X, y_reg, _ in test_loader:
            X = X.to(DEVICE)
            y_reg = y_reg.to(DEVICE)
            pred_reg, _ = model(X)
            mse_accum += mse_loss(pred_reg, y_reg).item() * X.size(0)
            n_samples += X.size(0)

    test_mse = mse_accum / n_samples
    print(f"Test MSE (5-step closes): {test_mse:.6f}")

    # ----- Save model + scaler + metadata -----
    joblib.dump(
        {"scaler": scaler, "feature_cols": feature_cols},
        MODEL_PATH.with_suffix(".meta.pkl")
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "test_mse": test_mse,
        },
        MODEL_PATH
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler + feature_cols to {MODEL_PATH.with_suffix('.meta.pkl')}")


if __name__ == "__main__":
    train()

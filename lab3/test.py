# test.py

import random
import numpy as np
import torch
import joblib

from config import DEVICE, MODEL_PATH
from data import prepare_data
from model import LSTMDSS


def main():
    # Build data (same pipeline as training)
    train_loader, val_loader, test_loader, scaler, feature_cols, df = prepare_data()
    test_ds = test_loader.dataset

    # Load model + meta
    meta = joblib.load(MODEL_PATH.with_suffix(".meta.pkl"))
    saved_feature_cols = meta["feature_cols"]

    if len(saved_feature_cols) != len(feature_cols):
        print("Warning: feature_cols length mismatch between saved meta and current code.")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = LSTMDSS(
        input_size=len(saved_feature_cols),
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n = len(test_ds)
    k = min(50, n)  # in case test set is smaller
    random.seed(42)
    indices = random.sample(range(n), k)

    correct = 0

    with torch.no_grad():
        for idx in indices:
            X, _, y_cls = test_ds[idx]  # we ignore regression target here
            X = torch.tensor(X, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            y_cls = int(y_cls)

            _, logits = model(X)
            pred_cls = int(logits.argmax(dim=1).item())

            if pred_cls == y_cls:
                correct += 1

    acc = correct / k
    print(f"Random 50-sample momentum accuracy: {acc*100:.2f}% ({correct}/{k})")


if __name__ == "__main__":
    main()

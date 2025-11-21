import numpy as np
import torch
from torch.utils.data import DataLoader
import random

from nvda_dss_lstm import (
    CSV_PATH,
    SEQ_LEN,
    PRED_HORIZON,
    DEVICE,
    load_and_engineer,
    make_splits,
    SeqDataset,
    LSTMForecaster,
)

MODEL_PATH = "nvda_lstm_dss.pt"


def evaluate_directional(model_path=MODEL_PATH, n_random_examples=100):
    # 1) Load data & rebuild splits exactly like in training
    X, y, times, feature_cols = load_and_engineer(CSV_PATH)
    X_scaled, y, scaler, train_idx, val_idx, test_idx = make_splits(
        X, y, times, SEQ_LEN, PRED_HORIZON
    )

    n_features = X_scaled.shape[1]
    test_ds = SeqDataset(X_scaled, y, test_idx, SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    print(f"Test samples: {len(test_ds)}")

    # 2) Load trained model + weights
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = LSTMForecaster(
        n_features=n_features,
        hidden_size=128,
        num_layers=2,
        pred_horizon=PRED_HORIZON,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 3) Run predictions over test set
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(yb.numpy())

    preds = np.vstack(all_preds)
    trues = np.vstack(all_trues)

    # 4) MSE vs zero baseline
    mse_model = np.mean((preds - trues) ** 2)
    mse_zero = np.mean(trues ** 2)

    print("\nMSE comparison (on test set):")
    print(f"  Model MSE: {mse_model:.8f}")
    print(f"  Zero-return baseline MSE: {mse_zero:.8f}")

    sign_true = np.sign(trues)  # -1, 0, +1
    sign_pred = np.sign(preds)

    step_acc = (sign_true == sign_pred).mean(axis=0)

    print("\nDirectional accuracy per step (horizon 1..{}):".format(PRED_HORIZON))
    for h, acc in enumerate(step_acc, start=1):
        print(f"  Horizon {h}: {acc:.3f}")

    # 6) Overall 5-candle direction accuracy (sum of returns)
    sum_true = trues.sum(axis=1)
    sum_pred = preds.sum(axis=1)

    dir_true = np.sign(sum_true)
    dir_pred = np.sign(sum_pred)

    overall_dir_acc = (dir_true == dir_pred).mean()
    print(f"\nOverall 5-candle direction accuracy: {overall_dir_acc:.3f}")

    # 7) Baseline: always predict "up" for 5-candle sum
    nonzero_mask = sum_true != 0
    baseline_dir = np.ones_like(sum_true)
    baseline_acc = (baseline_dir[nonzero_mask] == np.sign(sum_true[nonzero_mask])).mean()

    print(f"Baseline (always up) 5-candle direction accuracy: {baseline_acc:.3f}")

    # 8) Show random test examples
    n_test = len(test_idx)
    n_show = min(n_random_examples, n_test)

    print(f"\nShowing {n_show} random test examples:\n")

    random.seed(42)
    for k in random.sample(range(n_test), n_show):
        t_idx = test_idx[k]
        dt = times[t_idx]

        true_vec = trues[k]
        pred_vec = preds[k]

        true_sum = sum_true[k]
        pred_sum = sum_pred[k]

        print(f"Example #{k} | target start time: {dt}")
        print("  True 5-step log returns:   ", np.round(true_vec, 6))
        print("  Pred 5-step log returns:   ", np.round(pred_vec, 6))
        print("  True sum: {:.6f}, Pred sum: {:.6f}".format(true_sum, pred_sum))
        print("  True dir: {:+d}, Pred dir: {:+d}".format(int(np.sign(true_sum)), int(np.sign(pred_sum))))
        print("-" * 60)


if __name__ == "__main__":
    evaluate_directional()

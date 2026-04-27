"""
train_clf_bml.py - Train CNN+GRU classifier on BatteryML-derived features.

How to run:
    python train_clf_bml.py --content_dir ./content_bml --output_dir ./checkpoints_clf_bml

Train one BML family/subfolder only:
    python train_clf_bml.py --content_dir ./content_bml/CALB --output_dir ./checkpoints_clf_bml_CALB

Main outputs:
    checkpoints_clf_bml/best_clf_bml.pt
    checkpoints_clf_bml/dq_scaler_bml.pkl
    checkpoints_clf_bml/summary_scaler_bml.pkl
    checkpoints_clf_bml/clf_pred_bml.npy
    checkpoints_clf_bml/clf_true_bml.npy
"""

import argparse
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset_clf_bml import N_CLASSES, build_clf_dataloaders
from train_clf import BatteryRULClassifier, OrdinalLoss, evaluate, predict_cls, train_epoch


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading BML data...")
    train_loader, val_loader, test_loader, scalers = build_clf_dataloaders(
        content_dir=args.content_dir,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    dq_scaler, summary_scaler = scalers
    joblib.dump(dq_scaler, os.path.join(args.output_dir, "dq_scaler_bml.pkl"))
    joblib.dump(summary_scaler, os.path.join(args.output_dir, "summary_scaler_bml.pkl"))

    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    model = BatteryRULClassifier(
        cnn_dim=args.cnn_dim,
        gru_dim=args.gru_dim,
        gru_layers=args.gru_layers,
        summary_feats=summary_feats,
        n_classes=N_CLASSES,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    if args.loss == "ordinal":
        criterion = OrdinalLoss(n_classes=N_CLASSES)
        pred_fn = predict_cls
        print("Loss: OrdinalLoss\n")
    else:
        criterion = nn.CrossEntropyLoss()
        pred_fn = lambda logits: logits.argmax(dim=1)
        print("Loss: CrossEntropyLoss\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = -1.0
    best_path = os.path.join(args.output_dir, "best_clf_bml.pt")

    header = f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, pred_fn, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, pred_fn, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
            f"{va_loss:8.4f} | {va_acc:7.4f} | {lr:.2e}"
        )

        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    print("\n" + "=" * 60)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion, pred_fn, device)

    print(f"\nTest Loss: {te_loss:.4f}  Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            true,
            pred,
            target_names=["RUL>400", "RUL>300", "RUL>200", "RUL>100", "RUL<100"],
            zero_division=0,
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(true, pred))

    np.save(os.path.join(args.output_dir, "clf_pred_bml.npy"), pred)
    np.save(os.path.join(args.output_dir, "clf_true_bml.npy"), true)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", default="./content_bml")
    parser.add_argument("--output_dir", default="./checkpoints_clf_bml")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=600)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cnn_dim", type=int, default=32)
    parser.add_argument("--gru_dim", type=int, default=32)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss",
        default="ordinal",
        choices=["cross_entropy", "ordinal"],
        help="Loss function: ordinal or cross_entropy",
    )
    args = parser.parse_args()

    train(args)

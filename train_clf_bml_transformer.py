"""
train_clf_bml_transformer.py - Train Transformer classifier on BatteryML-derived features.

How to run:
    python train_clf_bml_transformer.py --content_dir ./content_bml --output_dir ./checkpoints_clf_bml_tf

Train one BML family/subfolder only:
    python train_clf_bml_transformer.py --content_dir ./content_bml/CALB --output_dir ./checkpoints_clf_bml_CALB_tf

Main outputs:
    checkpoints_clf_bml_tf/best_clf_bml.pt
    checkpoints_clf_bml_tf/dq_scaler_bml.pkl
    checkpoints_clf_bml_tf/summary_scaler_bml.pkl
    checkpoints_clf_bml_tf/clf_pred_bml.npy
    checkpoints_clf_bml_tf/clf_true_bml.npy
"""

import argparse
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset_clf_bml import N_CLASSES, N_INPUT, build_clf_dataloaders
from train_clf_transformer import BatteryRULClassifier, evaluate, train_epoch  # transformer model
from train_clf import OrdinalLoss, predict_cls                                  # shared utilities


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
    joblib.dump(dq_scaler,      os.path.join(args.output_dir, "dq_scaler_bml.pkl"))
    joblib.dump(summary_scaler, os.path.join(args.output_dir, "summary_scaler_bml.pkl"))

    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    model = BatteryRULClassifier(
        cnn_dim      = args.cnn_dim,
        d_model      = args.d_model,
        n_heads      = args.n_heads,
        n_enc_layers = args.n_enc_layers,
        d_ff         = args.d_ff,
        summary_feats= summary_feats,
        n_classes    = N_CLASSES,
        dropout      = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── Layer-by-layer summary ────────────────────────────────────────────────
    print("Model Summary:")
    print("=" * 80)
    print(f"  {'Layer':<38} {'Output Shape':<25} {'Params':>10}")
    print("-" * 80)

    _handles = []
    _summary = []

    def _hook(module, inp, out):
        if len(list(module.children())) == 0:          # leaf modules only
            n = sum(p.numel() for p in module.parameters())
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else "?"
            _summary.append((module.__class__.__name__, shape, n))

    for m in model.modules():
        _handles.append(m.register_forward_hook(_hook))

    from gen_features_bml import V_BINS as _V
    _dummy_dq      = torch.zeros(1, N_INPUT, 1, _V,         device=device)
    _dummy_summary = torch.zeros(1, N_INPUT, summary_feats, device=device)
    with torch.no_grad():
        model(_dummy_dq, _dummy_summary)

    for h in _handles:
        h.remove()

    for name, shape, n in _summary:
        print(f"  {name:<38} {str(shape):<25} {n:>10,}")

    print("=" * 80)
    print(f"  {'Total trainable parameters':<38} {'':25} {n_params:>10,}")
    print("=" * 80 + "\n")

    if args.loss == "ordinal":
        criterion = OrdinalLoss(n_classes=N_CLASSES)
        pred_fn   = predict_cls
        print("Loss: OrdinalLoss\n")
    else:
        criterion = nn.CrossEntropyLoss()
        pred_fn   = lambda logits: logits.argmax(dim=1)
        print("Loss: CrossEntropyLoss\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = -1.0
    best_path    = os.path.join(args.output_dir, "best_clf_bml.pt")

    header = f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc       = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
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
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion, device)

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
    parser.add_argument("--content_dir",  default="./content_bml")
    parser.add_argument("--output_dir",   default="./checkpoints_clf_bml_tf")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--n_samples",    type=int,   default=600)
    parser.add_argument("--val_ratio",    type=float, default=0.1)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--cnn_dim",      type=int,   default=32)
    parser.add_argument("--d_model",      type=int,   default=32)
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--n_enc_layers", type=int,   default=2)
    parser.add_argument("--d_ff",         type=int,   default=32)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--loss",
        default="ordinal",
        choices=["cross_entropy", "ordinal"],
        help="Loss function: ordinal or cross_entropy",
    )
    args = parser.parse_args()

    train(args)

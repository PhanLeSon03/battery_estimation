"""
train_clf.py — Train CNN+GRU classifier for battery RUL classification

Classes:
    0: RUL > 400
    1: RUL 300–400
    2: RUL 200–300
    3: RUL 100–200
    4: RUL < 100

Usage:
    python train_clf.py --content_dir ./content --output_dir ./checkpoints_clf
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset_clf import build_clf_dataloaders, N_CLASSES, N_INPUT
import joblib


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
class BatteryRULClassifier(nn.Module):
    def __init__(
        self,
        cnn_dim:       int   = 32,
        gru_dim:       int   = 32,
        gru_layers:    int   = 2,
        summary_feats: int   = 16,
        n_classes:     int   = N_CLASSES,
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16),  nn.GELU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(32, cnn_dim, kernel_size=5, padding=2), nn.BatchNorm1d(cnn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        self.summary_proj = nn.Sequential(
            nn.Linear(summary_feats, cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size    = cnn_dim * 2,
            hidden_size   = gru_dim,
            num_layers    = gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout        = dropout if gru_layers > 1 else 0.0,
        )

        self.post_gru_drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(gru_dim * 2, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, dq: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        B, T, _ = dq.shape

        dq_feat      = self.cnn(dq.reshape(B * T, 1, -1)).squeeze(-1).reshape(B, T, -1)
        summary_feat = self.summary_proj(summary)

        fused        = self.post_gru_drop(torch.cat([dq_feat, summary_feat], dim=-1))
        _, h_n       = self.gru(fused)
        h_last       = self.post_gru_drop(torch.cat([h_n[-2], h_n[-1]], dim=-1))

        return self.head(h_last)

# -------------------------------------------------------------------------
# Train one epoch
# -------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in loader:
        dq      = batch["dq"].to(device)
        summary = batch["summary"].to(device)
        labels  = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(dq, summary)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


# -------------------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred   = []
    all_true   = []

    for batch in loader:
        dq      = batch["dq"].to(device)
        summary = batch["summary"].to(device)
        labels  = batch["label"].to(device)

        logits = model(dq, summary)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        all_pred.extend(logits.argmax(dim=1).cpu().numpy())
        all_true.extend(labels.cpu().numpy())

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    acc      = (all_pred == all_true).mean()

    return total_loss / len(all_true), acc, all_pred, all_true


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading data...")
    train_loader, val_loader, test_loader, scalers = build_clf_dataloaders(
        content_dir = args.content_dir,
        batch_size  = args.batch_size,
        n_samples   = args.n_samples,
        val_ratio   = 0.1,
        num_workers = args.num_workers,
    )
    
    dq_scaler, summary_scaler = scalers
    # save right after building train dataset
    joblib.dump(dq_scaler,      os.path.join(args.output_dir, "dq_scaler.pkl"))
    joblib.dump(summary_scaler, os.path.join(args.output_dir, "summary_scaler.pkl"))

    model = BatteryRULClassifier(
        cnn_dim       = args.cnn_dim,
        gru_dim       = args.gru_dim,
        gru_layers    = args.gru_layers,
        summary_feats = 16,
        n_classes     = N_CLASSES,
        dropout       = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = 0.0

    header = f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc          = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _    = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
              f"{va_loss:8.4f} | {va_acc:7.4f} | {lr:.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_clf.pt"))

    # ---- Test ----
    print("\n" + "=" * 60)
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_clf.pt"), map_location=device)
    )
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {te_loss:.4f}  Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true, pred,
          target_names=["RUL>400", "RUL>300", "RUL>200", "RUL>100", "RUL<100"]))
    print("Confusion Matrix:")
    print(confusion_matrix(true, pred))

    np.save(os.path.join(args.output_dir, "clf_pred.npy"), pred)
    np.save(os.path.join(args.output_dir, "clf_true.npy"), true)
    print(f"\nSaved to {args.output_dir}/")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", default="./content")
    parser.add_argument("--output_dir",  default="./checkpoints_clf")
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--n_samples",   type=int,   default=600)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--cnn_dim",     type=int,   default=32)
    parser.add_argument("--gru_dim",     type=int,   default=32)
    parser.add_argument("--gru_layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=0)
    args = parser.parse_args()

    train(args)

"""
train_clf_transformer.py — Train Transformer classifier for battery RUL classification

Classes:
    0: RUL > 400
    1: RUL 300–400
    2: RUL 200–300
    3: RUL 100–200
    4: RUL < 100

Usage:
    python train_clf_transformer.py --content_dir ./content --output_dir ./checkpoints_clf_tf
"""

import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset_clf import build_clf_dataloaders, N_CLASSES, N_INPUT
import joblib

warnings.filterwarnings("ignore", message="enable_nested_tensor")


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
class BatteryRULClassifier(nn.Module):
    """
    Transformer encoder classifier for battery RUL.

    Input:
        dq:      (B, 32, 1000)   — delta-Qdlin curves
        summary: (B, 32, 16)     — per-cycle scalar features + PE

    Architecture:
        dq      → time-distributed Conv1D → (B, 32, cnn_dim)
        summary → Linear               → (B, 32, cnn_dim)
        concat                          → (B, 32, cnn_dim*2)
        Linear projection               → (B, 32, d_model)
        Positional encoding (sinusoidal)
        TransformerEncoder (n_layers × [MHA + FFN])  → (B, 32, d_model)
        CLS token / mean pool           → (B, d_model)
        MLP head                        → (B, N_CLASSES)

    Output:
        logits: (B, N_CLASSES)
    """

    def __init__(
        self,
        cnn_dim:       int   = 32,
        d_model:       int   = 128,
        n_heads:       int   = 4,
        n_enc_layers:  int   = 3,
        d_ff:          int   = 256,
        summary_feats: int   = 18,
        n_classes:     int   = N_CLASSES,
        dropout:       float = 0.1,
        max_seq_len:   int   = 64,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # ── Time-distributed 1D CNN ──────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32),  nn.GELU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(64, cnn_dim, kernel_size=5, padding=2), nn.BatchNorm1d(cnn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )


        # ── Summary projection ───────────────────────────────────────────
        self.summary_proj = nn.Sequential(
            nn.Linear(summary_feats, cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Input projection to d_model ──────────────────────────────────
        self.input_proj = nn.Linear(cnn_dim + cnn_dim, d_model)


        # ── Transformer encoder ──────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,    # Pre-LN (more stable)
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers          = n_enc_layers,
            norm                = nn.LayerNorm(d_model),
            enable_nested_tensor= False,
        )

        # ── Classification head ──────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    @staticmethod
    def _build_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, dq: torch.Tensor,  summary: torch.Tensor) -> torch.Tensor:
        B, T, C, F  = dq.shape

        # Time-distributed CNN on dQ curves
        dq_feat      = self.cnn(dq.reshape(B * T, 1, -1)).squeeze(-1).reshape(B, T, -1)  # (B, T, cnn_dim)

        summary_feat = self.summary_proj(summary)                                          # (B, T, cnn_dim)

        # Fuse and project to d_model
        fused = self.dropout(torch.cat([dq_feat, summary_feat], dim=-1))   # (B, T, cnn_dim*2)
        x     = self.input_proj(fused)                                      # (B, T, d_model)


        # Transformer encoder
        x = self.encoder(x)                                                 # (B, T, d_model)

        cls_out = x.mean(dim=1)                                             # (B, d_model)

        return self.head(cls_out)                                           # (B, n_classes)


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
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        val_ratio   = 0.2,
        num_workers = args.num_workers,
    )

    dq_scaler, summary_scaler = scalers
    joblib.dump(dq_scaler,      os.path.join(args.output_dir, "dq_scaler.pkl"))
    joblib.dump(summary_scaler, os.path.join(args.output_dir, "summary_scaler.pkl"))
    print("Scalers saved.")
    
    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}") 

    model = BatteryRULClassifier(
        cnn_dim       = args.cnn_dim,
        d_model       = args.d_model,
        n_heads       = args.n_heads,
        n_enc_layers  = args.n_enc_layers,
        d_ff          = args.d_ff,
        summary_feats = summary_feats,
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
        tr_loss, tr_acc       = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
              f"{va_loss:8.4f} | {va_acc:7.4f} | {lr:.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_clf.pt"))

    # ── Test ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_clf.pt"), map_location=device)
    )
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {te_loss:.4f}  Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true, pred,
          labels=list(range(N_CLASSES)),
          target_names=["RUL>400", "RUL>300", "RUL>200", "RUL>100", "RUL<100"],
          zero_division=0))
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
    parser.add_argument("--content_dir",  default="./content")
    parser.add_argument("--output_dir",   default="./checkpoints_clf_tf")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--n_samples",    type=int,   default=600)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--cnn_dim",      type=int,   default=128)
    parser.add_argument("--d_model",      type=int,   default=128)
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--n_enc_layers", type=int,   default=2)
    parser.add_argument("--d_ff",         type=int,   default=128)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--num_workers",  type=int,   default=0)
    args = parser.parse_args()

    train(args)

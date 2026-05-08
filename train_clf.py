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
    python train_clf.py --content_dir ./content_bml --output_dir ./checkpoints_clf_bml
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


class OrdinalLoss(nn.Module):
    """
    Takes raw logits (B, 5) — same as CrossEntropyLoss.
    Converts to cumulative probabilities internally using softmax.

    P(y > k) = sum_{j=k+1}^{K-1} softmax(logits)_j
    """
    def __init__(self, n_classes: int = 5, reduction: str = 'mean'):
        super().__init__()
        self.K         = n_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)                          # (B, 5)
        cum   = 1.0 - torch.cumsum(probs, dim=1)[:, :-1]             # (B, K-1)
        thresholds = torch.arange(self.K - 1, device=targets.device)
        labels     = (targets.unsqueeze(1) > thresholds).float()     # (B, K-1)
        loss = F.binary_cross_entropy(cum.clamp(1e-7, 1 - 1e-7), labels, reduction='none')
        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def predict_cls(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return probs.argmax(dim=-1)

def ordinal_predict(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    cum   = 1.0 - torch.cumsum(probs, dim=1)[:, :-1]   # (B, 4)
    return (cum > 0.5).sum(dim=1).long()                # (B,)



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
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size    = cnn_dim + cnn_dim,      # 2 channels × cnn_dim each + summary proj cnn_dim
            hidden_size   = gru_dim,
            num_layers    = gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if gru_layers > 1 else 0.0,
        )

        self.post_gru_drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(gru_dim * 2, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, dq: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        B, T, C, F = dq.shape                                       

        dq_feat      = self.cnn(dq.reshape(B * T, C, F)).squeeze(-1).reshape(B, T, -1)
        summary_feat = self.summary_proj(summary)

        fused    = self.post_gru_drop(torch.cat([dq_feat, summary_feat], dim=-1))
        out, h_n = self.gru(fused)
        h_last   = self.post_gru_drop(torch.cat([h_n[-2], h_n[-1]], dim=-1))

        return self.head(h_last)


# -------------------------------------------------------------------------
# Train one epoch
# -------------------------------------------------------------------------
def train_epoch(model, loader, criterion, pred_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for batch in loader:
        dq      = batch["dq"].to(device)
        summary = batch["summary"].to(device)
        labels  = batch["label"].to(device)
        optimizer.zero_grad()
        out  = model(dq, summary)
        loss = criterion(out, labels)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (pred_fn(out) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


# -------------------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, pred_fn, device):
    model.eval()
    total_loss = 0.0
    all_pred   = []
    all_true   = []
    for batch in loader:
        dq      = batch["dq"].to(device)
        summary = batch["summary"].to(device)
        labels  = batch["label"].to(device)
        out     = model(dq, summary)
        loss    = criterion(out, labels)
        total_loss += loss.item() * len(labels)
        all_pred.extend(pred_fn(out).cpu().numpy())
        all_true.extend(labels.cpu().numpy())
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    return total_loss / len(all_true), (all_pred == all_true).mean(), all_pred, all_true


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
    joblib.dump(dq_scaler,      os.path.join(args.output_dir, "dq_scaler.pkl"))
    joblib.dump(summary_scaler, os.path.join(args.output_dir, "summary_scaler.pkl"))

    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    model = BatteryRULClassifier(
        cnn_dim       = args.cnn_dim,
        gru_dim       = args.gru_dim,
        gru_layers    = args.gru_layers,
        summary_feats = summary_feats,
        n_classes     = N_CLASSES,
        dropout       = args.dropout,
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

    from gen_features import V_BINS as _V
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

    # ── Loss function ─────────────────────────────────────────────────────────
    use_ordinal = args.loss == "ordinal"
    if use_ordinal:
        criterion = OrdinalLoss(n_classes=N_CLASSES)
        pred_fn   = predict_cls
        print("Loss: OrdinalLoss (N-1 sigmoid thresholds)\n")
    else:
        criterion = nn.CrossEntropyLoss()
        pred_fn   = lambda logits: logits.argmax(dim=1)
        print("Loss: CrossEntropyLoss\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'max',
        factor   = 0.5,
        patience = 3,
        min_lr   = args.lr * 0.01,
        verbose  = True,
    )

    best_val_acc = 0.0

    header = f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc       = train_epoch(model, train_loader, criterion, pred_fn, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, pred_fn, device)
        scheduler.step(va_acc)

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
        torch.load(os.path.join(args.output_dir, "best_clf.pt"),
                   map_location=device, weights_only=True)
    )
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion, pred_fn, device)

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
    parser.add_argument("--loss", default="ordinal",
                    choices=["cross_entropy", "ordinal"],
                    help="Loss function: ordinal or cross_entropy")
    args = parser.parse_args()

    train(args)
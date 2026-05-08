"""
train_clf_bml.py - Train CNN+GRU classifier on BatteryML-derived features.

How to run:
    python train_clf_bml.py --content_dir ./content_bml --output_dir ./checkpoints_clf_bml

Train one BML family/subfolder only:
    python train_clf_bml.py --content_dir ./content_bml/MATR --output_dir ./checkpoints_clf_bml_MATR --es_mode 0

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
import cma

from dataset_clf_bml import N_CLASSES, N_INPUT, build_clf_dataloaders
from train_clf import BatteryRULClassifier, OrdinalLoss, evaluate, predict_cls, train_epoch


# -------------------------------------------------------------------------
# Parameter helpers — flatten / unflatten model weights
# -------------------------------------------------------------------------
def get_flat_params(model: nn.Module) -> np.ndarray:
    """Flatten all parameters into a single 1-D numpy array."""
    return np.concatenate([
        p.data.cpu().numpy().ravel()
        for p in model.parameters()
    ])


def set_flat_params(model: nn.Module, flat: np.ndarray) -> None:
    """Load a flat numpy array back into model parameters in-place."""
    offset = 0
    for p in model.parameters():
        size = p.numel()
        p.data.copy_(
            torch.tensor(
                flat[offset: offset + size].reshape(p.shape),
                dtype=p.dtype,
            )
        )
        offset += size


# -------------------------------------------------------------------------
# Fitness function — accuracy on one pass through loader
# -------------------------------------------------------------------------
@torch.no_grad()
def fitness(model: nn.Module, loader, device: torch.device) -> float:
    """Return accuracy (higher = better) over the full loader."""
    model.eval()
    correct = 0
    total   = 0
    for batch in loader:
        dq      = batch["dq"].to(device)
        summary = batch["summary"].to(device)
        labels  = batch["label"].to(device)
        logits  = model(dq, summary)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total   += len(labels)
    return correct / total


# -------------------------------------------------------------------------
# CMA-ES run — called when gradient training stagnates
# -------------------------------------------------------------------------
def cmaes_run(
    model:       nn.Module,
    train_loader,
    val_loader,
    device:      torch.device,
    n_gen:       int   = 50,    # max generations
    sigma:       float = 0.02,   # initial step size
    popsize:     int   = None,   # None = auto (4 + 3*ln(n_params))
    seed:        int   = 42,
) -> nn.Module:
    """
    CMA-ES fine-tuning loop. Optimizes val accuracy directly.
    Fitness evaluated on val_loader (not train_loader) to avoid overfitting.

    Returns model with best found parameters loaded.
    """
    x0       = get_flat_params(model).astype(np.float64)
    n_params = len(x0)

    best_acc  = fitness(model, val_loader,   device)
    train_acc = fitness(model, train_loader, device)
    print(f"  CMA-ES start — val: {best_acc:.4f}  train: {train_acc:.4f}"
          f"  n_params: {n_params:,}  sigma: {sigma:.4f}")

    cma_opts = {
        'seed':         seed,
        'maxiter':      n_gen,
        'verbose':      -9,      # suppress CMA's own output
        'tolx':         1e-8,    # stop if step size collapses
        'tolfun':       1e-7,    # stop if fitness change too small
        'CMA_diagonal': True,    # O(n) memory instead of O(n^2)
    }
    if popsize is not None:
        cma_opts['popsize'] = popsize

    es          = cma.CMAEvolutionStrategy(x0, sigma, cma_opts)
    best_params = x0.copy()

    header = f"  {'Gen':>5} | {'TrainAcc':>8} | {'Sigma':>10} | {'ValAcc':>8} | {'Improved':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    while not es.stop():
        solutions = es.ask()

        # FIX: fitness evaluated on val_loader, not train_loader
        fitnesses = []
        for sol in solutions:
            set_flat_params(model, sol.astype(np.float32))
            acc = fitness(model, val_loader, device)
            fitnesses.append(-acc)                   # CMA minimizes → negate

        es.tell(solutions, fitnesses)

        best_idx     = int(np.argmin(fitnesses))
        gen_best_acc = -fitnesses[best_idx]
        improved     = gen_best_acc > best_acc

        if improved:
            best_acc    = gen_best_acc
            best_params = solutions[best_idx].copy()

        # monitor train acc of the best offspring this generation
        set_flat_params(model, solutions[best_idx].astype(np.float32))
        train_acc = fitness(model, train_loader, device)

        # restore best params
        set_flat_params(model, best_params.astype(np.float32))

        print(f"  {es.countiter:5d} | {train_acc:8.4f} | {es.sigma:10.6f} | {best_acc:8.4f} | "
              f"{'yes' if improved else 'no':>9}")

    print(f"  CMA-ES stopped: {es.stop()}")
    set_flat_params(model, best_params.astype(np.float32))


# -------------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------------
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
        cnn_dim=args.cnn_dim,
        gru_dim=args.gru_dim,
        gru_layers=args.gru_layers,
        summary_feats=summary_feats,
        n_classes=N_CLASSES,
        dropout=args.dropout,
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'max',        # monitor val accuracy (higher = better)
        factor   = 0.5,          # new_lr = lr * factor
        patience = 6,            # wait 6 epochs with no improvement
        min_lr   = args.lr * 0.01,
        verbose  = True,
    )

    best_val_acc     = -1.0
    best_path        = os.path.join(args.output_dir, "best_clf_bml.pt")
    cnt_not_improved = 0

    # ── ES mode 3: skip gradient training entirely ────────────────────────
    if args.es_mode == 3:
        print("ES mode 3: CMA-ES only (no gradient training)")
        cmaes_run(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            device       = device,
            n_gen        = 50,
            sigma        = 0.02,
            popsize      = None,
            seed         = args.seed,
        )
        best_val_acc = fitness(model, val_loader, device)
        torch.save(model.state_dict(), best_path)
        print(f"  CMA-ES best val acc: {best_val_acc:.4f} — checkpoint saved.")

    else:
        # ── ES modes 0, 1, 2: gradient training loop ─────────────────────
        header = f"{'Epoch':>5} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>8}"
        print(header)
        print("-" * len(header))

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc       = train_epoch(model, train_loader, criterion, pred_fn, optimizer, device)
            va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, pred_fn, device)
            scheduler.step(va_acc)

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
                f"{va_loss:8.4f} | {va_acc:7.4f} | {lr:.2e}"
            )

            if va_acc >= best_val_acc:
                best_val_acc     = va_acc
                cnt_not_improved = 0
                torch.save(model.state_dict(), best_path)
            else:
                cnt_not_improved += 1

            # mode 2 (default): CMA-ES triggered mid-training on stagnation
            if args.es_mode == 2 and cnt_not_improved >= 5:
                print(f"\n[Epoch {epoch}] No improvement for {cnt_not_improved} epochs — running CMA-ES...")
                cmaes_run(
                    model        = model,
                    train_loader = train_loader,
                    val_loader   = val_loader,
                    device       = device,
                    n_gen        = 10,
                    sigma        = 0.02,
                    popsize      = None,
                    seed         = args.seed,
                )
                cma_val_acc = fitness(model, val_loader, device)
                if cma_val_acc > best_val_acc:
                    best_val_acc = cma_val_acc
                    torch.save(model.state_dict(), best_path)
                    print(f"  CMA-ES improved val acc to {best_val_acc:.4f} — checkpoint saved.")
                # else:
                #     model.load_state_dict(
                #         torch.load(best_path, map_location=device, weights_only=True)
                #     )
                #     print(f"  CMA-ES did not improve ({cma_val_acc:.4f} <= {best_val_acc:.4f}) — restoring gradient best.")
                cnt_not_improved = 0
           
        # ── ES mode 1: CMA-ES after gradient training finishes ────────────
        if args.es_mode == 1:
            print(f"\nES mode 1: running CMA-ES after gradient training...")
            model.load_state_dict(
                torch.load(best_path, map_location=device, weights_only=True)
            )
            cmaes_run(
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                device       = device,
                n_gen        = 50,
                sigma        = 0.02,
                popsize      = None,
                seed         = args.seed,
            )
            cma_val_acc = fitness(model, val_loader, device)
            if cma_val_acc > best_val_acc:
                best_val_acc = cma_val_acc
                torch.save(model.state_dict(), best_path)
                print(f"  CMA-ES improved val acc to {best_val_acc:.4f} — checkpoint saved.")
            else:
                model.load_state_dict(
                    torch.load(best_path, map_location=device, weights_only=True)
                )
                print(f"  CMA-ES did not improve ({cma_val_acc:.4f} <= {best_val_acc:.4f}) — restoring gradient best.")

    # ── Test ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion, pred_fn, device)

    print(f"\nTest Loss: {te_loss:.4f}  Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            true,
            pred,
            labels=list(range(N_CLASSES)),
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
    parser.add_argument("--output_dir",  default="./checkpoints_clf_bml")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--n_samples",   type=int,   default=600)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--cnn_dim",     type=int,   default=32)
    parser.add_argument("--gru_dim",     type=int,   default=32)
    parser.add_argument("--gru_layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument(
        "--loss",
        default="ordinal",
        choices=["cross_entropy", "ordinal"],
        help="Loss function: ordinal or cross_entropy",
    )
    parser.add_argument(
        "--es_mode", type=int, default=0,
        choices=[0, 1, 2, 3],
        help=(
            "Evolutionary strategy mode: "
            "0=no ES (default: gradient only), "
            "1=ES after gradient training, "
            "2=ES triggered on stagnation, "
            "3=ES only (no gradient training)"
        ),
    )
    args = parser.parse_args()

    train(args)
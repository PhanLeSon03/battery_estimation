"""
train_clf_es_bml.py — Train CNN+GRU classifier for battery RUL using CMA-ES
                       on BatteryML dataset (no backprop, no optimizer)

Classes:
    0: RUL > 400
    1: RUL 300–400
    2: RUL 200–300
    3: RUL 100–200
    4: RUL < 100

ES strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    - adapts full covariance of the search distribution each generation
    - requires: pip install cma

Usage:
    python train_clf_es_bml.py --content_dir ./content_bml --output_dir ./checkpoints_es_bml
    python train_clf_es_bml.py --content_dir ./content_bml --output_dir ./checkpoints_es_bml --pretrain_ckpt checkpoints_clf_bml/best_clf_bml.pt
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import cma   # pip install cma


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

from dataset_clf_bml import build_clf_dataloaders, N_CLASSES, N_INPUT, V_BINS  # BML dataset
from train_clf import BatteryRULClassifier, evaluate, OrdinalLoss, predict_cls, ordinal_predict 
import joblib


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False



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
# Evaluate — loss + acc + predictions (for final report)
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, pred_fn, device: torch.device):
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
# CMA-ES training loop
# -------------------------------------------------------------------------
def cmaes_train(
    model:       nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device:      torch.device,
    n_gen:       int   = 200,    # max generations
    sigma:       float = 0.02,   # initial step size (std of search distribution)
    popsize:     int   = None,   # CMA population size; None = auto (4 + 3*ln(n))
    output_dir:  str   = ".",
    seed:        int   = 42,
    acc_gap_tol: float = 0.01,   # stop when |train_acc - val_acc| <= this threshold
    acc_min:     float = 0.70,   # only trigger gap stop if val_acc is above this floor
):
    """
    CMA-ES training loop.

    CMA-ES adapts the full covariance matrix of the search distribution
    each generation, allowing it to learn correlations between parameters
    and scale the search appropriately per direction — far more efficient
    than isotropic (1+lambda)-ES for high-dimensional spaces.

    CMA minimizes, so we pass -accuracy as the objective.

    Key CMA-ES internals (handled by the `cma` library):
        - rank-mu update:  covariance update using all offspring weighted by rank
        - rank-one update: covariance update from cumulative evolution path
        - step size control: CSA (cumulative step size adaptation)
        - termination: stagnation, flat fitness, condition number, etc.
    """
    # initialise from current model weights — CMA works in float64
    x0       = get_flat_params(model).astype(np.float64)
    n_params = len(x0)

    # evaluate initial params
    best_acc  = fitness(model, val_loader,   device)
    train_acc = fitness(model, train_loader, device)
    print(f"Initial val acc: {best_acc:.4f}  | train acc: {train_acc:.4f}"
          f"  |  n_params: {n_params:,}  sigma0: {sigma:.4f}")

    # CMA options — suppress internal output, we print ourselves
    cma_opts = {
        'seed':        seed,
        'maxiter':     n_gen,
        'verbose':     -9,
        'tolx':        1e-8,
        'tolfun':      1e-7,
        'CMA_diagonal': True,   # O(n) memory instead of O(n^2)
    }
    if popsize is not None:
        cma_opts['popsize'] = popsize   # default: 4 + floor(3 * ln(n_params))

    es = cma.CMAEvolutionStrategy(x0, sigma, cma_opts)

    header = f"{'Gen':>5} | {'BestAcc':>8} | {'TrainAcc':>8} | {'Sigma':>10} | {'Improved':>9}"
    print("\n" + header)
    print("-" * len(header))

    best_params = x0.copy()
    gen         = 0

    while not es.stop():
        gen += 1

        # ask: sample population of candidate solutions from current distribution
        solutions = es.ask()                          # list of float64 arrays, len = popsize

        # evaluate each candidate — CMA minimizes so negate accuracy
        fitnesses = []
        for sol in solutions:
            set_flat_params(model, sol.astype(np.float32))
            acc = fitness(model, val_loader, device)
            fitnesses.append(-acc)                    # minimize -acc = maximize acc

        # tell: CMA updates mean, covariance matrix, and step size
        es.tell(solutions, fitnesses)

        # track best of this generation
        best_idx     = int(np.argmin(fitnesses))
        gen_best_acc = -fitnesses[best_idx]
        improved     = gen_best_acc > best_acc

        if improved:
            best_acc    = gen_best_acc
            best_params = solutions[best_idx].copy()
            set_flat_params(model, best_params.astype(np.float32))
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_clf.pt"),
            )

        # train acc of best offspring this generation (for monitoring overfitting)
        set_flat_params(model, solutions[best_idx].astype(np.float32))
        train_acc = fitness(model, train_loader, device)
        test_acc = fitness(model, test_loader, device)

        # restore best params into model
        set_flat_params(model, best_params.astype(np.float32))

        print(f"{gen:5d} | {best_acc:8.4f} | {train_acc:8.4f} | {test_acc:8.4f} | "
              f"{es.sigma:10.6f} | {'yes' if improved else 'no':>9}")

        # stop when val and train acc are close — model has converged without overfitting
        if best_acc >= acc_min and abs(train_acc - best_acc) <= acc_gap_tol:
            print(f"\nEarly stop: val_acc={best_acc:.4f} train_acc={train_acc:.4f} "
                  f"gap={abs(train_acc - best_acc):.4f} <= tol={acc_gap_tol}")
            break

    print(f"\nCMA-ES stopped: {es.stop()}")

    # restore best into model for final eval
    set_flat_params(model, best_params.astype(np.float32))
    return best_acc


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(
        args.output_dir,
        f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Log file: {log_path}")
    print(f"Args: {vars(args)}")
    print(f"Device: {device}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader, scalers = build_clf_dataloaders(
        content_dir = args.content_dir,
        batch_size  = args.batch_size,
        n_samples   = args.n_samples,
        val_ratio   = 0.2,
        num_workers = args.num_workers,
        seed        = args.seed,
    )

    dq_scaler, summary_scaler = scalers
    joblib.dump(dq_scaler,      os.path.join(args.output_dir, "dq_scaler_bml.pkl"))
    joblib.dump(summary_scaler, os.path.join(args.output_dir, "summary_scaler_bml.pkl"))

    summary_feats = summary_scaler.n_features_in_
    print(f"summary_feats: {summary_feats}")

    model = BatteryRULClassifier(
        cnn_dim       = args.cnn_dim,
        gru_dim       = args.gru_dim,
        gru_layers    = args.gru_layers,
        summary_feats = summary_feats,
        n_classes     = N_CLASSES,
        dropout       = 0.0,           # dropout off — ES evaluates deterministically
    ).to(device)

    print("\nModel architecture:")
    print(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total  = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters:     {n_total:,}")
    print(f"Trainable parameters: {n_params:,}\n")

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

    _dummy_dq      = torch.zeros(1, N_INPUT, 1, V_BINS,      device=device)  # BML: V_BINS from dataset_clf_bml
    _dummy_summary = torch.zeros(1, N_INPUT, summary_feats,  device=device)
    with torch.no_grad():
        model(_dummy_dq, _dummy_summary)

    for h in _handles:
        h.remove()

    for name, shape, n in _summary:
        print(f"  {name:<38} {str(shape):<25} {n:>10,}")

    print("=" * 80)
    print(f"  {'Total trainable parameters':<38} {'':25} {n_params:>10,}")
    print("=" * 80 + "\n")

    # ── Load pretrained weights (optional) ───────────────────────────────
    if args.pretrain_ckpt:
        ckpt = torch.load(args.pretrain_ckpt, map_location=device, weights_only=True)
        model_state = model.state_dict()
        matched, skipped = {}, []
        for k, v in ckpt.items():
            if k in model_state and model_state[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(f"{k}: ckpt{list(v.shape)} vs model"
                               f"{list(model_state[k].shape) if k in model_state else 'missing'}")
        model.load_state_dict(matched, strict=False)
        print(f"Pretrained weights loaded from: {args.pretrain_ckpt}")
        print(f"  Matched: {len(matched)}/{len(ckpt)} keys")
        if skipped:
            print(f"  Skipped (shape mismatch or missing):")
            for s in skipped:
                print(f"    {s}")
    else:
        print("No pretrained weights — starting from random init.")

    # ── Loss + pred_fn (for final evaluate only, not used in ES fitness) ─
    use_ordinal = args.loss == "ordinal"
    if use_ordinal:
        criterion = OrdinalLoss(n_classes=N_CLASSES)
        pred_fn   = predict_cls
        print("Loss: OrdinalLoss (N-1 sigmoid thresholds)\n")
    else:
        criterion = nn.CrossEntropyLoss()
        pred_fn   = lambda logits: logits.argmax(dim=1)
        print("Loss: CrossEntropyLoss\n")

    # ── CMA-ES training ───────────────────────────────────────────────────
    best_val_acc = cmaes_train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        device       = device,
        n_gen        = args.n_gen,
        sigma        = args.sigma,
        popsize      = args.popsize,
        output_dir   = args.output_dir,
        seed         = args.seed,
        acc_gap_tol  = args.acc_gap_tol,
        acc_min      = args.acc_min,
    )

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
    parser.add_argument("--content_dir", default="./content_bml")
    parser.add_argument("--output_dir",  default="./checkpoints_es_bml")
    parser.add_argument("--batch_size",  type=int,   default=256)   # larger batch = stable fitness signal
    parser.add_argument("--n_samples",   type=int,   default=600)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--cnn_dim",     type=int,   default=32)
    parser.add_argument("--gru_dim",     type=int,   default=32)
    parser.add_argument("--gru_layers",  type=int,   default=2)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--pretrain_ckpt", default=None,
                        help="path to best_clf.pt from train_clf_bml.py")
    parser.add_argument("--loss", default="ordinal",
                        choices=["cross_entropy", "ordinal"],
                        help="Loss function for final test evaluation only")
    # ── CMA-ES hyperparameters ───────────────────────────────────────────
    parser.add_argument("--n_gen",   type=int,   default=200,  help="max generations")
    parser.add_argument("--sigma",   type=float, default=0.02, help="initial step size (sigma0)")
    parser.add_argument("--popsize", type=int,   default=None,
                        help="CMA population size; None = auto (4 + 3*ln(n_params))")
    parser.add_argument("--acc_gap_tol", type=float, default=0.01,
                        help="stop when |train_acc - val_acc| <= this (convergence check)")
    parser.add_argument("--acc_min",     type=float, default=0.70,
                        help="gap stop only triggers when val_acc >= this floor")
    args = parser.parse_args()

    train(args)

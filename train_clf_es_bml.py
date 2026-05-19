"""
train_clf_es_bml.py — Train CNN+GRU classifier for battery RUL using Sparse CMA-ES
                       on BatteryML dataset, with weight heatmap export (no backprop, no optimizer)

Frozen:  CNN, summary_proj, post_gru_drop, head
ES only: GRU weights (gru.*)

Parallelism: offspring fitness evaluated in parallel via torch.multiprocessing.
             Each worker gets its own GPU/CPU model copy and evaluates one offspring.

Classes:
    0: RUL > 400
    1: RUL 300–400
    2: RUL 200–300
    3: RUL 100–200
    4: RUL < 100

Usage:
    python train_clf_es_bml.py --content_dir ./content_bml --output_dir ./checkpoints_es_bml
    python train_clf_es_bml.py --content_dir ./content_bml/HUST --output_dir ./checkpoints_clf_bml_HUST_es --pretrain_ckpt checkpoints_clf_bml_HUST/best_clf_bml.pt
"""

import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from sklearn.metrics import classification_report, confusion_matrix
import cma

from dataset_clf_bml import build_clf_dataloaders, N_CLASSES, N_INPUT, V_BINS
from train_clf import BatteryRULClassifier, evaluate, OrdinalLoss, predict_cls, ordinal_predict
import joblib
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Sparse ES parameter helpers
# FROZEN:  cnn, summary_proj, post_gru_drop, head
# ES ONLY: gru.*, head.*
# -------------------------------------------------------------------------
_ES_MODULES  = ("cnn", "summary_proj", "gru", "post_gru_drop", "head")
_SPARSE_META = []


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_sparse_es_mask(
    model:         nn.Module,
    keep_ratio:    float = 0.2,
    keep_bias:     bool  = True,
    zero_inactive: bool  = False,
) -> None:
    global _SPARSE_META
    _SPARSE_META = []

    keep_ratio = float(keep_ratio)
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")

    for name, p in model.named_parameters():
        if not any(name.startswith(m) for m in _ES_MODULES):
            continue
        arr = p.data.detach().cpu().numpy()
        if arr.ndim == 0:
            continue
        if p.ndim == 1 and keep_bias:
            mask = np.ones(arr.shape, dtype=bool)
        else:
            flat_abs = np.abs(arr).ravel()
            k = max(1, int(np.ceil(flat_abs.size * keep_ratio)))
            if k >= flat_abs.size:
                mask = np.ones(arr.shape, dtype=bool)
            else:
                threshold = np.partition(flat_abs, -k)[-k]
                mask = np.abs(arr) <= threshold   # keep LARGE weights

        if zero_inactive:
            pruned = arr.copy()
            pruned[~mask] = 0.0
            p.data.copy_(torch.tensor(pruned, dtype=p.dtype, device=p.device))

        _SPARSE_META.append({
            "name":   name,
            "param":  p,
            "shape":  tuple(p.shape),
            "mask":   mask,
            "active": int(mask.sum()),
            "total":  int(mask.size),
        })

    n_sparse = sum(m["active"] for m in _SPARSE_META)
    n_total  = sum(m["total"]  for m in _SPARSE_META)
    n_model  = sum(p.numel() for p in model.parameters())

    if n_sparse == 0:
        raise RuntimeError("Sparse ES mask is empty. Check _ES_MODULES or keep_ratio.")

    print("\nSparse ES mask  (frozen: cnn, summary_proj, post_gru_drop):")
    print("=" * 80)
    print(f"  {'Parameter':<45} {'Active/Total':>20} {'Ratio':>10}")
    print("-" * 80)
    for meta in _SPARSE_META:
        ratio = 100.0 * meta["active"] / meta["total"]
        print(f"  {meta['name']:<45} "
              f"{meta['active']:>8,}/{meta['total']:<8,} "
              f"{ratio:>9.2f}%")
    print("-" * 80)
    print(f"  {'ES active / ES total':<45} {n_sparse:>8,}/{n_total:<8,} "
          f"{100.0*n_sparse/n_total:>9.2f}%")
    print(f"  {'ES active / model total':<45} {n_sparse:>8,}/{n_model:<8,} "
          f"{100.0*n_sparse/n_model:>9.2f}%")
    print("=" * 80 + "\n")


def get_flat_params(model: nn.Module) -> np.ndarray:
    if not _SPARSE_META:
        raise RuntimeError("Call build_sparse_es_mask() first.")
    return np.concatenate([
        meta["param"].data.detach().cpu().numpy()[meta["mask"]].ravel()
        for meta in _SPARSE_META
    ]).astype(np.float64)


def set_flat_params(model: nn.Module, flat: np.ndarray) -> None:
    if not _SPARSE_META:
        raise RuntimeError("Call build_sparse_es_mask() first.")
    offset = 0
    flat   = np.asarray(flat)
    for meta in _SPARSE_META:
        p    = meta["param"]
        mask = meta["mask"]
        size = int(mask.sum())
        arr  = p.data.detach().cpu().numpy().copy()
        arr[mask] = flat[offset: offset + size]
        p.data.copy_(torch.tensor(arr, dtype=p.dtype, device=p.device))
        offset += size
    if offset != len(flat):
        raise ValueError(f"Unused values: used {offset}, got {len(flat)}")


def sparse_l1_penalty(model: nn.Module) -> float:
    if not _SPARSE_META:
        return 0.0
    total_abs = sum(float(np.abs(m["param"].data.detach().cpu().numpy()[m["mask"]]).sum())
                    for m in _SPARSE_META)
    total_num = sum(int(m["mask"].sum()) for m in _SPARSE_META)
    return total_abs / max(total_num, 1)


# -------------------------------------------------------------------------
# Weight heatmap visualization
# -------------------------------------------------------------------------
@torch.no_grad()
def save_weight_heatmaps(
    model_before,
    model_after,
    output_dir:      str,
    max_cols:        int  = 256,
    only_es_modules: bool = True,
    include_bias:    bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    before_dict = dict(model_before.named_parameters())
    after_dict  = dict(model_after.named_parameters())

    valid_layers = [
        name for name in before_dict
        if name in after_dict
        and before_dict[name].detach().cpu().numpy().ndim > 0
        and (not only_es_modules or any(name.startswith(m) for m in _ES_MODULES))
        and (include_bias or before_dict[name].ndim > 1)
    ]

    if not valid_layers:
        print("No layers to plot in heatmap.")
        return

    n_layers = len(valid_layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, max(3 * n_layers, 8)))
    if n_layers == 1:
        axes = np.array([axes])

    for row, name in enumerate(valid_layers):
        w_b = before_dict[name].detach().cpu().numpy()
        w_a = after_dict[name].detach().cpu().numpy()
        w_b2 = (w_b.reshape(1, -1) if w_b.ndim == 1 else w_b.reshape(w_b.shape[0], -1))[:, :max_cols]
        w_a2 = (w_a.reshape(1, -1) if w_a.ndim == 1 else w_a.reshape(w_a.shape[0], -1))[:, :max_cols]
        vmax = max(np.abs(w_b2).max(), np.abs(w_a2).max())
        axes[row, 0].imshow(w_b2, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f"Before ES\n{name}")
        axes[row, 0].set_ylabel(str(w_b2.shape))
        im = axes[row, 1].imshow(w_a2, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title(f"After ES\n{name}")

    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.995, pad=0.01).set_label("Weight Value")
    save_path = os.path.join(output_dir, "gru_weight_heatmaps.png")
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap: {save_path}")


# -------------------------------------------------------------------------
# Subset loader
# -------------------------------------------------------------------------
def _make_subset_loader(loader, max_batches: int = 10) -> DataLoader:
    dataset = loader.dataset
    n       = min(max_batches * loader.batch_size, len(dataset))
    idx     = np.random.choice(len(dataset), size=n, replace=False)
    subset  = Subset(dataset, idx)
    return DataLoader(subset,
                      batch_size  = loader.batch_size,
                      shuffle     = False,
                      num_workers = 0,
                      pin_memory  = loader.pin_memory)


# -------------------------------------------------------------------------
# Fitness — single model evaluation
# -------------------------------------------------------------------------
@torch.no_grad()
def fitness(model: nn.Module, loader, device: torch.device) -> float:
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
    return correct / total if total > 0 else 0.0


# -------------------------------------------------------------------------
# Parallel fitness worker (runs in subprocess)
# -------------------------------------------------------------------------
def _worker_fitness(rank: int,
                    flat_params: np.ndarray,
                    meta_list:   list,
                    model_state: dict,
                    model_kwargs: dict,
                    batches:     list,    # pre-fetched list of (dq, summary, labels)
                    result_queue: mp.Queue):
    """
    Worker process: load model, apply flat_params, evaluate on batches.
    Sends accuracy back via result_queue.
    """
    try:
        device = torch.device("cpu")   # each worker uses CPU (avoids GPU contention)

        # rebuild model
        model = BatteryRULClassifier(**model_kwargs).to(device)
        model.load_state_dict(model_state)
        model.eval()

        # apply sparse params
        offset = 0
        flat   = np.asarray(flat_params)
        for meta in meta_list:
            mask  = meta["mask"]
            size  = int(mask.sum())
            name  = meta["name"]
            # find param in model
            p = dict(model.named_parameters())[name]
            arr = p.data.detach().cpu().numpy().copy()
            arr[mask] = flat[offset: offset + size]
            p.data.copy_(torch.tensor(arr, dtype=p.dtype, device=device))
            offset += size

        correct = 0
        total   = 0
        with torch.no_grad():
            for dq, summary, labels in batches:
                logits   = model(dq.to(device), summary.to(device))
                correct += (logits.argmax(dim=1) == labels.to(device)).sum().item()
                total   += len(labels)

        acc = correct / total if total > 0 else 0.0
        result_queue.put((rank, acc))
    except Exception as e:
        result_queue.put((rank, 0.0))


# -------------------------------------------------------------------------
# Parallel fitness evaluation for all offspring in one generation
# -------------------------------------------------------------------------
def fitness_parallel(
    model:       nn.Module,
    solutions:   list,
    loader,
    device:      torch.device,
    n_workers:   int = 4,
    l1_lambda:   float = 0.0,
) -> list:
    """
    Evaluate fitness of all CMA-ES offspring in parallel.

    Strategy:
    - Pre-fetch a random subset of batches to CPU once (shared across workers)
    - Spawn n_workers processes, each handling len(solutions)//n_workers offspring
    - Workers use CPU to avoid GPU memory contention
    - If n_workers=1 or CUDA unavailable, falls back to sequential GPU eval

    Returns list of fitnesses (negated accuracy + penalty) same order as solutions.
    """
    # pre-fetch batches to CPU once — shared across all workers
    batches = [(b["dq"].cpu(), b["summary"].cpu(), b["label"].cpu())
               for b in loader]

    # gather lightweight metadata (no param tensors — not picklable across processes)
    meta_list = [{"name": m["name"], "mask": m["mask"]} for m in _SPARSE_META]
    model_state  = {k: v.cpu() for k, v in model.state_dict().items()}
    model_kwargs = {
        "cnn_dim":       model.head[0].in_features // 2,   # gru_dim*2 → gru_dim
        "gru_dim":       model.gru.hidden_size,
        "gru_layers":    model.gru.num_layers,
        "summary_feats": model.summary_proj[0].in_features,
        "n_classes":     N_CLASSES,
        "dropout":       0.0,
    }

    result_queue = mp.Queue()
    n_sol        = len(solutions)

    # chunk solutions across workers
    chunk_size = max(1, (n_sol + n_workers - 1) // n_workers)
    processes  = []

    for w in range(n_workers):
        start = w * chunk_size
        end   = min(start + chunk_size, n_sol)
        if start >= n_sol:
            break
        for i, sol in enumerate(solutions[start:end]):
            p = mp.Process(
                target = _worker_fitness,
                args   = (start + i, sol.astype(np.float32), meta_list,
                          model_state, model_kwargs, batches, result_queue),
                daemon = True,
            )
            p.start()
            processes.append(p)

    # collect results
    results = {}
    for _ in processes:
        rank, acc = result_queue.get()
        results[rank] = acc

    for p in processes:
        p.join(timeout=60)

    fitnesses = []
    for i in range(n_sol):
        acc     = results.get(i, 0.0)
        penalty = 0.0
        if l1_lambda > 0:
            # approximate L1 from flat params (no model needed)
            flat = solutions[i].astype(np.float32)
            penalty = float(np.abs(flat).mean()) * l1_lambda
        fitnesses.append(-acc + penalty)

    return fitnesses


# -------------------------------------------------------------------------
# CMA-ES training loop
# -------------------------------------------------------------------------
def cmaes_train(
    model:       nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device:      torch.device,
    n_gen:       int   = 200,
    sigma:       float = 0.01,
    popsize:     int   = None,
    output_dir:  str   = ".",
    seed:        int   = 42,
    acc_gap_tol: float = 0.01,
    acc_min:     float = 0.70,
    l1_lambda:   float = 0.0,
    n_workers:   int   = 4,
):
    """
    Sparse CMA-ES — GRU weights only.
    Offspring fitness evaluated in parallel across n_workers CPU processes.
    Monitoring (train/val/test) still runs on GPU.
    """
    x0       = get_flat_params(model).astype(np.float64)
    n_params = len(x0)
    n_total  = sum(p.numel() for p in model.parameters())

    train_acc = fitness(model, train_loader, device)
    best_acc  = fitness(model, val_loader,   device)
    test_acc  = fitness(model, test_loader,  device)
    print(f"GRU ES params: {n_params:,} active / {n_total:,} model total")
    print(f"Parallel workers: {n_workers}")
    print(f"Initial  train={train_acc:.4f}  val={best_acc:.4f}  test={test_acc:.4f}  sigma={sigma:.4f}")

    cma_opts = {
        'seed':         seed,
        'maxiter':      n_gen,
        'verbose':      -9,
        'tolx':         1e-8,
        'tolfun':       1e-7,
        'CMA_diagonal': True,
    }
    if popsize is not None:
        cma_opts['popsize'] = popsize

    es          = cma.CMAEvolutionStrategy(x0, sigma, cma_opts)
    best_params = x0.copy()
    best_acc_x  = 0.0

    header = f"{'Gen':>5} | {'TrainAcc':>8} | {'ValAcc':>8} | {'TestAcc':>8} | {'Sigma':>10} | {'Improved':>9}"
    print("\n" + header)
    print("-" * len(header))

    while not es.stop():
        # fresh random val subset — fast fitness signal
        fast_loader = _make_subset_loader(val_loader, max_batches=10)

        solutions = es.ask()

        # ── PARALLEL fitness evaluation ───────────────────────────────────
        if n_workers > 1:
            fitnesses = fitness_parallel(
                model     = model,
                solutions = solutions,
                loader    = fast_loader,
                device    = device,
                n_workers = n_workers,
                l1_lambda = l1_lambda,
            )
        else:
            # sequential fallback (GPU, no multiprocessing)
            fitnesses = []
            for sol in solutions:
                set_flat_params(model, sol.astype(np.float32))
                acc     = fitness(model, fast_loader, device)
                penalty = l1_lambda * sparse_l1_penalty(model)
                fitnesses.append(-acc + penalty)

        es.tell(solutions, fitnesses)

        best_idx     = int(np.argmin(fitnesses))
        gen_best_acc = -fitnesses[best_idx]

        improved = gen_best_acc > best_acc_x
        if improved:
            best_acc_x  = gen_best_acc
            best_params = solutions[best_idx].copy()
            set_flat_params(model, best_params.astype(np.float32))
            torch.save(model.state_dict(), os.path.join(output_dir, "best_clf.pt"))

        # monitoring — GPU, full loaders
        set_flat_params(model, solutions[best_idx].astype(np.float32))
        train_acc = fitness(model, train_loader, device)
        val_acc   = fitness(model, val_loader,   device)
        test_acc  = fitness(model, test_loader,  device)

        # restore best
        set_flat_params(model, best_params.astype(np.float32))

        print(f"{es.countiter:5d} | {train_acc:8.4f} | {val_acc:8.4f} | {test_acc:8.4f} | "
              f"{es.sigma:10.6f} | {'yes' if improved else 'no':>9}")

        if best_acc >= acc_min and abs(val_acc - train_acc) <= acc_gap_tol:
            print(f"\nEarly stop: train={train_acc:.4f} val={val_acc:.4f} "
                  f"gap={abs(val_acc - train_acc):.4f} <= tol={acc_gap_tol}")
            break

    print(f"\nCMA-ES stopped: {es.stop()}")
    set_flat_params(model, best_params.astype(np.float32))
    return best_acc


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    mp.set_start_method("spawn", force=True)   # required for CUDA + multiprocessing

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
        seed        = args.seed,
    )

    if args.pretrain_ckpt:
        ckpt_dir       = os.path.dirname(args.pretrain_ckpt)
        dq_scaler      = joblib.load(os.path.join(ckpt_dir, "dq_scaler_bml.pkl"))
        summary_scaler = joblib.load(os.path.join(ckpt_dir, "summary_scaler_bml.pkl"))
    else:
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
        dropout       = 0.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── Layer-by-layer summary ─────────────────────────────────────────────
    print("Model Summary:")
    print("=" * 80)
    print(f"  {'Layer':<38} {'Output Shape':<25} {'Params':>10}")
    print("-" * 80)
    _handles      = []
    _layer_summary = []

    def _hook(module, inp, out):
        if len(list(module.children())) == 0:
            n     = sum(p.numel() for p in module.parameters())
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else "?"
            _layer_summary.append((module.__class__.__name__, shape, n))

    for m in model.modules():
        _handles.append(m.register_forward_hook(_hook))

    _dummy_dq      = torch.zeros(1, N_INPUT, 1, V_BINS,     device=device)
    _dummy_summary = torch.zeros(1, N_INPUT, summary_feats, device=device)
    with torch.no_grad():
        model(_dummy_dq, _dummy_summary)
    for h in _handles:
        h.remove()

    for name, shape, n in _layer_summary:
        print(f"  {name:<38} {str(shape):<25} {n:>10,}")
    print("=" * 80)
    print(f"  {'Total trainable parameters':<38} {'':25} {n_params:>10,}")
    print("=" * 80 + "\n")

    # ── Load pretrained weights ────────────────────────────────────────────
    if args.pretrain_ckpt:
        ckpt        = torch.load(args.pretrain_ckpt, map_location=device, weights_only=True)
        model_state = model.state_dict()
        matched, skipped = {}, []
        for k, v in ckpt.items():
            if k in model_state and model_state[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(f"{k}: ckpt{list(v.shape)} vs "
                               f"{list(model_state[k].shape) if k in model_state else 'missing'}")
        model.load_state_dict(matched, strict=False)
        print(f"Pretrained weights loaded from: {args.pretrain_ckpt}")
        print(f"  Matched: {len(matched)}/{len(ckpt)} keys")
        if skipped:
            for s in skipped:
                print(f"    {s}")
    else:
        print("No pretrained weights — starting from random init.")

    # ── Loss + pred_fn ────────────────────────────────────────────────────
    if args.loss == "ordinal":
        criterion = OrdinalLoss(n_classes=N_CLASSES)
        pred_fn   = predict_cls
        print("Loss: OrdinalLoss\n")
    else:
        criterion = nn.CrossEntropyLoss()
        pred_fn   = lambda logits: logits.argmax(dim=1)
        print("Loss: CrossEntropyLoss\n")

    # ── Build sparse ES mask ───────────────────────────────────────────────
    build_sparse_es_mask(
        model,
        keep_ratio    = args.es_keep_ratio,
        keep_bias     = args.es_keep_bias,
        zero_inactive = args.es_zero_inactive,
    )

    model_before_es = copy.deepcopy(model).cpu()

    # ── CMA-ES training ───────────────────────────────────────────────────
    best_val_acc = cmaes_train(
        model        = model,
        train_loader = val_loader,
        val_loader   = train_loader,
        test_loader  = test_loader,
        device       = device,
        n_gen        = args.n_gen,
        sigma        = args.sigma,
        popsize      = args.popsize,
        output_dir   = args.output_dir,
        seed         = args.seed,
        acc_gap_tol  = args.acc_gap_tol,
        acc_min      = args.acc_min,
        l1_lambda    = args.es_l1_lambda,
        n_workers    = args.n_workers,
    )

    # ── Test ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_clf.pt"),
                   map_location=device, weights_only=True)
    )
    te_loss, te_acc, pred, true = evaluate(model, test_loader, criterion, pred_fn, device)

    print(f"\nTest Loss: {te_loss:.4f}  Accuracy: {te_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
          true, pred,
          labels      = list(range(N_CLASSES)),
          target_names= ["RUL>400", "RUL>300", "RUL>200", "RUL>100", "RUL<100"],
          zero_division= 0))
    print("Confusion Matrix:")
    print(confusion_matrix(true, pred))

    np.save(os.path.join(args.output_dir, "clf_pred.npy"), pred)
    np.save(os.path.join(args.output_dir, "clf_true.npy"), true)
    print(f"\nSaved to {args.output_dir}/")

    # ── Weight heatmaps ───────────────────────────────────────────────────
    if not args.no_weight_heatmaps:
        save_weight_heatmaps(
            model_before    = model_before_es,
            model_after     = copy.deepcopy(model).cpu(),
            output_dir      = os.path.join(args.output_dir, "weight_heatmaps"),
            max_cols        = args.heatmap_max_cols,
            only_es_modules = True,
            include_bias    = args.heatmap_include_bias,
        )


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir",  default="./content_bml")
    parser.add_argument("--output_dir",   default="./checkpoints_es_bml")
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--n_samples",    type=int,   default=600)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--cnn_dim",      type=int,   default=32)
    parser.add_argument("--gru_dim",      type=int,   default=32)
    parser.add_argument("--gru_layers",   type=int,   default=2)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--pretrain_ckpt", default=None)
    parser.add_argument("--loss", default="ordinal",
                        choices=["cross_entropy", "ordinal"])
    # ── CMA-ES ────────────────────────────────────────────────────────────
    parser.add_argument("--n_gen",       type=int,   default=20)
    parser.add_argument("--sigma",       type=float, default=0.02)
    parser.add_argument("--popsize",     type=int,   default=None)
    parser.add_argument("--acc_gap_tol", type=float, default=0.01)
    parser.add_argument("--acc_min",     type=float, default=0.70)
    # ── Sparse ES ─────────────────────────────────────────────────────────
    parser.add_argument("--es_keep_ratio",    type=float, default=0.3)
    parser.add_argument("--es_l1_lambda",     type=float, default=0.0)
    parser.add_argument("--es_keep_bias",     action="store_true", default=True)
    parser.add_argument("--es_zero_inactive", action="store_true")
    # ── Parallelism ───────────────────────────────────────────────────────
    parser.add_argument("--n_workers",   type=int,   default=4,
                        help="parallel CPU workers for offspring fitness eval; 1=sequential GPU")
    # ── Heatmaps ──────────────────────────────────────────────────────────
    parser.add_argument("--no_weight_heatmaps",   action="store_true")
    parser.add_argument("--heatmap_max_cols",     type=int,  default=256)
    parser.add_argument("--heatmap_include_bias", action="store_true")
    args = parser.parse_args()

    train(args)
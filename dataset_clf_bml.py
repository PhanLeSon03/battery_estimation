"""
dataset_clf_bml.py - BatteryML RUL classification dataset.

Mirrors dataset_clf.py (MIT) exactly. The only intentional differences are:
  - 8 summary scalars (BML lacks IR / temperature) instead of 12
  - cycle_life is recomputed from the actual 80% Qd EOL at load time, and
    all per-cycle arrays are truncated to that point (so post-EOL cycles are
    excluded). This is BML-specific because BML files are commonly cycled
    until total death.
  - observed_cycle() maps array index to the real cycle number, since BML
    cycle indexing can be non-sequential.

Classes are fixed (same as MIT):
    0: RUL > 400
    1: 300 < RUL ≤ 400
    2: 200 < RUL ≤ 300
    3: 100 < RUL ≤ 200
    4: RUL ≤ 100

How to run:
    python dataset_clf_bml.py --content_dir content_bml
    python dataset_clf_bml.py --content_dir content_bml/CALB
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


EOL_FRACTION = 0.80   # capacity retention threshold that defines end-of-life

N_EARLY   = 8     # always include first 8 cycles
N_RANDOM  = 24    # random consecutive window
N_INPUT   = N_EARLY + N_RANDOM   # 32 total
N_CLASSES = 5

V_BINS    = 1000
REF_CYCLE = 9


def rul_to_class(rul: float) -> int:
    if rul > 400: return 0
    if rul > 300: return 1
    if rul > 200: return 2
    if rul > 100: return 3
    return 4


def _find_eol_idx(qd: np.ndarray, eol_fraction: float = EOL_FRACTION) -> int:
    """Return the array index that marks end-of-life for this cell.

    Q_initial = max Qd over the first min(10, n) positive finite cycles
    (skips formation anomalies and early dips).

    Primary rule: first cycle past the reference window where Qd crosses
    below eol_fraction * Q_initial.
    Fallback (cell never aged below 80 %): index of the minimum Qd in the
    post-reference range — i.e. the most-aged cycle the cell ever reached.
    """
    qd = np.asarray(qd, dtype=np.float32)
    valid = np.isfinite(qd) & (qd > 0)
    if valid.sum() < 2:
        return len(qd)

    valid_idx = np.where(valid)[0]
    n_ref     = min(10, len(valid_idx))
    q_init    = float(np.max(qd[valid_idx[:n_ref]]))
    if q_init <= 0:
        return len(qd)

    q_eol   = eol_fraction * q_init
    ref_end = valid_idx[n_ref - 1]

    for i in valid_idx:
        if i <= ref_end:
            continue
        if qd[i] < q_eol:
            return int(i)

    # Never aged below 80 % — fall back to min-Qd cycle in post-ref range
    post_ref = valid_idx[valid_idx > ref_end]
    if post_ref.size == 0:
        return len(qd)
    return int(post_ref[np.argmin(qd[post_ref])])


def _arr(data, key: str, dtype=np.float32) -> np.ndarray:
    if key not in data.files:
        return np.zeros(0, dtype=dtype)
    return np.asarray(data[key], dtype=dtype).reshape(-1)


# -------------------------------------------------------------------------
# Load one .npz file
# -------------------------------------------------------------------------
def _load_npz_cell(path: str) -> dict:
    with np.load(path, allow_pickle=True) as d:
        qd          = _arr(d, "qd")
        cycle_index = _arr(d, "cycle_index", dtype=np.int32)
        qdlin_raw   = np.asarray(d["qdlin"], dtype=np.float32)
        dqdv_raw    = (np.asarray(d["dqdv"], dtype=np.float32)
                       if "dqdv" in d.files else np.zeros((0,), dtype=np.float32))

        # ── Truncate to 80% EOL ──────────────────────────────────────────
        eol_idx = _find_eol_idx(qd)
        n_keep  = min(eol_idx + 1, len(qd))   # include the EOL cycle itself

        if eol_idx < len(qd) and cycle_index.size and eol_idx < cycle_index.size:
            cycle_life = int(cycle_index[eol_idx])
        elif eol_idx < len(qd):
            cycle_life = int(eol_idx + 1)
        else:
            cycle_life = (int(cycle_index[-1]) if cycle_index.size
                          else int(len(qd)))

        return {
            "cycle_life":  cycle_life,
            "cycle_index": cycle_index[:n_keep],
            "summary": {
                "QDischarge": qd[:n_keep],
                "chargetime": _arr(d, "chargetime")[:n_keep],
                "dqdv_max":   _arr(d, "dqdv_max")[:n_keep],
                "dqdv_min":   _arr(d, "dqdv_min")[:n_keep],
                "dqdv_avg":   _arr(d, "dqdv_avg")[:n_keep],
                "log_std_dq": _arr(d, "log_std_dq")[:n_keep],
                "log_std_I":  _arr(d, "log_std_I")[:n_keep],
                "log_std_ct": _arr(d, "log_std_ct")[:n_keep],
            },
            "qdlin": [x for x in qdlin_raw[:n_keep]],
            "dqdv":  [x for x in dqdv_raw[:n_keep]] if dqdv_raw.ndim > 1 else [],
        }


def load_all_npz(content_dir: str) -> dict:
    print("Loading BML .npz feature files...")
    cells: dict = {}
    root = Path(content_dir)

    for path in sorted(root.rglob("*.npz")):
        if path.name.startswith("._"):
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = Path(path.name)
        cell_id = "__".join(rel.with_suffix("").parts)
        try:
            cells[cell_id] = _load_npz_cell(str(path))
        except Exception as exc:
            print(f"  WARNING: could not load {path}: {exc}")

    print(f"  Loaded {len(cells)} BML cells")
    return cells


# -------------------------------------------------------------------------
# One sample = (early_8 + random_24, label)
# Multiple samples per cell drawn at different window starts
# -------------------------------------------------------------------------
def extract_clf_samples(cell: dict, n_samples: int = 10, seed: int = None) -> list:
    """
    Extract multiple classification samples from one cell.
    Each sample uses first 8 cycles + a random consecutive 24-cycle window.
    Window is drawn from cycles 8 onwards.

    Returns list of dicts with keys: dq, summary, label, rul.
    """
    rng = np.random.default_rng(seed)

    cycle_life  = int(cell["cycle_life"])
    cycle_index = np.asarray(cell.get("cycle_index", []), dtype=np.int32).reshape(-1)
    summary     = cell["summary"]
    qdlin_list  = cell["qdlin"]

    qd         = np.asarray(summary["QDischarge"], dtype=np.float32).reshape(-1)
    ct         = np.asarray(summary["chargetime"],  dtype=np.float32).reshape(-1)
    dqdv_max   = np.asarray(summary["dqdv_max"],    dtype=np.float32).reshape(-1)
    dqdv_min   = np.asarray(summary["dqdv_min"],    dtype=np.float32).reshape(-1)
    dqdv_avg   = np.asarray(summary["dqdv_avg"],    dtype=np.float32).reshape(-1)
    log_std_dq = np.asarray(summary["log_std_dq"],  dtype=np.float32).reshape(-1)
    log_std_I  = np.asarray(summary["log_std_I"],   dtype=np.float32).reshape(-1)
    log_std_ct = np.asarray(summary["log_std_ct"],  dtype=np.float32).reshape(-1)

    n_cyc = min(
        len(qd), len(ct), len(dqdv_max), len(dqdv_min), len(dqdv_avg),
        len(log_std_dq), len(log_std_I), len(log_std_ct), len(qdlin_list),
    )
    if cycle_index.size:
        n_cyc = min(n_cyc, cycle_index.size)

    # need at least early + random window
    if n_cyc < N_INPUT:
        return []

    ref_idx   = min(REF_CYCLE, n_cyc - 1)
    ref_qdlin = np.asarray(qdlin_list[ref_idx], dtype=np.float32).reshape(-1)

    def observed_cycle(c: int) -> int:
        if cycle_index.size and c < cycle_index.size:
            return int(cycle_index[c])
        return c + 1

    def get_dq(c: int) -> np.ndarray:
        q      = np.asarray(qdlin_list[c], dtype=np.float32).reshape(-1)
        length = min(len(q), len(ref_qdlin), V_BINS)
        out    = np.zeros(V_BINS, dtype=np.float32)
        out[:length] = q[:length] - ref_qdlin[:length]
        return out

    def get_summary_row(c: int, d_pos: int = 5) -> np.ndarray:
        pos = c

        def safe(arr: np.ndarray) -> float:
            return float(arr[pos]) if pos < len(arr) and np.isfinite(arr[pos]) else 0.0

        cycle_num = max(1, observed_cycle(c))
        pe = np.asarray(
            [
                np.sin(cycle_num / 3000 ** (2 * i / d_pos))
                if i % 2 == 0
                else np.cos(cycle_num / 3000 ** ((2 * i - 1) / d_pos))
                for i in range(1, d_pos)
            ],
            dtype=np.float32,
        )
        scalar_feats = np.asarray(
            [
                safe(qd), safe(ct), safe(dqdv_max), safe(dqdv_min), safe(dqdv_avg),
                safe(log_std_dq), safe(log_std_I), safe(log_std_ct),
            ],
            dtype=np.float32,
        )
        return np.concatenate([scalar_feats, pe])   # (12,)

    # ── Group all valid window starts by class ────────────────────────────
    max_start    = n_cyc - N_RANDOM
    class_starts = {i: [] for i in range(N_CLASSES)}
    for start in range(N_EARLY, max_start + 1):
        if start + N_RANDOM > n_cyc - 4:   # 4-cycle safety margin, same as MIT
            continue
        end_cycle = observed_cycle(start + N_RANDOM - 1)
        rul       = max(0, cycle_life - end_cycle)
        label     = rul_to_class(rul)
        class_starts[label].append(start)

    # ── Sample equally from each class ────────────────────────────────────
    n_per_class = max(1, n_samples // N_CLASSES)
    chosen: list = []
    for label, starts_for_class in class_starts.items():
        if not starts_for_class:
            continue
        pick = rng.choice(
            starts_for_class,
            size=min(n_per_class, len(starts_for_class)),
            replace=False,
        )
        chosen.extend([(int(s), label) for s in pick])

    # ── Build sample dicts ────────────────────────────────────────────────
    samples = []
    for start, label in chosen:
        window_cycles = list(range(start, start + N_RANDOM))
        all_cycles    = list(range(N_EARLY)) + window_cycles   # 32 total
        end_cycle     = observed_cycle(start + N_RANDOM - 1)

        samples.append({
            "dq":      np.stack([get_dq(c)          for c in all_cycles]),  # (32, 1000)
            "summary": np.stack([get_summary_row(c) for c in all_cycles]),  # (32, 12)
            "label":   label,
            "rul":     max(0, cycle_life - end_cycle),
        })

    return samples


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class BMLBatteryClsDataset(Dataset):
    def __init__(
        self,
        cells:     dict,
        cell_ids:  list,
        n_samples: int             = 500,
        scalers:   Optional[Tuple] = None,
        seed:      int             = 42,
    ):
        raw     = []
        skipped = 0

        for i, cid in enumerate(cell_ids):
            if cid not in cells:
                continue
            samples = extract_clf_samples(cells[cid], n_samples=n_samples,
                                          seed=seed + i)
            if samples:
                raw.extend(samples)
            else:
                skipped += 1

        print(f"  Cells valid: {len(cell_ids) - skipped}  "
              f"Skipped: {skipped}  "
              f"Total samples: {len(raw)}")

        if not raw:
            raise ValueError("No valid samples.")

        dq      = np.stack([r["dq"]      for r in raw])   # (N, 32, 1000)
        summary = np.stack([r["summary"] for r in raw])   # (N, 32, 12)
        labels  = np.asarray([r["label"] for r in raw], dtype=np.int64)

        N, T, F = summary.shape

        for c in range(N_CLASSES):
            print(f"  Class {c}: {(labels == c).sum()} samples")

        if scalers is None:
            self.dq_scaler      = StandardScaler()
            self.summary_scaler = StandardScaler()
            dq      = self.dq_scaler.fit_transform(
                dq.reshape(N * T, V_BINS)).reshape(N, T, V_BINS)
            summary = self.summary_scaler.fit_transform(
                summary.reshape(N * T, F)).reshape(N, T, F)
        else:
            self.dq_scaler, self.summary_scaler = scalers
            dq      = self.dq_scaler.transform(
                dq.reshape(N * T, V_BINS)).reshape(N, T, V_BINS)
            summary = self.summary_scaler.transform(
                summary.reshape(N * T, F)).reshape(N, T, F)

        self.dq       = torch.tensor(dq,      dtype=torch.float32)
        self.summary  = torch.tensor(summary, dtype=torch.float32)
        self.labels   = torch.tensor(labels,  dtype=torch.long)
        self.cell_ids = [r for r in cell_ids if r in cells]

    def __len__(self):
        return len(self.dq)

    def __getitem__(self, idx):
        return {
            "dq":      self.dq[idx],
            "summary": self.summary[idx],
            "label":   self.labels[idx],
        }

    def get_scalers(self):
        return self.dq_scaler, self.summary_scaler


# -------------------------------------------------------------------------
# Build DataLoaders
# -------------------------------------------------------------------------
def build_clf_dataloaders(
    content_dir: str,
    batch_size:  int   = 32,
    n_samples:   int   = 500,
    val_ratio:   float = 0.2,
    num_workers: int   = 0,
    seed:        int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple]:
    cells = load_all_npz(content_dir)

    ids = list(cells.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_val   = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:n_val]
    trn_ids = ids[n_val:]
    print(f"Split — Train: {len(trn_ids)}  Val/Test: {len(val_ids)}")

    print("Building train dataset...")
    train_ds = BMLBatteryClsDataset(cells, trn_ids, n_samples, seed=seed)
    scalers  = train_ds.get_scalers()

    print("Building val dataset...")
    val_ds   = BMLBatteryClsDataset(cells, val_ids, n_samples,
                                    scalers=scalers, seed=seed + 1)

    test_ds = val_ds

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        scalers,
    )


def _describe_loader(name: str, loader: DataLoader) -> None:
    ds = loader.dataset
    print(f"{name}: {len(ds)} samples")
    batch = next(iter(loader))
    print(f"  dq:      {tuple(batch['dq'].shape)}")
    print(f"  summary: {tuple(batch['summary'].shape)}")
    print(f"  label:   {tuple(batch['label'].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate BML RUL classification dataloaders."
    )
    parser.add_argument("--content_dir",  default="./content_bml")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--n_samples",    type=int,   default=500)
    parser.add_argument("--val_ratio",    type=float, default=0.2)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, scalers = build_clf_dataloaders(
        content_dir=args.content_dir,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print("\nDataloader check:")
    _describe_loader("Train", train_loader)
    _describe_loader("Val",   val_loader)
    _describe_loader("Test",  test_loader)
    print(f"Scaler features — dq: {scalers[0].n_features_in_}  summary: {scalers[1].n_features_in_}")


if __name__ == "__main__":
    main()

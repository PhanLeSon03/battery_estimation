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
        qc          = _arr(d, "qc")
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
                "Qd":    qd[:n_keep],
                "Qc":    qd[:n_keep],
                "c_t":    _arr(d, "c_t")[:n_keep],
                "dc_t":    _arr(d, "dc_t")[:n_keep],
                "dqdv_slope_max": _arr(d, "dqdv_slope_max")[:n_keep],
                "dqdv_slope_min": _arr(d, "dqdv_slope_min")[:n_keep],
                "dqdv_min":      _arr(d, "dqdv_min")[:n_keep],
                "dqdv_avg":      _arr(d, "dqdv_avg")[:n_keep],
                "log_std_Qd":    _arr(d, "log_std_Qd")[:n_keep],
                "log_std_Qc":    _arr(d, "log_std_Qc")[:n_keep],
                "log_std_Id":     _arr(d, "log_std_Id")[:n_keep],
                "log_std_Ic":     _arr(d, "log_std_Ic")[:n_keep],
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
# Sample index — lightweight tuples, no data loaded
# -------------------------------------------------------------------------
def build_sample_index(cells: dict, cell_ids: list, seed: int = 42,
                       n_samples: int = 500) -> list:
    """
    Returns a list of lightweight index tuples:
        (cell_id, window_start, label, rul)
    No actual signal data is loaded here.
    """
    index = []

    for i, cid in enumerate(cell_ids):
        if cid not in cells:
            continue
        cell_rng = np.random.default_rng(seed + i + 3)   # independent per cell

        cell        = cells[cid]
        cycle_life  = int(cell["cycle_life"])
        cycle_index = np.asarray(cell.get("cycle_index", []), dtype=np.int32).reshape(-1)
        n_cyc       = min(
            len(cell["summary"]["Qd"]),
            len(cell["qdlin"]),
        )
        if cycle_index.size:
            n_cyc = min(n_cyc, cycle_index.size)

        if n_cyc < N_INPUT:
            continue

        max_start    = n_cyc - N_RANDOM
        class_starts = {c: [] for c in range(N_CLASSES)}

        for start in range(N_EARLY, max_start + 1):
            if start + N_RANDOM > n_cyc - 4:   # 4-cycle safety margin
                continue
            end_cycle = int(cycle_index[start + N_RANDOM - 1]) 
            rul       = max(0, cycle_life - end_cycle)
            label     = rul_to_class(rul)
            class_starts[label].append(start)

        # n_per_class = max(1, n_samples // N_CLASSES)
        n_per_class = min(len(class_starts[c]) for c in range(N_CLASSES))
        
        for label, starts_list in class_starts.items():
            if not starts_list:
                continue
            pick = cell_rng.choice(starts_list,
                                   size=min(n_per_class, len(starts_list)),
                                   replace=False)
            for s in pick:
                end_cycle = int(cycle_index[int(s) + N_RANDOM - 1])
                index.append((cid, int(s), label,
                               max(0, cycle_life - end_cycle)))

    return index


# -------------------------------------------------------------------------
# Per-cell cache — built once on first access, sliced per __getitem__
# -------------------------------------------------------------------------
_SUMMARY_KEYS_BML = (
    "Qd", "Qc", "c_t", "dc_t",
    "dqdv_slope_max", "dqdv_slope_min", "dqdv_min", "dqdv_avg",
    "log_std_Qd", "log_std_Qc", "log_std_Id", "log_std_Ic"
) 

# _SUMMARY_KEYS_BML = (
#     "Qd",  "c_t", 
#     "dqdv_min", 
#     "log_std_Qd", "log_std_Qc", "log_std_Id"
# ) 

_N_SUMMARY_BML = len(_SUMMARY_KEYS_BML) + 4   # X scalars + 4 PE values


class _CellCache:
    """
    Pre-builds all cycle arrays for a cell on first access.
    Subsequent __getitem__ calls only slice — no numpy reconstruction.
        dq_all      : (n_cyc, 1, V_BINS)  — qdlin-ref
        summary_all : (n_cyc, 14)          — scalar feats + PE
    """
    __slots__ = ("dq_all", "summary_all", "_built")

    def __init__(self):
        self.dq_all      = None
        self.summary_all = None
        self._built      = False

    def build(self, cell: dict) -> None:
        if self._built:
            return
        qdlin_list  = cell["qdlin"]
        summary     = cell["summary"]
        cycle_index = np.asarray(cell.get("cycle_index", []), dtype=np.int32).reshape(-1)
        n_cyc       = len(qdlin_list)
        ref_idx     = min(REF_CYCLE, n_cyc - 1)
        ref_qdlin   = np.asarray(qdlin_list[ref_idx], dtype=np.float32).reshape(-1)

        dq_arr      = np.empty((n_cyc, V_BINS),        dtype=np.float32)
        summary_arr = np.empty((n_cyc, _N_SUMMARY_BML), dtype=np.float32)

        for c in range(n_cyc):
            q          = np.asarray(qdlin_list[c], dtype=np.float32).reshape(-1)
            length     = min(len(q), len(ref_qdlin), V_BINS)
            dq_arr[c]  = 0.0
            dq_arr[c, :length] = q[:length] - ref_qdlin[:length]
            summary_arr[c] = _get_summary_row(summary, c, cycle_index)

        self.dq_all      = np.expand_dims(dq_arr, axis=1)   # (n_cyc, 1, V_BINS)
        self.summary_all = summary_arr                       # (n_cyc, 14)
        self._built      = True


# -------------------------------------------------------------------------
# Feature helpers
# -------------------------------------------------------------------------
def _get_summary_row(summary: dict, c: int,
                     cycle_index: np.ndarray = None, d_pos: int = 5) -> np.ndarray:
    # use real cycle number for PE if available
    if cycle_index is not None and cycle_index.size and c < cycle_index.size:
        cycle_num = max(1, int(cycle_index[c]))
    else:
        cycle_num = c + 1

    def safe(arr: np.ndarray) -> float:
        return float(arr[c]) if c < len(arr) and np.isfinite(arr[c]) else 0.0

    pe = np.asarray([
        np.sin(cycle_num / 3000 ** (2 * i / d_pos)) if i % 2 == 0 else
        np.cos(cycle_num / 3000 ** ((2 * i - 1) / d_pos))
        for i in range(1, d_pos)
    ], dtype=np.float32)

    scalar_feats = np.asarray(
        [safe(np.asarray(summary[k], dtype=np.float32)) for k in _SUMMARY_KEYS_BML],
        dtype=np.float32,
    )
    return np.concatenate([scalar_feats, pe])   # (14,)


def _build_sample_tensors(cell: dict, start: int,
                           dq_scaler, summary_scaler,
                           cache: Optional[_CellCache] = None) -> Tuple:
    """Featurise one sample — fast path uses cache, slow path builds on-the-fly."""
    window_cycles = list(range(start, start + N_RANDOM))
    all_cycles    = list(range(N_EARLY)) + window_cycles   # 32 total

    if cache is not None and cache._built:
        # fast path: just slice pre-built arrays
        dq_seq      = cache.dq_all[all_cycles]       # (32, 1, V_BINS)
        summary_seq = cache.summary_all[all_cycles]  # (32, 14)
    else:
        # slow path: build on-the-fly (used during scaler fitting)
        qdlin_list  = cell["qdlin"]
        summary     = cell["summary"]
        cycle_index = np.asarray(cell.get("cycle_index", []), dtype=np.int32).reshape(-1)
        ref_idx     = min(REF_CYCLE, len(qdlin_list) - 1)
        ref_qdlin   = np.asarray(qdlin_list[ref_idx], dtype=np.float32).reshape(-1)

        dq_rows = []
        for c in all_cycles:
            q      = np.asarray(qdlin_list[c], dtype=np.float32).reshape(-1)
            length = min(len(q), len(ref_qdlin), V_BINS)
            row    = np.zeros(V_BINS, dtype=np.float32)
            row[:length] = q[:length] - ref_qdlin[:length]
            dq_rows.append(row)

        dq_seq      = np.expand_dims(np.stack(dq_rows), axis=1)   # (32, 1, V_BINS)
        summary_seq = np.stack([_get_summary_row(summary, c, cycle_index)
                                 for c in all_cycles])              # (32, 14)

    if dq_scaler is not None:
        T, C, F = dq_seq.shape
        dq_seq      = dq_scaler.transform(dq_seq.reshape(T, C * F)).reshape(T, C, F)
        summary_seq = summary_scaler.transform(summary_seq)

    return (
        torch.tensor(dq_seq,      dtype=torch.float32),   # (32, 1, V_BINS)
        torch.tensor(summary_seq, dtype=torch.float32),   # (32, 14)
    )


# -------------------------------------------------------------------------
# Dataset — index in RAM, data loaded per __getitem__ call
# -------------------------------------------------------------------------
class BMLBatteryClsDataset(Dataset):
    def __init__(
        self,
        cells:      dict,
        cell_ids:   list,
        n_samples:  int             = 500,
        scalers:    Optional[Tuple] = None,
        seed:       int             = 42,
        fit_scaler: bool            = False,
    ):
        self.cells = cells
        self.index = build_sample_index(cells, cell_ids,
                                        seed=seed, n_samples=n_samples)

        # one cache per cell, built lazily on first __getitem__ access
        self._cache: dict[str, _CellCache] = {
            cid: _CellCache() for cid in cell_ids if cid in cells
        }

        print(f"  Total index entries: {len(self.index)}")
        labels = [e[2] for e in self.index]
        for c in range(N_CLASSES):
            print(f"  Class {c}: {sum(l == c for l in labels)} samples")

        if not self.index:
            raise ValueError("No valid samples.")

        if scalers is not None:
            self.dq_scaler, self.summary_scaler = scalers
        elif fit_scaler:
            self.dq_scaler, self.summary_scaler = self._fit_scalers(seed)
        else:
            self.dq_scaler      = None
            self.summary_scaler = None

    # ── Fit scalers by sampling a subset ─────────────────────────────────
    def _fit_scalers(self, seed: int, max_fit: int = 10000):
        print("  Fitting scalers on subset...")
        rng    = np.random.default_rng(seed)
        subset = rng.choice(len(self.index),
                            size=min(max_fit, len(self.index)),
                            replace=False)
        dq_list, sum_list = [], []
        for idx in subset:
            cid, start, _, _ = self.index[idx]
            dq_seq, sum_seq  = _build_sample_tensors(
                self.cells[cid], start, None, None)
            dq_list.append(dq_seq.numpy())
            sum_list.append(sum_seq.numpy())

        dq_arr  = np.concatenate(dq_list,  axis=0)   # (N*32, 1, V_BINS)
        sum_arr = np.concatenate(sum_list, axis=0)   # (N*32, 14)

        N, C, F   = dq_arr.shape
        dq_arr_2d = dq_arr.reshape(N, C * F)         # (N*32, V_BINS)

        dq_scaler      = StandardScaler().fit(dq_arr_2d)
        summary_scaler = StandardScaler().fit(sum_arr)
        print("  Scalers fitted.")
        return dq_scaler, summary_scaler

    def get_scalers(self):
        return self.dq_scaler, self.summary_scaler

    def __len__(self):
        return len(self.index)

    # ── Load one sample on demand ─────────────────────────────────────────
    def __getitem__(self, idx):
        cid, start, label, _ = self.index[idx]
        cache = self._cache[cid]
        if not cache._built:                          # lazy build on first access
            cache.build(self.cells[cid])
        dq, summary = _build_sample_tensors(
            self.cells[cid], start,
            self.dq_scaler, self.summary_scaler,
            cache=cache)
        return {
            "dq":      dq,
            "summary": summary,
            "label":   torch.tensor(label, dtype=torch.long),
        }


# -------------------------------------------------------------------------
# Build DataLoaders
# -------------------------------------------------------------------------
def build_clf_dataloaders(
    content_dir: str,
    batch_size:  int   = 32,
    n_samples:   int   = 600,
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
    test_ids = ids[n_val:2*n_val]
    trn_ids = ids[2*n_val:]

    def _format_cell_list(name: str, ids: list, per_line: int = 4) -> str:
        if not ids:
            return f"{name} = []"
        lines  = [ids[i:i+per_line] for i in range(0, len(ids), per_line)]
        indent = " " * (len(name) + 4)
        inner  = (",\n" + indent).join(", ".join(f"'{n}'" for n in line) for line in lines)
        return f"{name} = [{inner}]"

    print(_format_cell_list("TrainCellName", trn_ids))
    print(_format_cell_list("ValCellName",   val_ids))
    print(_format_cell_list("TestCellName",  test_ids))

    print(f"Split — Train: {len(trn_ids)}  Val: {len(val_ids)}   Test: {len(test_ids)}")


    print("Building train dataset...")
    train_ds = BMLBatteryClsDataset(cells, trn_ids, n_samples,
                                    seed=seed, fit_scaler=True)
    scalers  = train_ds.get_scalers()

    print("Building val dataset...")
    val_ds   = BMLBatteryClsDataset(cells, val_ids, n_samples,
                                    scalers=scalers, seed=seed + 1)

    print("Building test dataset...")
    test_ds = BMLBatteryClsDataset(cells, test_ids, n_samples,
                                    scalers=scalers, seed=seed + 2)

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
    parser.add_argument("--n_samples",    type=int,   default=600)
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

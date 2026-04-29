"""
dataset_clf.py — Battery RUL Classification Dataset

Input:  first 8 cycles + random consecutive 24 cycles → 32 cycles total
Target: 5 classes based on RUL
    0: RUL > 400
    1: RUL 300–400
    2: RUL 200–300
    3: RUL 100–200
    4: RUL < 100
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
from gen_features import V_BINS

N_EARLY    = 8     # always include first 8 cycles
N_RANDOM   = 24    # random consecutive window
N_INPUT    = N_EARLY + N_RANDOM   # 32 total
N_CLASSES  = 5

REF_CYCLE    = 9


def rul_to_class(rul: float) -> int:
    if rul > 400: return 0
    if rul > 300: return 1
    if rul > 200: return 2
    if rul > 100: return 3
    return 4


# -------------------------------------------------------------------------
# Load one .npz file
# -------------------------------------------------------------------------
def _load_npz_cell(path: str) -> dict:
    d = np.load(path)
    return {
        "cycle_life": int(d["cycle_life"]),
        "summary": {
            "QDischarge":     d["qd"],
            "IR":             d["IR"],
            "Tmax":           d["tmax"],
            "Tavg":           d["tavg"],
            "chargetime":     d["chargetime"],
            "dqdv_slope_max": d["dqdv_slope_max"],
            "dqdv_slope_min": d["dqdv_slope_min"],
            "dqdv_min":       d["dqdv_min"],
            "dqdv_avg":       d["dqdv_avg"],
            "log_std_dq":     d["log_std_dq"],
            "log_std_dc":     d["log_std_dc"],
            "log_std_T":      d["log_std_T"],
            "log_std_I":      d["log_std_I"],
            "log_std_ct":     d["log_std_ct"],
        },
        "qdlin": list(d["qdlin"]),   # list of (1000,) arrays 
        "dqdv": list(d["dqdv"]),     # list of (1000,) arrays 
    }


def load_all_npz(content_dir: str) -> Tuple[dict, dict, dict]:
    print("Loading .npz feature files...")
    b1, b2, b3 = {}, {}, {}

    for fname in sorted(os.listdir(content_dir)):
        if not fname.endswith(".npz"):
            continue
        cell_id = fname[:-4]   # e.g. "batch1c000"
        path    = os.path.join(content_dir, fname)
        try:
            cell = _load_npz_cell(path)
            if   cell_id.startswith("batch1"): b1[cell_id] = cell
            elif cell_id.startswith("batch2"): b2[cell_id] = cell
            elif cell_id.startswith("batch3"): b3[cell_id] = cell
        except Exception as e:
            print(f"  WARNING: could not load {fname}: {e}")

    # print(f"Total — train: {len(b1) + len(b2)} cells  test: {len(b3)} cells\n")
    return b1, b2, b3


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
        cell_rng = np.random.default_rng(seed + i + 3) 
        
        cell       = cells[cid]
        cycle_life = int(cell["cycle_life"])
        n_cyc      = min(len(cell["summary"]["QDischarge"]) - 1,
                         len(cell["qdlin"]))

        if n_cyc < N_INPUT:
            continue

        max_start   = n_cyc - N_RANDOM
        class_starts = {c: [] for c in range(N_CLASSES)}

        for start in range(N_EARLY, max_start + 1):
            if start + N_RANDOM > n_cyc - 4:
                continue
            rul   = max(0, cycle_life - (start + N_RANDOM))
            label = rul_to_class(rul)
            class_starts[label].append(start)

        n_per_class = max(1, n_samples // N_CLASSES)
        for label, starts_list in class_starts.items():
            if not starts_list:
                continue
            pick = cell_rng.choice(starts_list,
                              size=min(n_per_class, len(starts_list)),
                              replace=False)
            for s in pick:
                index.append((cid, int(s), label,
                               max(0, cycle_life - (int(s) + N_RANDOM))))

    return index


# -------------------------------------------------------------------------
# Per-cell cache — built once on first access, sliced per __getitem__
# -------------------------------------------------------------------------
class _CellCache:
    """
    Pre-builds all cycle arrays for a cell on first access.
    Subsequent __getitem__ calls only slice — no numpy reconstruction.
        dq_all      : (n_cyc, 2, V_BINS)  — [qdlin-ref, dqdv] stacked
        summary_all : (n_cyc, 18)          — scalar feats + PE
    """
    __slots__ = ("dq_all", "summary_all", "_built")

    def __init__(self):
        self.dq_all      = None
        self.summary_all = None
        self._built      = False

    def build(self, cell: dict) -> None:
        if self._built:
            return
        qdlin_list = cell["qdlin"]
        dqdv_list  = cell["dqdv"]
        summary    = cell["summary"]
        n_cyc      = len(qdlin_list)
        ref_qdlin  = np.array(qdlin_list[REF_CYCLE], dtype=np.float32).reshape(-1)

        dq_arr      = np.empty((n_cyc, V_BINS), dtype=np.float32)
        dqdv_arr    = np.empty((n_cyc, V_BINS), dtype=np.float32)
        summary_arr = np.empty((n_cyc, 14),     dtype=np.float32)

        for c in range(n_cyc):
            q              = np.array(qdlin_list[c], dtype=np.float32).reshape(-1)
            dq_arr[c]      = q - ref_qdlin
            dqdv_arr[c]    = np.array(dqdv_list[c], dtype=np.float32).reshape(-1)
            summary_arr[c] = _get_summary_row(summary, c)

        # self.dq_all      = np.stack([dq_arr, dqdv_arr], axis=1)  # (n_cyc, 2, V_BINS)
        self.dq_all = np.expand_dims(dq_arr,   axis=1)
        self.summary_all = summary_arr                            
        self._built      = True


# -------------------------------------------------------------------------
# Feature helpers — called on-the-fly per batch
# -------------------------------------------------------------------------
def _get_dq(qdlin_list, ref_qdlin, c):
    q   = np.array(qdlin_list[c], dtype=np.float32).reshape(-1)
    out = q - ref_qdlin
    return out


def _get_summary_row(summary, c, d_pos=5):
    pos = c + 1
    def safe(arr):
        return float(arr[pos]) if pos < len(arr) else 0.0

    pe = np.array([
        np.sin(pos / 3000 ** (2 * i / d_pos)) if i % 2 == 0 else
        np.cos(pos / 3000 ** ((2 * i - 1) / d_pos))
        for i in range(1, d_pos)
    ], dtype=np.float32)

    scalar_feats = np.array([
        safe(summary["QDischarge"]), 
        #safe(summary["IR"]),
        #safe(summary["Tmax"]),       safe(summary["Tavg"]),
        safe(summary["chargetime"]),
        safe(summary["dqdv_slope_max"]), safe(summary["dqdv_slope_min"]),
        safe(summary["dqdv_min"]),       safe(summary["dqdv_avg"]),
        safe(summary["log_std_dq"]),     safe(summary["log_std_dc"]),
        #safe(summary["log_std_T"]),     
        safe(summary["log_std_I"]),
        safe(summary["log_std_ct"]),
    ], dtype=np.float32)
    
    return np.concatenate([scalar_feats, pe])   # (14,)


def _build_sample_tensors(cell: dict, start: int,
                           dq_scaler, summary_scaler,
                           cache: Optional[_CellCache] = None) -> Tuple:
    """Load and featurise one sample on-the-fly."""
    
    window_cycles = list(range(start, start + N_RANDOM))
    all_cycles    = list(range(N_EARLY)) + window_cycles   # 32 total

    if cache is not None and cache._built:
        # fast path: just slice pre-built arrays
        dq_seq      = cache.dq_all[all_cycles]             # (32, 2, V_BINS)
        summary_seq = cache.summary_all[all_cycles]        # (32, 18)
    else:
        # slow path: build on-the-fly (used during scaler fitting)
        qdlin_list = cell["qdlin"]
        dqdv_list  = cell["dqdv"]
        summary    = cell["summary"]
        ref_qdlin  = np.array(qdlin_list[REF_CYCLE], dtype=np.float32).reshape(-1)

        dq_seq      = np.stack([_get_dq(qdlin_list, ref_qdlin, c)
                                 for c in all_cycles])            # (32, V_BINS)
        dqdv_seq    = np.stack([np.array(dqdv_list[c], dtype=np.float32).reshape(-1)
                                 for c in all_cycles])            # (32, V_BINS)
        summary_seq = np.stack([_get_summary_row(summary, c)
                                 for c in all_cycles])            # (32, 18)

        dq_seq   = np.expand_dims(dq_seq,   axis=1)   
        dqdv_seq = np.expand_dims(dqdv_seq, axis=1) 
        # dq_seq   = np.concatenate([dq_seq, dqdv_seq], axis=1)

    if dq_scaler is not None:
        T, C, F = dq_seq.shape
        dq_seq = dq_scaler.transform(dq_seq.reshape(T, C * F))
        dq_seq = dq_seq.reshape(T, C, F)

        summary_seq = summary_scaler.transform(summary_seq)

    return (
        torch.tensor(dq_seq,      dtype=torch.float32),      # (32, 2, V_BINS)
        torch.tensor(summary_seq, dtype=torch.float32),     # (32, 18)
    )


# -------------------------------------------------------------------------
# Dataset — index in RAM, data loaded per __getitem__ call
# -------------------------------------------------------------------------
class MITBatteryClsDataset(Dataset):
    def __init__(
        self,
        cells:      dict,
        cell_ids:   list,
        n_samples:  int             = 500,
        scalers:    Optional[Tuple] = None,
        seed:       int             = 42,
        fit_scaler: bool            = False,
    ):
        self.cells   = cells
        self.index   = build_sample_index(cells, cell_ids,
                                          seed=seed, n_samples=n_samples)

        # one cache per cell, built lazily on first __getitem__ access
        self._cache: dict[str, _CellCache] = {
            cid: _CellCache() for cid in cell_ids if cid in cells
        }

        print(f"  Total index entries: {len(self.index)}")
        labels = [e[2] for e in self.index]
        for c in range(N_CLASSES):
            print(f"  Class {c}: {sum(l == c for l in labels)} samples")

        if scalers is not None:
            self.dq_scaler, self.summary_scaler = scalers
        elif fit_scaler:
            self.dq_scaler, self.summary_scaler = self._fit_scalers(seed)
        else:
            self.dq_scaler      = None
            self.summary_scaler = None

    # ── Fit scalers by sampling a subset ─────────────────────────────────
    def _fit_scalers(self, seed: int, max_fit: int = 2000):
        print("  Fitting scalers on subset...")
        
        # subset   = [i for i in range(len(self.index))] 
        rng      = np.random.default_rng(seed)
        subset = rng.choice(len(self.index),size=min(max_fit, len(self.index)), replace=False)
        
        dq_list, dqdv_list, sum_list = [], [], []
        for idx in subset:
            cid, start, _, _ = self.index[idx]
            dq_seq, sum_seq  = _build_sample_tensors(
                self.cells[cid], start, None, None)
            dq_list.append(dq_seq.numpy())
            sum_list.append(sum_seq.numpy())

        dq_arr   = np.concatenate(dq_list,  axis=0)     # (N*32, 2, 1000)
        sum_arr  = np.concatenate(sum_list, axis=0)     # (N*32, 18)
        
        
 
        N, C, F = dq_arr.shape
        dq_arr_2d = dq_arr.reshape(N, C * F)            # (N*32, 2000)


        dq_scaler        = StandardScaler().fit(dq_arr_2d)
        summary_scaler   = StandardScaler().fit(sum_arr)
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
        if not cache._built:                              # build on first access
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
    val_ratio:   float = 0.1,
    num_workers: int   = 0,
    seed:        int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple]:

    b1, b2, b3 = load_all_npz(content_dir)
    
    if False: # batch1 + batch12 (train + valid), batch3 (test)

        trainval = {**b1, **b2}
        ids      = list(trainval.keys())
        rng      = np.random.default_rng(seed)
        rng.shuffle(ids)
        n_val    = max(1, int(len(ids) * val_ratio))
        val_ids  = ids[:n_val]
        trn_ids  = ids[n_val:]
        tst_ids  = list(b3.keys())

        print(f"Split — Train: {len(trn_ids)}  Val: {len(val_ids)}  Test: {len(tst_ids)}")
        
    else: # train: 80%, test 20%
        trainval   = {**b1, **b2, **b3}

        all_ids = list(trainval.keys())
        rng     = np.random.default_rng(seed)
        rng.shuffle(all_ids)
        n_val   = max(1, int(len(all_ids) * val_ratio))
        
        
        trn_ids = all_ids[2*n_val:]
        val_ids = all_ids[:n_val]
        tst_ids = all_ids[n_val:2*n_val]
        
        per_line = 4
        lines = [val_ids[i:i+per_line] for i in range(0, len(val_ids), per_line)]
        inner = ",\n                ".join(", ".join(f"'{n}'" for n in line) for line in lines)
        print(f"ValidationCellName = [{inner}]")

        print(f"Split — Train: {len(trn_ids)}  Val: {len(val_ids)} Test: {len(tst_ids)}")
        
        print(f'trn_ids:{trn_ids}')
        print(f'test_ids:{tst_ids}')

    print("Building train dataset...")
    train_ds = MITBatteryClsDataset(trainval, trn_ids, n_samples,
                                     seed=seed, fit_scaler=True)
    scalers  = train_ds.get_scalers()

    print("Building val dataset...")
    val_ds   = MITBatteryClsDataset(trainval, val_ids, n_samples,
                                     scalers=scalers, seed=seed + 1)

    print("Building test dataset...")
    test_ds  = MITBatteryClsDataset(trainval, val_ids, n_samples,
                                     scalers=scalers, seed=seed + 2)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        scalers,
    )
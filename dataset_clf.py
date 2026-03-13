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


N_EARLY    = 8     # always include first 8 cycles
N_RANDOM   = 24    # random consecutive window
N_INPUT    = N_EARLY + N_RANDOM   # 32 total
N_CLASSES  = 5

V_BINS       = 1000
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
            "QDischarge": d["qd"],
            "IR":         d["ir"],
            "Tmax":       d["tmax"],
            "Tavg":       d["tavg"],
            "chargetime": d["chargetime"],
            "dqdv_max":   d["dqdv_max"],
            "dqdv_min":   d["dqdv_min"],
            "dqdv_avg":   d["dqdv_avg"],
            "log_std_dq":   d["log_std_dq"],
            "log_std_T":   d["log_std_T"],
            "log_std_ir":   d["log_std_ir"],
            "log_std_ct":   d["log_std_ct"],
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

    print(f"Total — train: {len(b1) + len(b2)} cells  test: {len(b3)} cells\n")
    return b1, b2, b3


# -------------------------------------------------------------------------
# One sample = (early_8 + random_24, label)
# We generate multiple samples per cell by sampling different windows
# -------------------------------------------------------------------------
def extract_clf_samples(cell: dict, n_samples: int = 10, seed: int = None) -> list:
    """
    Extract multiple classification samples from one cell.
    Each sample uses first 8 cycles + a random consecutive 24-cycle window.
    The window is drawn from cycles 8 onwards (after the early window).

    Returns list of dicts with keys: dq, summary, label
    """
    rng = np.random.default_rng(seed)

    cycle_life = int(cell["cycle_life"])
    summary    = cell["summary"]
    qdlin_list = cell["qdlin"]
    

    qd  = np.array(summary["QDischarge"], dtype=np.float32).reshape(-1)
    ir  = np.array(summary["IR"],         dtype=np.float32).reshape(-1)
    tmx = np.array(summary["Tmax"],       dtype=np.float32).reshape(-1)
    tav = np.array(summary["Tavg"],       dtype=np.float32).reshape(-1)
    ct  = np.array(summary["chargetime"], dtype=np.float32).reshape(-1)
    dqdv_max = np.array(summary["dqdv_max"], dtype=np.float32).reshape(-1)
    dqdv_min = np.array(summary["dqdv_min"], dtype=np.float32).reshape(-1)
    dqdv_avg = np.array(summary["dqdv_avg"], dtype=np.float32).reshape(-1)
    log_std_dq = np.array(summary["log_std_dq"], dtype=np.float32).reshape(-1)
    log_std_T = np.array(summary["log_std_T"], dtype=np.float32).reshape(-1)
    log_std_ir = np.array(summary["log_std_ir"], dtype=np.float32).reshape(-1)
    log_std_ct = np.array(summary["log_std_ct"], dtype=np.float32).reshape(-1)

    n_cyc = min(len(qd) - 1, len(qdlin_list))   # usable cycles

    # need at least early + random window
    if n_cyc < N_INPUT:
        return []

    cap_ref = qd[1] if qd[1] > 0 else 1.1

    ref_qdlin = np.array(qdlin_list[REF_CYCLE], dtype=np.float32).reshape(-1)

    def get_dq(c):
        q      = np.array(qdlin_list[c], dtype=np.float32).reshape(-1)
        length = min(len(q), len(ref_qdlin), V_BINS)
        out    = np.zeros(V_BINS, dtype=np.float32)
        out[:length] = q[:length] - ref_qdlin[:length]
        return out
    
    def get_summary_row(c, d_pos=4):
        pos = c + 1
        def safe(arr):
            return float(arr[pos]) if pos < len(arr) else 0.0
        
        pe  = np.array([
            np.sin(pos / 3000 ** (2 * i / d_pos)) if i % 2 == 0 else
            np.cos(pos / 3000 ** ((2 * i - 1) / d_pos))
            for i in range(d_pos)
        ], dtype=np.float32)   # (4,)


        scalar_feats = np.array([
            safe(qd), safe(ir), safe(tmx), safe(tav), safe(ct),
            safe(dqdv_max), safe(dqdv_min), safe(dqdv_avg),
            safe(log_std_dq), safe(log_std_T), safe(log_std_ir), safe(log_std_ct),
        ], dtype=np.float32)   # (12,)

        return np.concatenate([scalar_feats, pe])   # (16,)


    samples = []

    # ── Group all valid window starts by class ────────────────────────────
    max_start = n_cyc - N_RANDOM
    class_starts = {i: [] for i in range(N_CLASSES)}
    for start in range(N_EARLY, max_start + 1):
        if start + N_RANDOM > n_cyc - 4: # for safety
            continue
        rul   = max(0, cycle_life - (start + N_RANDOM))
        label = rul_to_class(rul)
        class_starts[label].append(start)

    # ── Sample equally from each class ────────────────────────────────────
    n_per_class = max(1, n_samples // N_CLASSES)
    chosen = []   # list of (start, label)
    for label, starts_for_class in class_starts.items():
        if not starts_for_class:
            continue
        pick = rng.choice(starts_for_class,
                          size=min(n_per_class, len(starts_for_class)),
                          replace=False)
        chosen.extend([(int(s), label) for s in pick])

    # top-up to n_samples if needed
    # all_valid = [(s, rul_to_class(max(0, cycle_life - s - N_RANDOM)))
    #              for s in range(N_EARLY, max_start + 1)
    #              if s + N_RANDOM <= n_cyc - 1]
    # while len(chosen) < n_samples and all_valid:
    #     chosen.append(all_valid[rng.integers(0, len(all_valid))])

    # ── Build sample dicts ────────────────────────────────────────────────
    samples = []
    for start, label in chosen:
        window_cycles = list(range(start, start + N_RANDOM))
        all_cycles    = list(range(N_EARLY)) + window_cycles   # 32 total

        dq_seq      = np.stack([get_dq(c)          for c in all_cycles])  # (32, 1000)
        summary_seq = np.stack([get_summary_row(c) for c in all_cycles])  # (32, 16)

        samples.append({
            "dq":      dq_seq,
            "summary": summary_seq,
            "label":   label,
            "rul":     max(0, cycle_life - (start + N_RANDOM)),
        })

    return samples


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class MITBatteryClsDataset(Dataset):
    def __init__(
        self,
        cells:      dict,
        cell_ids:   list,
        n_samples:  int            = 500,
        scalers:    Optional[Tuple]= None,
        seed:       int            = 42,
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
        summary = np.stack([r["summary"] for r in raw])   # (N, 32, 9)
        labels  = np.array([r["label"]   for r in raw], dtype=np.int64)

        N, T, F = summary.shape

        # print class distribution
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
        
        trainval = {**b1, **b2, **b3}
        
        all_ids  = list(trainval.keys())
        rng      = np.random.default_rng(seed)
        rng.shuffle(all_ids)
        n_val    = max(1, int(len(all_ids) * 0.2))
        val_ids  = all_ids[:n_val]
        trn_ids  = all_ids[n_val:]
        tst_ids  = val_ids   # test set == validation set
        print(f"Split — Train: {len(trn_ids)}  Val/Test: {len(val_ids)}")
        print(f"Val/Test: {val_ids}")

    print("Building train dataset...")
    train_ds = MITBatteryClsDataset(trainval, trn_ids, n_samples, seed=seed)
    scalers  = train_ds.get_scalers()
    
    print("Building val dataset...")
    val_ds   = MITBatteryClsDataset(trainval, val_ids, n_samples, scalers=scalers, seed=seed+1)

    print("Building test dataset...")
    test_ds  = val_ds #MITBatteryClsDataset(trainval, tst_ids, n_samples, scalers=None, seed=seed+2)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        scalers,
    )

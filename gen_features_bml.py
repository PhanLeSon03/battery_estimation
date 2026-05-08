"""
gen_features_bml.py - Extract BatteryML .pkl files into MIT-style .npz features.

The BML raw files do not consistently provide temperature or internal
resistance, so this exporter intentionally omits those fields.  It keeps the
same core shape used by the MIT pipeline:

    qdlin:      (N_cycles, 1000)
    dqdv:       (N_cycles, 1000)
    per-cycle summary arrays used by dataset_clf_bml.py

Usage:
    python gen_features_bml.py --data_dir ./Raw_BML --out_dir ./content_bml
    python gen_features_bml.py --data_dir /home/jupyter/sonpl/Cuong_Battery/battery_estimation/Raw_BML/MATR --out_dir ./content_bml/MATR

Useful variants:
    python gen_features_bml.py --data_dir ./Raw_BML --out_dir content_bml --max_files 5

The --max_files option is only for smoke testing.
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


V_BINS = 1000
LABEL_DIR_NAMES = ("Life labels", "Life labels 2")
EOL_FRACTION = 0.80   # capacity retention threshold that defines end-of-life
MIN_CUR  = 1e-1

def _find_eol_idx(qd: np.ndarray, eol_fraction: float = EOL_FRACTION) -> int:
    """Return the array index that marks end-of-life for this cell.

    Q_initial = max Qd over the first min(10, n) positive finite cycles
    (skips formation anomalies and any early dip).

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

    q_eol     = eol_fraction * q_init
    ref_end   = valid_idx[n_ref - 1]

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


def _interp_nan(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1).copy()
    if arr.size == 0:
        return arr
    nans = ~np.isfinite(arr)
    if not nans.any():
        return arr
    idx = np.arange(arr.size)
    valid = ~nans
    if not valid.any():
        arr[:] = 0.0
        return arr
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr


def _safe_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    try:
        return _interp_nan(np.asarray(value, dtype=np.float32).reshape(-1))
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _log_std(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return float(20.0 * np.log10(max(float(np.std(arr)), 1e-9)))


def _interp(curve: np.ndarray, n: int = V_BINS) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if curve.size == n:
        return curve
    if curve.size == 0:
        return np.zeros(n, dtype=np.float32)
    # linear interpolation to exactly n points
    x_old = np.linspace(0, 1, curve.size)
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, curve).astype(np.float32)


def _collect_voltage_limits(cell: dict, cycles: list[dict]) -> tuple[float, float]:
    vmin = _safe_float(cell.get("min_voltage_limit_in_V"), np.nan)
    vmax = _safe_float(cell.get("max_voltage_limit_in_V"), np.nan)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        return vmin, vmax

    samples = []
    for cyc in cycles[: min(len(cycles), 50)]:
        v = _safe_array(cyc.get("voltage_in_V"))
        current = _safe_array(cyc.get("current_in_A"))
        n = min(v.size, current.size)
        if n >= 2:
            discharge = np.isfinite(v[:n]) & np.isfinite(current[:n]) & (current[:n] < -MIN_CUR)
            if discharge.sum() >= 2:
                samples.append(v[:n][discharge])
                continue
        if v.size:
            samples.append(v[np.isfinite(v)])
    if not samples:
        return 0.0, 1.0

    all_v = np.concatenate(samples)
    if all_v.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(all_v, 1))
    hi = float(np.nanpercentile(all_v, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _dedupe_interp(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return np.zeros(grid.size, dtype=np.float32)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size < 2:
        return np.zeros(grid.size, dtype=np.float32)

    y_sum = np.zeros(unique_x.size, dtype=np.float64)
    counts = np.zeros(unique_x.size, dtype=np.float64)
    np.add.at(y_sum, inverse, y)
    np.add.at(counts, inverse, 1.0)
    unique_y = y_sum / np.maximum(counts, 1.0)

    return np.interp(grid, unique_x, unique_y).astype(np.float32)




def _discharge_arrays(cycle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    qd      = _safe_array(cycle.get("discharge_capacity_in_Ah"))
    
    voltage = _safe_array(cycle.get("voltage_in_V"))
    current = _safe_array(cycle.get("current_in_A"))
    time_s  = _safe_array(cycle.get("time_in_s"))
    

    n = min(voltage.size, qd.size, current.size, time_s.size)

    voltage = voltage[:n]
    qd      = qd[:n]
    current = current[:n]
    time_s  = time_s[:n]
    valid     = np.isfinite(voltage) & np.isfinite(qd) & np.isfinite(current) & np.isfinite(time_s)
    discharge = valid & (current < -MIN_CUR)
    
    
    discharge_idx = np.where(discharge)
    
    voltage = voltage[discharge_idx]
    qd      = qd[discharge_idx]
    current = current[discharge_idx]
    time_s  = time_s[discharge_idx]
    time_s  = time_s - time_s[0]


    # print('_discharge_arrays')
    # print(f'voltage.size {voltage.size}, qd.size {qd.size}, current.size {current.size}, time_s.size {time_s.size}')
    # print(current)
    

    return voltage, qd, current, time_s


def _charge_arrays(cycle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voltage = _safe_array(cycle.get("voltage_in_V"))
    qc      = _safe_array(cycle.get("charge_capacity_in_Ah"))
    current = _safe_array(cycle.get("current_in_A"))
    time_s  = _safe_array(cycle.get("time_in_s"))
    
    n = min(voltage.size, qc.size, current.size, time_s.size)

    voltage = voltage[:n]
    qc      = qc[:n]
    current = current[:n]
    time_s  = time_s[:n]
    
    valid  = np.isfinite(voltage) & np.isfinite(qc) & np.isfinite(current) & np.isfinite(time_s)
    charge = valid & (current > MIN_CUR)
    
    charge_idx = np.where(charge)
    
    voltage = voltage[charge_idx]
    qc      = qc[charge_idx]
    current = current[charge_idx]
    time_s  = time_s[charge_idx]
    time_s  = time_s - time_s[0]
    
    # print('_charge_arrays')
    # print(f'voltage.size {voltage.size}, qc.size {qc.size}, current.size {current.size}, time_s.size {time_s.size}')
    # print(current)
 
    return voltage, qc, current, time_s


def _cycle_qdlin(cycle: dict, voltage_grid: np.ndarray) -> np.ndarray:
    voltage, qd, _, _ = _discharge_arrays(cycle)
    if voltage.size < 2:
        return np.zeros(V_BINS, dtype=np.float32)

    usable = np.isfinite(voltage) & np.isfinite(qd)
    if usable.sum() < 2:
        return np.zeros(V_BINS, dtype=np.float32)

    qd_max = float(np.nanmax(qd[usable]))
    if qd_max <= 0:
        return np.zeros(V_BINS, dtype=np.float32)

    return _dedupe_interp(voltage[usable], qd[usable], voltage_grid)


def _cycle_charge_time(cycle: dict) -> float:
    time_s = _safe_array(cycle.get("time_in_s"))
    current = _safe_array(cycle.get("current_in_A"))
    n = min(time_s.size, current.size)
    if n < 2:
        return 0.0

    time_s = time_s[:n]
    current = current[:n]
    valid = np.isfinite(time_s) & np.isfinite(current)
    if valid.sum() < 2:
        return 0.0

    charge = valid & (current > MIN_CUR)
    if charge.sum() >= 2:
        return float(np.nanmax(time_s[charge]) - np.nanmin(time_s[charge]))
    return 0.0


def _cycle_number(cycle: dict, idx: int, already_spent: int) -> int:
    raw = cycle.get("cycle_number", idx + 1)
    try:
        cycle_num = int(raw)
    except Exception:
        cycle_num = idx + 1
    if already_spent > 0 and cycle_num <= idx + 2:
        return already_spent + cycle_num
    return cycle_num


def _normalization_keys(path: Path) -> list[str]:
    name = path.name
    stem = path.stem
    keys = {name, stem}
    keys.add(re.sub(r"--(\d+)$", r"-#\1", name))
    keys.add(re.sub(r"--(\d+)$", r"-#\1", stem))
    keys.add(name.replace("--", "-#"))
    keys.add(stem.replace("--", "-#"))
    return list(keys)


def load_life_labels(data_dir: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for dirname in LABEL_DIR_NAMES:
        label_dir = data_dir / dirname
        if not label_dir.is_dir():
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            if json_path.name.startswith("._"):
                continue
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"  WARNING: could not read {json_path}: {exc}")
                continue
            for key, value in data.items():
                try:
                    labels[key] = int(value)
                    labels[Path(key).stem] = int(value)
                except Exception:
                    continue
    return labels


def find_cycle_life(path: Path, cell: dict, cycles: list[dict], labels: dict[str, int]) -> int:
    for key in _normalization_keys(path):
        if key in labels:
            return int(labels[key])

    cell_id = str(cell.get("cell_id", path.stem))
    for key in (cell_id, f"{cell_id}.pkl"):
        if key in labels:
            return int(labels[key])

    cycle_numbers = [
        _cycle_number(cyc, i, int(cell.get("already_spent_cycles") or 0))
        for i, cyc in enumerate(cycles)
    ]
    if cycle_numbers:
        return int(max(cycle_numbers))
    return int(len(cycles))


def extract_pkl(path: Path, data_dir: Path, out_dir: Path, labels: dict[str, int]) -> bool:
    if path.stat().st_size == 0:
        print(f"  SKIP empty file: {path}")
        return False

    with path.open("rb") as f:
        cell = pickle.load(f)
    if not isinstance(cell, dict):
        print(f"  SKIP not a dict: {path}")
        return False

    cycles = cell.get("cycle_data")
    if not isinstance(cycles, list) or not cycles:
        print(f"  SKIP no cycle_data: {path}")
        return False

    vmin, vmax = _collect_voltage_limits(cell, cycles)
    voltage_grid = np.linspace(vmin, vmax, V_BINS, dtype=np.float32)
    already_spent = int(cell.get("already_spent_cycles") or 0)

    Qd = []
    Qc = []
    c_t = []
    dqdv_slope_max = []
    dqdv_slope_min = []
    dqdv_min = []
    dqdv_avg = []
    log_std_Qd = []
    log_std_Qc = []
    log_std_Id = []
    log_std_Ic = []
    dc_t = []
    qdlin_list = []
    dqdv_list = []
    cycle_index = []

    for idx, cycle in enumerate(cycles):
        if not isinstance(cycle, dict):
            continue

        _, dq_raw, Id, discharge_time = _discharge_arrays(cycle)
        _, cq_raw, Ic, charge_time = _charge_arrays(cycle)

        qd_curve = _cycle_qdlin(cycle, voltage_grid)
        
        dqdv_curve = np.gradient(qd_curve, voltage_grid).astype(np.float32)
        dqdv_curve = _interp_nan(dqdv_curve)

        Qd.append(float(np.nanmax(dq_raw)) if dq_raw.size else 0.0)
        Qc.append(float(np.nanmax(cq_raw)) if dq_raw.size else 0.0)
        c_t.append(charge_time[-1])
#         print(f'charge_time: {charge_time[-1]}')
#         print(f'discharge_time: {discharge_time[-1]}')
        
        dqdv  = dqdv_curve[100:900] # get midle window                        
        dqdv = np.convolve(dqdv, np.ones(10)/10, mode='valid') 
        dqdv_slope = np.diff(dqdv)
        dqdv_slope_max.append(float(np.max(dqdv_slope)))
        dqdv_slope_min.append(float(np.min(dqdv_slope)))
        
        dqdv_min.append(float(np.nanmin(dqdv_curve)) if dqdv_curve.size else 0.0)
        dqdv_avg.append(float(np.nanmean(dqdv_curve)) if dqdv_curve.size else 0.0)
        log_std_Qd.append(_log_std(dq_raw))
        log_std_Qc.append(_log_std(cq_raw))
        log_std_Id.append(_log_std(Id))
        log_std_Ic.append(_log_std(Ic))
        dc_t.append(discharge_time[-1])
        qdlin_list.append(_interp(qd_curve))
        dqdv_list.append(_interp(dqdv_curve))
        cycle_index.append(_cycle_number(cycle, idx, already_spent))

    if len(qdlin_list) == 0:
        print(f"  SKIP no usable cycles: {path}")
        return False

    # ── Truncate arrays at EOL (80 % of Q_init, or min-Qd fallback) ──────
    qd_arr  = _interp_nan(np.asarray(Qd, dtype=np.float32))
    qc_arr  = _interp_nan(np.asarray(Qc, dtype=np.float32))
    
    ci_arr  = np.asarray(cycle_index, dtype=np.int32)
    eol_idx = _find_eol_idx(qd_arr)
    n_keep  = min(eol_idx + 1, len(qd_arr))   # include the EOL cycle itself

    if eol_idx < len(qd_arr):
        cycle_life = int(ci_arr[eol_idx]) if ci_arr.size else int(eol_idx + 1)
        # Report whether 80 % was actually crossed or we used the min-Qd fallback
        q_init   = float(np.max(qd_arr[: min(10, len(qd_arr))]))
        retained = qd_arr[eol_idx] / q_init if q_init > 0 else 0.0
        if retained < EOL_FRACTION:
            print(f"    EOL at idx {eol_idx} → cycle {cycle_life}  "
                  f"(Qd at {retained*100:.1f}% of initial)")
        else:
            print(f"    Cell never aged below {EOL_FRACTION*100:.0f}% — using "
                  f"min-Qd cycle: idx {eol_idx} → cycle {cycle_life}  "
                  f"(Qd at {retained*100:.1f}% of initial)")
            if retained > EOL_FRACTION + 0.05:
                print(f" This cell is not finished its life-cycle ...")
                return False
    else:
        cycle_life = int(ci_arr[-1]) if ci_arr.size else int(len(qd_arr))
        print(f"    No usable Qd data; using last cycle {cycle_life}")
  
    # cycle_life = labels[cell_id]
    # n_keep = len(ci_arr)
    # for idx, ci in enumerate(ci_arr):
    #     if ci >= cycle_life:
    #         n_keep = idx
    #         break

    cell_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cell.get("cell_id") or path.stem))
    rel_parent = path.parent.relative_to(data_dir)
    cell_out_dir = out_dir / rel_parent
    cell_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cell_out_dir / f"{cell_id}.npz"
    np.savez_compressed(
        out_path,
        source_file=str(path),
        cell_id=cell_id,
        cycle_life=np.array(cycle_life, dtype=np.int32),
        cycle_index=ci_arr[:n_keep],
        qd         = qd_arr[:n_keep],
        qc         = qc_arr[:n_keep],
        IR         = np.zeros_like(qd_arr[:n_keep]), # dummy values
        tmax       = np.zeros_like(qd_arr[:n_keep]), # dummy values
        tavg       = np.zeros_like(qd_arr[:n_keep]), # dummy values
        c_t        =_interp_nan(np.asarray(c_t,  dtype=np.float32))[:n_keep],
        dqdv_slope_max  =_interp_nan(np.asarray(dqdv_slope_max,   dtype=np.float32))[:n_keep],
        dqdv_slope_min  =_interp_nan(np.asarray(dqdv_slope_min,   dtype=np.float32))[:n_keep],
        dqdv_min  =_interp_nan(np.asarray(dqdv_min,   dtype=np.float32))[:n_keep],
        dqdv_avg  =_interp_nan(np.asarray(dqdv_avg,   dtype=np.float32))[:n_keep],
        log_std_Qd=_interp_nan(np.asarray(log_std_Qd, dtype=np.float32))[:n_keep],
        log_std_Qc=_interp_nan(np.asarray(log_std_Qc, dtype=np.float32))[:n_keep],
        log_std_Id =_interp_nan(np.asarray(log_std_Id,  dtype=np.float32))[:n_keep],
        log_std_Ic =_interp_nan(np.asarray(log_std_Ic,  dtype=np.float32))[:n_keep],
        dc_t      =_interp_nan(np.asarray(dc_t, dtype=np.float32))[:n_keep],
        qdlin     =np.stack(qdlin_list, axis=0).astype(np.float32)[:n_keep],
        dqdv =    np.stack(dqdv_list,  axis=0).astype(np.float32)[:n_keep],
    )
    return True


def iter_pkl_files(data_dir: Path) -> list[Path]:
    skipped_dirs = set(LABEL_DIR_NAMES) | {"READMEs"}
    paths = []
    for path in data_dir.rglob("*.pkl"):
        rel_parts = set(path.relative_to(data_dir).parts[:-1])
        if rel_parts & skipped_dirs:
            continue
        if path.name.startswith("._"):
            continue
        paths.append(path)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./Raw/Raw_BML")
    parser.add_argument("--out_dir", default="./content_bml")
    parser.add_argument("--max_files", type=int, default=None, help="Optional smoke-test limit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_life_labels(data_dir)
    files = iter_pkl_files(data_dir)
    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"Found {len(files)} .pkl files")
    print(f"Loaded {len(labels)} life-label aliases")

    saved = 0
    failed = 0
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path}")
        try:
            if extract_pkl(path, data_dir, out_dir, labels):
                saved += 1
        except Exception as exc:
            failed += 1
            print(f"  WARNING: failed {path}: {exc}")

    print(f"\nDone. Saved: {saved}  Failed: {failed}  Output: {out_dir}")


if __name__ == "__main__":
    main()

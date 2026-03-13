"""
gen_features.py — Extract features from MIT battery .mat files and save .npz per cell

Usage:
    python gen_features.py --data_dir ./data --out_dir ./content

Output:
    ./content/batch1c000.npz
    ./content/batch1c001.npz
    ...

Each .npz contains exactly what load_mat_batch() built per cell:
    cycle_life   scalar int
    qd           (N_cycles,) float32
    ir           (N_cycles,) float32
    tmax         (N_cycles,) float32
    tavg         (N_cycles,) float32
    chargetime   (N_cycles,) float32
    dqdv_max     (N_cycles,) float32
    dqdv_min     (N_cycles,) float32
    dqdv_avg     (N_cycles,) float32
    qdlin        (N_cycles, 1000) float32
"""

import os
import argparse
import numpy as np
import h5py

V_BINS = 1000

MAT_FILES = {
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat": "batch1",
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat": "batch2",
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat": "batch3",
}


def _deref_scalar(f, ref) -> float:
    return float(np.array(f[ref]).flat[0])


def _deref_qdlin(f, ref) -> np.ndarray:
    return np.array(f[ref], dtype=np.float32).reshape(-1)


def extract_mat(mat_path: str, batch_prefix: str, out_dir: str):
    print(f"  Loading {os.path.basename(mat_path)} ...", end=" ", flush=True)
    saved = 0

    with h5py.File(mat_path, "r") as f:
        batch   = f["batch"]
        n_cells = batch["cycle_life"].shape[0]

        for i in range(n_cells):
            try:
                cycle_life = int(_deref_scalar(f, batch["cycle_life"][i, 0]))

                sum_grp    = f[batch["summary"][i, 0]]
                qd         = np.array(sum_grp["QDischarge"], dtype=np.float32).reshape(-1)
                ir         = np.array(sum_grp["IR"],         dtype=np.float32).reshape(-1)
                tmax       = np.array(sum_grp["Tmax"],       dtype=np.float32).reshape(-1)
                tavg       = np.array(sum_grp["Tavg"],       dtype=np.float32).reshape(-1)
                chargetime = np.array(sum_grp["chargetime"], dtype=np.float32).reshape(-1)
                
                '''
                === sample values — cell 0, cycle 0 ===
                I                              = [ref] shapes=[(1, 762), (1, 764), (1, 758)]
                Qc                             = [ref] shapes=[(1, 762), (1, 764), (1, 758)]
                Qd                             = [ref] shapes=[(1, 762), (1, 764), (1, 758)]
                Qdlin                          = [ref] shapes=[(1, 1000), (1, 1000), (1, 1000)]
                T                              = [ref] shapes=[(1, 762), (1, 764), (1, 758)]
                Tdlin                          = [ref] shapes=[(1, 1000), (1, 1000), (1, 1000)]
                V                              = [ref] shapes=[(1, 762), (1, 764), (1, 758)]
                discharge_dQdV                 = [ref] shapes=[(1, 1000), (1, 1000), (1, 1000)]
                t                              = [ref] shapes=[(1, 762), (1, 764), (1, 758)]
                '''

                cyc_grp  = f[batch["cycles"][i, 0]]
                qdlin_ds = cyc_grp["Qdlin"]
                dqdv_ds  = cyc_grp["discharge_dQdV"]
                qd_ds    = cyc_grp["Qd"]
                T_ds     = cyc_grp["T"]
                ir_ds    = cyc_grp["I"]
                ct_ds    = cyc_grp["t"]
                
                
                n_cyc    = qdlin_ds.shape[0]

                qdlin_list = []
                dqdv_list  = []
                dqdv_max   = []
                dqdv_min   = []
                dqdv_avg   = []
                log_std_dq = [] #float(10.0 * np.log10(max(safe(qdlin_std), 1e-9)))
                log_std_T  = [] #float(10.0 * np.log10(max(safe(tmx),       1e-9)))
                log_std_ir = [] #float(10.0 * np.log10(max(safe(ir),        1e-9)))
                log_std_ct = [] #float(10.0 * np.log10(max(safe(ct),        1e-9)))

                for j in range(n_cyc):
                    try:
                        curve = _deref_qdlin(f, qdlin_ds[j, 0])
                        if len(curve) != V_BINS:
                            tmp = np.zeros(V_BINS, dtype=np.float32)
                            tmp[:min(len(curve), V_BINS)] = curve[:V_BINS]
                            curve = tmp
                        qdlin_list.append(curve)
                    except Exception:
                        qdlin_list.append(np.zeros(V_BINS, dtype=np.float32))
                        
                        
                    try:
                        curve = _deref_qdlin(f, dqdv_ds[j, 0])
                        if len(curve) != V_BINS:
                            tmp = np.zeros(V_BINS, dtype=np.float32)
                            tmp[:min(len(curve), V_BINS)] = curve[:V_BINS]
                            curve = tmp
                        dqdv_list.append(curve)
                    except Exception:
                        dqdv_list.append(np.zeros(V_BINS, dtype=np.float32))
                        
                    try:
                        dqdv = f[dqdv_ds[j, 0]][()].flatten().astype(np.float32)
                        qd_c  = f[qd_ds[j, 0]][()].flatten().astype(np.float32)
                        T_c   = f[T_ds[j, 0]][()].flatten().astype(np.float32)
                        ir_c  = f[ir_ds[j, 0]][()].flatten().astype(np.float32)
                        ct_c  = f[ct_ds[j, 0]][()].flatten().astype(np.float32)

                        dqdv_max.append(float(np.max(dqdv)))
                        dqdv_min.append(float(np.min(dqdv)))
                        dqdv_avg.append(float(np.mean(dqdv)))
                        log_std_dq.append(float(10.0 * np.log10(np.std(qd_c).clip(1e-9))))
                        log_std_T.append( float(10.0 * np.log10(np.std(T_c).clip(1e-9))))
                        log_std_ir.append(float(10.0 * np.log10(np.std(ir_c).clip(1e-9))))
                        log_std_ct.append(float(10.0 * np.log10(np.std(ct_c).clip(1e-9))))
                    except Exception:
                        dqdv_max.append(0.0)
                        dqdv_min.append(0.0)
                        dqdv_avg.append(0.0)
                        log_std_dq.append(0.0)
                        log_std_T.append(0.0)
                        log_std_ir.append(0.0)
                        log_std_ct.append(0.0)

                out_path = os.path.join(out_dir, f"{batch_prefix}c{i:03d}.npz")
                np.savez_compressed(
                    out_path,
                    cycle_life = np.array(cycle_life),
                    qd         = qd,
                    ir         = ir,
                    tmax       = tmax,
                    tavg       = tavg,
                    chargetime = chargetime,
                    dqdv_max   = np.array(dqdv_max, dtype=np.float32),
                    dqdv_min   = np.array(dqdv_min, dtype=np.float32),
                    dqdv_avg   = np.array(dqdv_avg, dtype=np.float32),
                    log_std_dq = np.array(log_std_dq, dtype=np.float32),
                    log_std_T = np.array(log_std_T, dtype=np.float32),
                    log_std_ir = np.array(log_std_ir, dtype=np.float32),
                    log_std_ct = np.array(log_std_ct, dtype=np.float32),
                    qdlin      = np.stack(qdlin_list, axis=0),   # (N_cycles, 1000)
                    dqdv       = np.stack(dqdv_list, axis=0),    # (N_cycles, 1000)
                )
                saved += 1

            except Exception as e:
                print(f"\n    Skipped cell {i}: {e}")

    print(f"{saved} cells saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--out_dir",  default="./content")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for fname, bname in MAT_FILES.items():
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        extract_mat(path, bname, args.out_dir)

    print(f"\nDone. Files saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
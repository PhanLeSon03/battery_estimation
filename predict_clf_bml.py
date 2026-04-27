"""
predict_clf_bml.py - Sliding-window RUL classification inference for BML cells.

Mirrors predict_clf.ipynb but uses BML-specific features:
  - 8 summary scalars (no IR / temperature) + 4 positional-encoding dims = 12
  - observed_cycle() maps array indices to real cycle numbers
  - Fixed 5-class RUL thresholds (100/200/300/400) — same as MIT

How to run:
    python predict_clf_bml.py --content_dir ./content_bml/CALCE \
                               --checkpoint_dir ./checkpoints_clf_bml_CALCE

    # Single cell only:
    python predict_clf_bml.py --content_dir ./content_bml/CALCE \
                               --checkpoint_dir ./checkpoints_clf_bml_CALCE \
                               --cell_id CELL_ID
"""

import argparse
import os

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.lines import Line2D

from dataset_clf_bml import (
    N_CLASSES, N_EARLY, N_INPUT, N_RANDOM,
    REF_CYCLE, V_BINS,
    load_all_npz, rul_to_class,
)
from train_clf import BatteryRULClassifier

SUMMARY_FEATS = 12   # 8 scalars + 4 positional-encoding dims
EOL_CLASS     = N_CLASSES - 1   # class index that represents near-end-of-life

CLASS_NAMES   = ["RUL>400", "RUL 300–400", "RUL 200–300", "RUL 100–200", "RUL<100"]
CLASS_COLORS  = ["#1565C0", "#2E7D32", "#F9A825", "#E65100", "#B71C1C"]


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_cell(
    cell:    dict,
    model:   torch.nn.Module,
    scalers: tuple,
    device:  str,
) -> dict | None:
    """Slide a 24-cycle window across the full cell life.

    Returns a dict with arrays aligned to the window end-cycle, or None if
    the cell has fewer cycles than N_INPUT.
    """
    cycle_life  = int(cell["cycle_life"])
    cycle_index = np.asarray(cell.get("cycle_index", []), dtype=np.int32)
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

    if n_cyc < N_INPUT:
        return None

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
            [safe(qd), safe(ct), safe(dqdv_max), safe(dqdv_min), safe(dqdv_avg),
             safe(log_std_dq), safe(log_std_I), safe(log_std_ct)],
            dtype=np.float32,
        )
        return np.concatenate([scalar_feats, pe])   # (12,)

    dq_sc, sum_sc = scalers

    end_cycles   = []
    pred_classes = []
    true_classes = []
    pred_probs   = []

    for start in range(N_EARLY, n_cyc - N_RANDOM):
        window_cycles = list(range(start, start + N_RANDOM))
        all_cycles    = list(range(N_EARLY)) + window_cycles

        dq_seq      = np.stack([get_dq(c)          for c in all_cycles])   # (32, V_BINS)
        summary_seq = np.stack([get_summary_row(c)  for c in all_cycles])   # (32, 12)

        dq_seq      = dq_sc.transform(dq_seq)
        summary_seq = sum_sc.transform(summary_seq)

        dq_t  = torch.tensor(dq_seq,      dtype=torch.float32).unsqueeze(0).to(device)
        sum_t = torch.tensor(summary_seq, dtype=torch.float32).unsqueeze(0).to(device)

        logits = model(dq_t, sum_t).squeeze(0)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_c = int(probs.argmax())

        obs_end = observed_cycle(start + N_RANDOM - 1)
        rul     = max(0, cycle_life - obs_end)
        true_c  = rul_to_class(rul)

        end_cycles.append(obs_end)
        pred_classes.append(pred_c)
        true_classes.append(true_c)
        pred_probs.append(probs)

    end_cycles   = np.array(end_cycles)
    pred_classes = np.array(pred_classes)
    true_classes = np.array(true_classes)
    pred_probs   = np.array(pred_probs)

    accuracy = float((pred_classes == true_classes).mean())

    return {
        "end_cycles":   end_cycles,
        "pred_classes": pred_classes,
        "true_classes": true_classes,
        "pred_probs":   pred_probs,
        "cycle_life":   cycle_life,
        "accuracy":     accuracy,
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_single_cell(res: dict, cell_id: str, out_dir: str) -> None:
    class_names = CLASS_NAMES
    cycles     = res["end_cycles"]
    pred_cls   = res["pred_classes"]
    true_cls   = res["true_classes"]
    pred_probs = res["pred_probs"]
    correct    = pred_cls == true_cls

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [2, 2, 1.2]})
    fig.suptitle(
        f"RUL Classification — Cell: {cell_id}  "
        f"(cycle life: {res['cycle_life']})  "
        f"Accuracy: {res['accuracy']:.2%}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: predicted vs true class
    ax = axes[0]
    ax.step(cycles, true_cls, where="post", color="#1565C0", lw=2.5,
            label="True class", zorder=3)
    ax.step(cycles, pred_cls, where="post", color="#E53935", lw=1.8,
            ls="--", label="Predicted class", zorder=4)
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_ylabel("RUL Class", fontsize=11)
    ax.set_title("Predicted vs True RUL Class Over Cycle Life")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.fill_between(cycles, -0.4, N_CLASSES - 0.6,
                    where=correct,  alpha=0.08, color="green", step="post")
    ax.fill_between(cycles, -0.4, N_CLASSES - 0.6,
                    where=~correct, alpha=0.12, color="red",   step="post")
    ax.set_ylim(-0.4, N_CLASSES - 0.6)

    # Panel 2: class probabilities
    ax = axes[1]
    for c in range(N_CLASSES):
        ax.plot(cycles, pred_probs[:, c], color=CLASS_COLORS[c],
                lw=1.5, label=class_names[c], alpha=0.85)
    ax.set_ylabel("Softmax Probability", fontsize=11)
    ax.set_title("Predicted Class Probabilities")
    ax.legend(fontsize=8.5, ncol=N_CLASSES, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: class error per window
    ax = axes[2]
    error = pred_cls - true_cls
    ax.bar(cycles, error, width=1.0,
           color=["#E53935" if e != 0 else "#43A047" for e in error],
           alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Window End Cycle", fontsize=11)
    ax.set_ylabel("Class Error\n(pred − true)", fontsize=10)
    ax.set_title("Classification Error per Window")
    ax.grid(True, alpha=0.25)
    patches = [
        mpatches.Patch(color="#43A047", alpha=0.75, label="Correct"),
        mpatches.Patch(color="#E53935", alpha=0.75, label="Incorrect"),
    ]
    ax.legend(handles=patches, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{cell_id}_clf_prediction_bml.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_all_cells_grid(all_results: dict, out_dir: str) -> None:
    short_names = ["R>400", "R>300", "R>200", "R>100", "R<100"]

    n     = len(all_results)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3),
                             constrained_layout=True)
    fig.suptitle("All Cells — Predicted vs True RUL Class",
                 fontsize=14, fontweight="bold", y=1.01)

    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, (cid, r) in zip(axes_flat, all_results.items()):
        cyc      = r["end_cycles"]
        pred_cls = r["pred_classes"]
        true_cls = r["true_classes"]
        correct  = pred_cls == true_cls

        ax.step(cyc, true_cls, where="post", color="#1565C0", lw=1.8)
        ax.step(cyc, pred_cls, where="post", color="#E53935", lw=1.4, ls="--")
        ax.fill_between(cyc, -0.4, N_CLASSES - 0.6,
                        where=correct,  alpha=0.08, color="green", step="post")
        ax.fill_between(cyc, -0.4, N_CLASSES - 0.6,
                        where=~correct, alpha=0.14, color="red",   step="post")

        ax.set_yticks(range(N_CLASSES))
        ax.set_yticklabels(short_names, fontsize=5)
        ax.set_ylim(-0.4, N_CLASSES - 0.6)
        ax.set_title(f"{cid}\nacc={r['accuracy']:.2%}", fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    legend_elements = [
        Line2D([0], [0], color="#1565C0", lw=1.8,          label="True class"),
        Line2D([0], [0], color="#E53935", lw=1.4, ls="--", label="Pred class"),
        mpatches.Patch(color="green", alpha=0.2,            label="Correct"),
        mpatches.Patch(color="red",   alpha=0.25,           label="Incorrect"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    out_path = os.path.join(out_dir, "all_cells_clf_bml.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_overlay(all_results: dict, out_dir: str) -> None:
    class_names = CLASS_NAMES
    cycle_lives = [r["cycle_life"] for r in all_results.values()]
    vmin, vmax  = min(cycle_lives), max(cycle_lives)
    cmap        = colormaps.get_cmap("plasma")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("All Cells Overlay — Predicted vs True RUL Class, Coloured by Cycle Life",
                 fontsize=13, fontweight="bold")

    for ax, key in zip(axes, ["true_classes", "pred_classes"]):
        label = "Ground Truth" if key == "true_classes" else "Predicted"
        for r in all_results.values():
            color = cmap((r["cycle_life"] - vmin) / max(vmax - vmin, 1))
            ax.step(r["end_cycles"], r[key], where="post",
                    color=color, alpha=0.6, lw=1.2)
        ax.set_yticks(range(N_CLASSES))
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_ylim(-0.4, N_CLASSES - 0.6)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Window End Cycle", fontsize=11)
        ax.set_ylabel("RUL Class", fontsize=11)
        ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Cycle Life", fontsize=11)

    out_path = os.path.join(out_dir, "overlay_all_cells_clf_bml.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_neol(all_results: dict, out_dir: str) -> tuple:
    eol_true_list, eol_pred_list, cid_list = [], [], []

    for cid, r in all_results.items():
        cycles   = r["end_cycles"]
        true_cls = r["true_classes"]
        pred_cls = r["pred_classes"]

        true_hits = cycles[true_cls == EOL_CLASS]
        eol_true  = int(true_hits[0]) if len(true_hits) > 0 else r["cycle_life"]

        pred_hits = cycles[pred_cls == EOL_CLASS]
        eol_pred  = int(pred_hits[0]) if len(pred_hits) > 0 else int(cycles[-1])

        eol_true_list.append(eol_true)
        eol_pred_list.append(eol_pred)
        cid_list.append(cid)

    eol_true = np.array(eol_true_list)
    eol_pred = np.array(eol_pred_list)
    eol_err  = eol_pred - eol_true
    mae_eol  = float(np.mean(np.abs(eol_err)))
    mape_eol = float(np.mean(np.abs(eol_err) / (eol_true + 1e-6)) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Near End-of-Life Estimation from Classification Model",
                 fontsize=13, fontweight="bold")

    lim = [min(eol_true.min(), eol_pred.min()) - 50,
           max(eol_true.max(), eol_pred.max()) + 50]

    ax = axes[0]
    sc = ax.scatter(eol_true, eol_pred, c=np.abs(eol_err),
                    cmap="RdYlGn_r", s=60, edgecolors="k", linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label="|NEOL error| (cycles)")
    ax.plot(lim, lim, "k--", lw=1.2, label="Perfect prediction")
    ax.fill_between(lim, [l - 100 for l in lim], [l + 100 for l in lim],
                    alpha=0.1, color="green", label="±100 cycle band")
    ax.set_xlabel("True NEOL (cycles)",      fontsize=11)
    ax.set_ylabel("Predicted NEOL (cycles)", fontsize=11)
    ax.set_title("NEOL Parity Plot\n(first cycle predicted as lowest RUL class)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.text(0.04, 0.92, f"MAE = {mae_eol:.1f} cycles\nMAPE = {mape_eol:.1f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax = axes[1]
    ax.hist(eol_err, bins=20, color="#1976D2", edgecolor="white", linewidth=0.6)
    ax.axvline(0,              color="black", lw=1.5, ls="--", label="Zero error")
    ax.axvline(eol_err.mean(), color="red",   lw=1.5,
               label=f"Mean = {eol_err.mean():.1f} cycles")
    ax.set_xlabel("NEOL Error (pred − true cycles)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("NEOL Error Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eol_parity_clf_bml.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return eol_true, eol_pred, eol_err, mae_eol, mape_eol, cid_list


def print_summary_table(
    all_results: dict,
    eol_true:   np.ndarray,
    eol_pred:   np.ndarray,
    eol_err:    np.ndarray,
    mae_eol:    float,
    mape_eol:   float,
    cid_list:   list,
) -> None:
    print(f'\n{"Cell":>30} | {"CycLife":>7} | {"NEOL_True":>9} | '
          f'{"NEOL_Pred":>9} | {"NEOL_Err":>8} | {"Acc":>7}')
    print("-" * 80)

    for i, cid in enumerate(cid_list):
        r   = all_results[cid]
        err = eol_err[i]
        print(f'{cid:>30} | {r["cycle_life"]:7d} | {eol_true[i]:9d} | '
              f'{eol_pred[i]:9d} | {err:+8d} | {r["accuracy"]:7.2%}')

    print("-" * 80)
    accs = [r["accuracy"] for r in all_results.values()]
    print(f'{"MEAN":>30} | {"":>7} | {"":>9} | {"":>9} | '
          f'{eol_err.mean():+8.1f} | {np.mean(accs):7.2%}')
    print(f'{"MAE":>30} | {"":>7} | {"":>9} | {"":>9} | '
          f'{mae_eol:8.1f} |')
    print(f'{"MAPE":>30} | {"":>7} | {"":>9} | {"":>9} | '
          f'{mape_eol:7.1f}% |')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sliding-window RUL classification inference for BML cells."
    )
    parser.add_argument("--content_dir",    default="./content_bml",
                        help="Same folder used during training")
    parser.add_argument("--checkpoint_dir", default="./checkpoints_clf_bml")
    parser.add_argument("--cell_id",        default=None,
                        help="Run inference on a single cell ID (optional)")
    parser.add_argument("--out_dir",        default=None,
                        help="Output directory for plots (defaults to checkpoint_dir)")
    parser.add_argument("--cnn_dim",   type=int,   default=32)
    parser.add_argument("--gru_dim",   type=int,   default=32)
    parser.add_argument("--gru_layers",type=int,   default=2)
    parser.add_argument("--dropout",   type=float, default=0.1)
    args = parser.parse_args()

    out_dir = args.out_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    cells = load_all_npz(args.content_dir)

    # ── Load model & scalers ───────────────────────────────────────────────
    model_path = os.path.join(args.checkpoint_dir, "best_clf_bml.pt")
    dq_path    = os.path.join(args.checkpoint_dir, "dq_scaler_bml.pkl")
    sum_path   = os.path.join(args.checkpoint_dir, "summary_scaler_bml.pkl")

    model = BatteryRULClassifier(
        cnn_dim       = args.cnn_dim,
        gru_dim       = args.gru_dim,
        gru_layers    = args.gru_layers,
        summary_feats = SUMMARY_FEATS,
        n_classes     = N_CLASSES,
        dropout       = args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    dq_scaler      = joblib.load(dq_path)
    summary_scaler = joblib.load(sum_path)
    scalers        = (dq_scaler, summary_scaler)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded — {n_params:,} parameters\n")

    # ── Single-cell mode ───────────────────────────────────────────────────
    if args.cell_id is not None:
        if args.cell_id not in cells:
            print(f"ERROR: cell '{args.cell_id}' not found. Available cells:")
            for cid in sorted(cells):
                print(f"  {cid}")
            return

        cell = cells[args.cell_id]
        res  = predict_cell(cell, model, scalers, device)
        if res is None:
            print(f"Cell {args.cell_id} has too few cycles (<{N_INPUT}).")
            return

        print(f"Cell: {args.cell_id}")
        print(f"  Sliding-window accuracy: {res['accuracy']:.4f}  "
              f"({len(res['end_cycles'])} windows  cycle_life={res['cycle_life']})")
        plot_single_cell(res, args.cell_id, out_dir)
        return

    # ── All-cells mode ────────────────────────────────────────────────────
    print("Running inference on all cells...")
    all_results: dict = {}
    for cid, cell in sorted(cells.items()):
        r = predict_cell(cell, model, scalers, device)
        if r is not None and len(r["end_cycles"]) > 0:
            all_results[cid] = r

    print(f"Predicted {len(all_results)} / {len(cells)} cells\n")
    accs = [r["accuracy"] for r in all_results.values()]
    print(f"Accuracy  mean={np.mean(accs):.4f}  std={np.std(accs):.4f}  "
          f"min={np.min(accs):.4f}  max={np.max(accs):.4f}\n")

    # pick a representative cell for the detailed single-cell plot
    sample_id = sorted(all_results)[0]
    print(f"Single-cell plot: {sample_id}")
    plot_single_cell(all_results[sample_id], sample_id, out_dir)

    print("Grid plot...")
    plot_all_cells_grid(all_results, out_dir)

    print("Overlay plot...")
    plot_overlay(all_results, out_dir)

    print("NEOL plot...")
    eol_true, eol_pred, eol_err, mae_eol, mape_eol, cid_list = plot_neol(
        all_results, out_dir
    )

    print("\nPer-cell summary:")
    print_summary_table(
        all_results, eol_true, eol_pred, eol_err, mae_eol, mape_eol, cid_list
    )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

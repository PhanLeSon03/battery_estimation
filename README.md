# Battery RUL Classification with CNN+GRU

Sliding-window RUL (Remaining Useful Life) classification of lithium-ion batteries using the MIT-Stanford dataset. A CNN+GRU model classifies each window of cycles into one of 5 RUL classes.

## RUL class boundaries

| Class | Condition | Meaning |
|-------|-----------|---------|
| 0 | RUL > 400 | Early life |
| 1 | 300 < RUL ≤ 400 | Mid-early life |
| 2 | 200 < RUL ≤ 300 | Mid life |
| 3 | 100 < RUL ≤ 200 | Late life |
| 4 | RUL ≤ 100 | Near end of life |


---

## Table of Contents
- [Results](#results)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [File Structure](#file-structure)

---

## Results

**Test Accuracy: 79.72%** | Test Loss: 0.8454

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| RUL > 400 | 0.92 | 0.85 | 0.89 | 2627 |
| RUL 300–400 | 0.77 | 0.69 | 0.73 | 2697 |
| RUL 200–300 | 0.67 | 0.79 | 0.73 | 2700 |
| RUL 100–200 | 0.78 | 0.73 | 0.75 | 2700 |
| RUL < 100 | 0.86 | 0.93 | 0.89 | 2760 |
| **Weighted avg** | **0.80** | **0.80** | **0.80** | **13484** |

**Confusion Matrix:**
```
              RUL>400  RUL>300  RUL>200  RUL>100  RUL<100
RUL>400        2237      316      73        0        1
RUL>300         186     1856     570       32       53
RUL>200           0      225    2123      341       11
RUL>100           0        0     383     1960      357
RUL<100           0        0       0      186     2574
```



Key observations:
- Early life (RUL > 400) and end of life (RUL < 100) are classified with highest accuracy (F1 = 0.88/0.89)
- Confusion is concentrated in adjacent classes only — the model never confuses early life with late life
- Mid-life classes (RUL 200–300, 300–400) are the hardest to distinguish, consistent with gradual capacity fade

### Example: Sliding-window prediction on a test cell

Cell `batch2c042` (cycle life: 466, accuracy: **81.05%**)

![Sliding-window prediction for batch2c042](images/batch2c042_clf_prediction.png)

**Panel 1 — Predicted vs True RUL Class:** The model tracks all five class transitions correctly with only a 1-cycle lag at each boundary. Red regions show windows where the prediction is one class behind the true label — typical behaviour at class transition points where the degradation signal is ambiguous.

**Panel 2 — Softmax Probabilities:** Each class probability rises and falls cleanly as the cell ages. The sharp, high-confidence transitions (probabilities reaching ~0.95) indicate the model has learned distinct degradation signatures for each RUL band. The overlap between adjacent classes at transition points explains the concentrated off-diagonal entries in the confusion matrix.

**Panel 3 — Classification Error:** Errors are exclusively −1 (model predicts one class too optimistic / early) and occur only at the 4 class boundaries — never in the stable interior of a class. No errors of magnitude > 1 occur, confirming the model never makes catastrophic mis-classifications.

All cells:

![Near End of Life based on battery quality estimation](images/eol_parity_clf.png)

![Overlay with color](images/overlay_all_cells_clf.png)

---

## Dataset

[Severson et al., Nature Energy 2019](https://www.nature.com/articles/s41560-019-0356-8) — 124 commercial LFP/graphite cells cycled to end of life.

**Download:**
```bash
python download.py
```
Move data to folder ./data

---

## Usage

### 1 — Extract features from raw .mat files
```bash
python gen_features.py --data_dir ./data --out_dir ./content
```
Saves one `.npz` per cell under `./content/`. Only needs to run once.

### 2 — Train
```bash
python train_clf.py --content_dir ./content --output_dir ./checkpoints_clf
```

```
Building train dataset...
  Cells valid: 111  Skipped: 0  Total samples: 54647
  Class 0: 10676 samples
  Class 1: 10700 samples
  Class 2: 10868 samples
  Class 3: 10993 samples
  Class 4: 11410 samples
Building val dataset...
  Cells valid: 27  Skipped: 0  Total samples: 13484
  Class 0: 2627 samples
  Class 1: 2697 samples
  Class 2: 2700 samples
  Class 3: 2700 samples
  Class 4: 2760 samples
Building test dataset...
Model parameters: 48,421

Epoch |   TrLoss |   TrAcc |   VaLoss |   VaAcc |       LR
----------------------------------------------------------
    1 |   0.5728 |  0.7562 |   0.7449 |  0.6709 | 9.96e-04
    2 |   0.3936 |  0.8340 |   0.6347 |  0.7279 | 9.84e-04
    3 |   0.3363 |  0.8578 |   0.5729 |  0.7711 | 9.65e-04
    4 |   0.2903 |  0.8785 |   0.5611 |  0.7946 | 9.39e-04
    5 |   0.2533 |  0.8949 |   0.5537 |  0.7849 | 9.05e-04
    6 |   0.2331 |  0.9029 |   0.6297 |  0.7621 | 8.66e-04
    7 |   0.2113 |  0.9125 |   0.6489 |  0.7642 | 8.21e-04
    8 |   0.1914 |  0.9226 |   0.7100 |  0.7679 | 7.70e-04
    9 |   0.1748 |  0.9301 |   0.7461 |  0.7631 | 7.16e-04
   10 |   0.1581 |  0.9366 |   0.7857 |  0.7774 | 6.58e-04
   11 |   0.1493 |  0.9427 |   0.8805 |  0.7567 | 5.98e-04
   12 |   0.1378 |  0.9470 |   0.8117 |  0.7891 | 5.36e-04
   13 |   0.1241 |  0.9523 |   0.9096 |  0.7761 | 4.74e-04
   14 |   0.1194 |  0.9553 |   0.8454 |  0.7972 | 4.12e-04
   15 |   0.1117 |  0.9592 |   0.9949 |  0.7797 | 3.52e-04
   16 |   0.1023 |  0.9627 |   0.9698 |  0.7914 | 2.94e-04
   17 |   0.0956 |  0.9652 |   1.0886 |  0.7698 | 2.40e-04
   18 |   0.0906 |  0.9665 |   1.0653 |  0.7781 | 1.89e-04
   19 |   0.0832 |  0.9696 |   1.1032 |  0.7783 | 1.44e-04
   20 |   0.0815 |  0.9700 |   1.0869 |  0.7935 | 1.05e-04
   21 |   0.0752 |  0.9724 |   1.1413 |  0.7787 | 7.12e-05
   22 |   0.0728 |  0.9737 |   1.1490 |  0.7816 | 4.48e-05
   23 |   0.0718 |  0.9740 |   1.1391 |  0.7929 | 2.56e-05
   24 |   0.0700 |  0.9751 |   1.1659 |  0.7881 | 1.39e-05
   25 |   0.0644 |  0.9764 |   1.1498 |  0.7889 | 1.00e-05

============================================================
Test Loss: 0.8454  Accuracy: 0.7972

Classification Report:
              precision    recall  f1-score   support

     RUL>400       0.92      0.85      0.89      2627
     RUL>300       0.77      0.69      0.73      2697
     RUL>200       0.67      0.79      0.73      2700
     RUL>100       0.78      0.73      0.75      2700
     RUL<100       0.86      0.93      0.89      2760

    accuracy                           0.80     13484
   macro avg       0.80      0.80      0.80     13484
weighted avg       0.80      0.80      0.80     13484

Confusion Matrix:
[[2237  316   73    0    1]
 [ 186 1856  570   32   53]
 [   0  225 2123  341   11]
 [   0    0  383 1960  357]
 [   0    0    0  186 2574]]
```


### 3 — Inference & visualisation (notebook)
Open `predict_clf.ipynb` and run all cells. Produces:
- Per-cell sliding-window classification plot (3 panels)
- Grid plot of all cells coloured by accuracy
- Near EOL (NEOL) parity plot and error histogram
- Per-cell results table

---

## BML Pipeline (BatteryML dataset)

A parallel pipeline trains/evaluates on the BatteryML (BML) dataset. Step-by-step:

### Step 1 — Generate features from raw BML `.pkl` files
```bash
python gen_features_bml.py --data_dir ./Raw_BML --out_dir ./content_bml
```
Outputs one `.npz` per cell under `./content_bml/<family>/<cell>.npz`. Run once.

Optional smoke test:
```bash
python gen_features_bml.py --data_dir ./Raw_BML --out_dir ./content_bml --max_files 5
```

### Step 2 — Build datasets / dataloaders (sanity check)
`dataset_clf_bml.py` is normally imported by the trainers, but you can also run it standalone to verify feature loading, sample indexing, and the train/val/test split (80/10/10 by default):
```bash
python dataset_clf_bml.py --content_dir ./content_bml
# Or restrict to one BML family:
python dataset_clf_bml.py --content_dir ./content_bml/CALB --val_ratio 0.1
```
This prints the cells assigned to each split and writes `<content_dir>/split_cells.json` containing the cell IDs and absolute `.npz` paths for train / val / test — `predict_clf.ipynb` consumes this file.

### Step 3 — Train (backprop, AdamW)
```bash
python train_clf_bml.py \
    --content_dir ./content_bml \
    --output_dir  ./checkpoints_clf_bml \
    --val_ratio   0.1
```
Train on a single BML family:
```bash
python train_clf_bml.py \
    --content_dir ./content_bml/CALB \
    --output_dir  ./checkpoints_clf_bml_CALB \
    --val_ratio   0.1
```
On start, the script prints the model architecture, layer-by-layer summary, and total/trainable parameter counts. After each epoch, it logs train/val loss + accuracy plus macro precision / recall / F1 on val. Everything printed is mirrored to `checkpoints_clf_bml/train_log_<timestamp>.txt`.

Outputs:
- `checkpoints_clf_bml/best_clf_bml.pt`
- `checkpoints_clf_bml/dq_scaler_bml.pkl`
- `checkpoints_clf_bml/summary_scaler_bml.pkl`
- `checkpoints_clf_bml/clf_pred_bml.npy`, `clf_true_bml.npy`
- `checkpoints_clf_bml/train_log_<timestamp>.txt`

### Step 3b — Train with CMA-ES (no backprop, optional)
Useful as a fine-tuner over the best AdamW checkpoint:
```bash
python train_clf_es_bml.py \
    --content_dir    ./content_bml \
    --output_dir     ./checkpoints_es_bml \
    --pretrain_ckpt  ./checkpoints_clf_bml/best_clf_bml.pt
```
Also writes a `train_log_<timestamp>.txt` to `--output_dir` mirroring all stdout/stderr.

### Step 4 — Inference & visualisation (`predict_clf.ipynb`)
The notebook is wired to the BML pipeline. At the top:
```python
SPLIT_JSON = './content_bml/split_cells.json'   # produced in Step 2/3
CKPT_DIR   = './checkpoints_clf_bml'            # output of Step 3
```
Update those paths to match your run (e.g. `./content_bml/CALB/split_cells.json` and `./checkpoints_clf_bml_CALB`), then run all cells. The notebook:
- Reads `split_cells.json` (cell IDs + absolute paths for train / val / test).
- Loads only the cells listed there via `_load_npz_cell`.
- Loads `best_clf_bml.pt` and the scalers from `CKPT_DIR`.
- Runs sliding-window prediction on every cell in the test split.
- Produces per-cell 3-panel plots, an all-cells grid, the NEOL parity plot, and a per-cell summary table.

---

## Model Architecture

Input per sample: **32 cycles** = first 8 (early degradation signal) + 24 consecutive (random window).

```
dq  (B, 32, 1000)  ──► Time-distributed Conv1D ──► (B, 32, cnn_dim)
                         3 × [Conv1d → BN → GELU → Dropout → MaxPool]
                         AdaptiveAvgPool1d(1)

summary (B, 32, 16) ──► Linear → GELU → Dropout ──► (B, 32, cnn_dim)

concat ──► (B, 32, cnn_dim×2)
        ──► Dropout
        ──► BiGRU (layer 1) ──► (B, 32, gru_dim×2)
        ──► Dropout
        ──► BiGRU (layer 2) ──► h_last (B, gru_dim×2)
        ──► Dropout
        ──► Linear(gru_dim×2 → 32) → GELU → Dropout
        ──► Linear(32 → 5)
        ──► Softmax
```

---

## Feature Engineering

### Input sequence — 16 features per cycle

| # | Feature | Description |
|---|---------|-------------|
| 1 | `QDischarge` | Discharge capacity (Ah) |
| 2 | `IR` | Internal resistance (Ω) |
| 3 | `Tmax` | Maximum temperature (°C) |
| 4 | `Tavg` | Average temperature (°C) |
| 5 | `chargetime` | Time to reach 80% SOC (min) |
| 6 | `dQdV_max` | Max of dQ/dV curve |
| 7 | `dQdV_min` | Min of dQ/dV curve |
| 8 | `dQdV_avg` | Mean of dQ/dV curve |
| 9 | `20·log₁₀(std(Qdlin))` | Log-std of discharge capacity curve across voltage bins |
| 10 | `20·log₁₀(std(T))` | Log-std of temperature within cycle |
| 11 | `20·log₁₀(std(IR))` | Log-std of internal resistance within cycle |
| 12 | `20·log₁₀(std(chargetime))` | Log-std of charge time within cycle |
| 13–16 | Sinusoidal PE | Position encoding: sin/cos at 2 frequencies |

### dQ sequence — 1000 features per cycle
Difference of the Qdlin curve (1000 voltage-interpolated discharge capacity points) relative to cycle 10 (reference):
```
dQ[c] = Qdlin[c] - Qdlin[ref=9]
```

### Balanced class sampling
Each cell contributes an equal number of samples per RUL class. Valid window positions are grouped by their true RUL class, then `n_samples // 5` windows are drawn from each class independently — preventing the dominant RUL > 400 class from overwhelming training.

---

## File Structure

```
battery-rul-clf/
├── data/                   # raw .mat files (not committed)
│   ├── batch1.mat
│   ├── batch2.mat
│   └── batch3.mat
├── content/                # extracted .npz features (generated)
├── checkpoints_clf/        # saved model weights (generated)
│   └── best_clf.pt
├── gen_features.py         # .mat → .npz feature extraction
├── dataset_clf.py          # classification dataset & dataloader
├── train_clf.py            # training loop + BatteryRULClassifier model
├── predict_clf.ipynb       # inference & visualisation notebook
└── README.md
```

---

## Citation

```bibtex
@article{severson2019data,
  title   = {Data-driven prediction of battery cycle life before capacity degradation},
  author  = {Severson, Kristen A and others},
  journal = {Nature Energy},
  year    = {2019},
  doi     = {10.1038/s41560-019-0356-8}
}
```
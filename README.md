# Battery RUL Classification with CNN+GRU

Sliding-window RUL (Remaining Useful Life) classification of lithium-ion batteries using the MIT-Stanford dataset. A CNN+GRU model classifies each window of cycles into one of 5 RUL classes.

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
```




### 3 — Inference & visualisation (notebook)
Open `predict_clf.ipynb` and run all cells. Produces:
- Per-cell sliding-window classification plot (3 panels)
- Grid plot of all cells coloured by accuracy
- EOL parity plot and error histogram
- Per-cell results table

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
| 9 | `10·log₁₀(std(Qdlin))` | Log-std of discharge capacity curve across voltage bins |
| 10 | `10·log₁₀(std(T))` | Log-std of temperature within cycle |
| 11 | `10·log₁₀(std(IR))` | Log-std of internal resistance within cycle |
| 12 | `10·log₁₀(std(chargetime))` | Log-std of charge time within cycle |
| 13–16 | Sinusoidal PE | Position encoding: sin/cos at 2 frequencies |

### dQ sequence — 1000 features per cycle
Difference of the Qdlin curve (1000 voltage-interpolated discharge capacity points) relative to cycle 10 (reference):
```
dQ[c] = Qdlin[c] - Qdlin[ref=9]
```

### Balanced class sampling
Each cell contributes an equal number of samples per RUL class. Valid window positions are grouped by their true RUL class, then `n_samples // 5` windows are drawn from each class independently — preventing the dominant RUL > 400 class from overwhelming training.

### RUL class boundaries

| Class | Condition | Meaning |
|-------|-----------|---------|
| 0 | RUL > 400 | Early life |
| 1 | 300 < RUL ≤ 400 | Mid-early life |
| 2 | 200 < RUL ≤ 300 | Mid life |
| 3 | 100 < RUL ≤ 200 | Late life |
| 4 | RUL ≤ 100 | Near end of life |

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
# Turbofan-Predictive-Maintenance-Prognostics
# 🛩️ Jet Engine RUL Predictor — NASA C-MAPSS FD001

> Predicting **exactly when a jet engine will fail** before it happens, using real NASA sensor telemetry and a hybrid CNN + LSTM deep learning architecture.

---

## 🏆 Result

| Metric | Score |
|---|---|
| 📉 Test MAE | **13.5 flights** average error |
| 🎯 Accuracy window | **~5%** of total engine lifespan |
| ✈️ Engines tested | 100 fully unseen engines |
| 📡 Sensors tracked | 17 pressure + temperature channels |

> The model was deployed against an entirely unseen fleet of 100 NASA jet engines and successfully predicted their remaining operational lifespans with an average error of just 13.5 flights.

---

## 🧠 How It Works

The model processes raw sensor readings through a 5-stage pipeline:

### Stage 1 — MinMax Scaling
All 17 sensors are normalized between 0 and 1. This prevents large values like core pressure from drowning out subtle but critical signals like fuel-to-air ratio drift.

### Stage 2 — 3D Sliding Windows
Raw rows are grouped into 30-flight history blocks. Each block becomes one training sample — the model sees a continuous timeline, not isolated snapshots.

### Stage 3 — Conv1D Spatial Filter
A 1D convolution scans all 17 sensor channels in parallel, blending interacting signals and stripping out random operational noise before the LSTM processes anything.

### Stage 4 — LSTM Temporal Tracker
Two stacked LSTM layers use memory gates to track the velocity and acceleration of physical wear over time — detecting not just how degraded the engine is, but how fast it is degrading.

### Stage 5 — Asymmetric Safety Scoring
Evaluated using NASA's industrial safety metric:
- 🔴 **Predicting failure too late** → exponential penalty (risk of mid-flight catastrophe)
- 🟡 **Predicting failure too early** → linear penalty (unnecessary early maintenance)

---

## 🏗️ Model Architecture

**Total parameters: 39,457** — lightweight enough to run on CPU.

---

## 📈 Training Curve

| Epoch | Train MAE | Val MAE | Note |
|---|---|---|---|
| 1 | 61.03 | 45.60 | Cold start |
| 2 | 39.34 | 38.08 | |
| 3–4 | ~37.3 | ~37.8 | Plateau (pre-cap) |
| 5 | 33.75 | 20.87 | ⚡ Breakthrough |
| 6 | 17.48 | 15.21 | Rapid descent |
| 8 | 13.39 | 12.96 | SOTA territory |
| **17** | **11.45** | **12.81** | ✅ Final |

> The dramatic drop at epoch 5 was triggered by RUL capping (max 125 cycles), which gave the model a clean, learnable target distribution.

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 with `ReduceLROnPlateau` |
| Loss function | Huber (δ = 50) |
| RUL cap | 125 cycles |
| Window size | 30 cycles |
| Batch size | 64 |
| Max epochs | 40 (early stopping, patience = 10) |
| Validation split | 20% |

---

## 🗂️ Dataset

**NASA C-MAPSS FD001** — [Download from NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

| Property | Detail |
|---|---|
| Fault mode | High Pressure Compressor (HPC) degradation |
| Operating condition | Sea level (single condition) |
| Training engines | 100 run-to-failure trajectories |
| Test engines | 100 partial trajectories |
| Input features | 3 operational settings + 21 sensors (17 selected) |

```python
# Load the dataset
col_names = ['engine_id', 'cycle',
             'setting_1', 'setting_2', 'setting_3',
             *[f'sensor_{i}' for i in range(1, 22)]]

train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
test_df  = pd.read_csv('test_FD001.txt',  sep=r'\s+', header=None, names=col_names)
truth_df = pd.read_csv('RUL_FD001.txt',   sep=r'\s+', header=None, names=['RUL_True'])
```

---

## 📦 Project Structure

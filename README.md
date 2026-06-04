# MAFN — Modality-Adaptive Fusion Network

Lightweight fusion of **mmWave radar** (point cloud) and a **wearable IMU** for
in-home **human activity recognition (HAR, 12 classes)** and **fall detection
(FD, binary)**, with robustness to a **missing modality** at inference. A single
checkpoint operates under full-modality, radar-only and IMU-only inputs — the
available modality is selected at test time through a modality mask, without
retraining.

This repository accompanies the paper and provides the model definition and an
evaluation script on the OctoNet test set. Training code and baselines are not
included.

## Highlights

- **LTCE** (Lightweight Temporal Context Encoder): a per-modality temporal
  encoder on a dilated depthwise-separable backbone. The radar front-end is
  task-adaptive — point cloud **+** per-frame micro-Doppler velocity spectrum for
  HAR, point cloud only for FD.
- **BMAF** (Bidirectional Modality-Adaptive Fusion): bidirectional
  cross-attention, mean-max pooling, a learned `[MASK]` vector for an absent
  modality, and a presence-aware adaptive gate.
- Compact: the lightweight **MAFN-S** uses about **390K** parameters.

## Repository layout

```
MAFN/
├── mafn/
│   ├── model.py      # MAFN: LTCE encoders + BMAF fusion + MLP head
│   ├── ltce.py       # depthwise-separable temporal encoders (radar / IMU front-ends)
│   ├── bmaf.py       # bidirectional cross-attention, pooling, mask token, gate
│   ├── data.py       # OctoNet test-set loader
│   └── metrics.py    # accuracy / F1 / sensitivity / specificity / AUROC / Youden
├── test.py           # evaluate a checkpoint under full / radar-only / IMU-only
├── scripts/
│   └── prepare_octonet.py   # build the test set from the public OctoNet release
├── checkpoints/      # trained weights (MAFN d=192 and MAFN-S d=128; HAR and FD)
├── data/
│   └── octonet_index.csv    # per-recording index used by the prep script
├── requirements.txt
└── LICENSE
```

## Installation

Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

## Prepare the test data

The OctoNet arrays are not redistributed here. Download the public **OctoNet**
release, then build the test split with the provided script and the tracked
record index (`data/octonet_index.csv`):

```bash
python scripts/prepare_octonet.py \
    --octonet-root /path/to/OctoNet \
    --index data/octonet_index.csv \
    --out data/octonet
```

This segments each repetition's node-1 radar point cloud and wrist IMU, computes
the per-frame micro-Doppler spectrum, and writes `data/octonet/` (which `test.py`
reads by default). Processing is deterministic, so the output matches the data
the checkpoints were evaluated on. Please cite OctoNet and follow its terms of use.

## Usage

```bash
python test.py --task har                  # MAFN (d=192), 12-class HAR
python test.py --task fall                 # MAFN (d=192), fall detection
python test.py --task har  --variant mafn-s   # MAFN-S (d=128)
python test.py --task fall --variant mafn-s
```

Each run loads one checkpoint and evaluates it under three inference regimes —
**full** (radar + IMU), **radar-only**, and **IMU-only** — selected purely by the
modality mask. HAR reports accuracy / macro-F1 / weighted-F1; FD reports recall,
specificity, precision, F1, AUROC, false-alarm rate (FAR) and missed-detection
rate (MDR), with the operating point set by Youden's J (AUROC is threshold-free).

## Checkpoints and expected output

The checkpoints are representative single-seed runs (trained with RAdam, modality
dropout, and Stochastic Weight Averaging for 60 epochs); the paper reports
mean ± standard deviation over multiple seeds, so individual numbers differ
slightly from the reported means. Running `test.py` on the provided checkpoints
reproduces the following on the OctoNet test set.

| Variant | Task | `d_model` | Params | File |
|---|---|---|---|---|
| MAFN    | HAR  | 192 | 821K | `checkpoints/mafn_har.pth` |
| MAFN    | FD   | 192 | 777K | `checkpoints/mafn_fall.pth` |
| MAFN-S  | HAR  | 128 | 390K | `checkpoints/mafn_s_har.pth` |
| MAFN-S  | FD   | 128 | 349K | `checkpoints/mafn_s_fall.pth` |

**HAR — macro-F1**

| Variant | full | radar-only | IMU-only |
|---|---|---|---|
| MAFN   | 0.521 | 0.389 | 0.341 |
| MAFN-S | 0.534 | 0.344 | 0.340 |

**FD — AUROC (full / radar-only / IMU-only) and full-modality MDR**

| Variant | AUROC full | AUROC radar | AUROC IMU | MDR (full) |
|---|---|---|---|---|
| MAFN   | 0.998 | 0.986 | 0.803 | 0.000 |
| MAFN-S | 0.990 | 0.852 | 0.894 | 0.000 |

Both tasks are trained separately; each checkpoint carries a single task head.

## Model

`mafn/model.py` is the entry point. Each modality is encoded by its own LTCE into
a shared temporal embedding space; BMAF fuses the two streams bidirectionally and
weighs them with a presence-aware gate; a lightweight MLP head produces the
prediction. When a modality is absent, its encoder branch is skipped and the
fusion module receives a learned `[MASK]` vector in its place.

## License

Released under the MIT License (see `LICENSE`). The OctoNet dataset is covered by
its own license; please refer to the original release.

# Author: Boya Zhang <by.zhang1@siat.ac.cn>
# Date: 2026.05.31
"""Build the OctoNet HAR / fall-detection subset used by MAFN.

This reproduces the cached arrays consumed by ``mafn.data`` from the public
OctoNet release. For each activity repetition it segments the node-1 mmWave
point cloud and the wrist IMU by their wall-clock timestamps, resamples both to a
common window length ``T``, and writes:

    <out>/processed/<sample_id>/radar_pc.npy : (T, P, 4)  point cloud [x, y, z, velocity]
    <out>/processed/<sample_id>/radar_md.npy : (T, 64)    per-frame micro-Doppler spectrum
    <out>/processed/<sample_id>/imu.npy      : (T, 13)    wrist IMU channels
    <out>/manifest.csv                       : one row per repetition

Usage
-----
    python scripts/prepare_octonet.py \
        --octonet-root /path/to/OctoNet \
        --index data/octonet_index.csv \
        --out data/octonet

``--octonet-root`` is the directory of the downloaded OctoNet release; the index
file lists, per recording, the relative mmWave / IMU paths, the recording start
time, the per-repetition cut timestamps, and the activity labels. Processing all
recordings in order with a fixed seed makes the output deterministic.
"""
from __future__ import annotations

import argparse
import ast
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HAR_CLASSES = ['sit', 'walk', 'sleep', 'bow', 'pickup', 'squat', 'lunge',
               'legraise', 'stretchoneself', 'turn', 'jog', 'stagger']

# Subject-independent split (by subject id).
VAL_SUBJECTS = {9, 10}
TEST_SUBJECTS = {7, 8}


def parse_start(s: str) -> datetime:
    return datetime.strptime(str(s).strip(), '%Y-%m-%d %H:%M:%S.%f')


def parse_cut(s: str) -> datetime:
    return datetime.strptime(s.strip(), '%Y-%m-%d %H.%M.%S.%f')


def split_of(subject: int) -> str:
    if subject in TEST_SUBJECTS:
        return 'TEST'
    if subject in VAL_SUBJECTS:
        return 'VAL'
    return 'TRAIN'


def load_mmwave(path: Path, max_coord: float = 30.0):
    """Return (timestamps, points) for an mmWave stream.

    A small fraction of raw detections carry finite but astronomically large
    coordinates/velocity (near the float32 limit); they pass an ``isfinite``
    check yet overflow downstream, so any point whose |x|, |y|, |z| or
    |velocity| exceeds a physical bound is dropped.
    """
    ts, pts = [], []
    try:
        with open(path, 'rb') as f:
            while True:
                fr = pickle.load(f)
                d = fr['data']
                ts.append(pd.Timestamp(fr['timestamp']).to_pydatetime())
                p = np.stack([d['x'], d['y'], d['z'], d['velocity']], axis=1).astype(np.float32)
                keep = np.isfinite(p).all(axis=1) & (np.abs(p) <= max_coord).all(axis=1)
                pts.append(p[keep])
    except EOFError:
        pass
    return np.array(ts), pts


def resample_idx(n_src: int, n_dst: int) -> np.ndarray:
    if n_src <= 0:
        return np.zeros(n_dst, dtype=int)
    return np.clip(np.round(np.linspace(0, n_src - 1, n_dst)).astype(int), 0, n_src - 1)


def fix_points(frame: np.ndarray, P: int, rng: np.random.Generator) -> np.ndarray:
    """Pad-or-sample a (n, 4) point set to exactly (P, 4)."""
    n = frame.shape[0]
    if n == 0:
        return np.zeros((P, 4), dtype=np.float32)
    if n >= P:
        return frame[rng.choice(n, P, replace=False)]
    return np.concatenate([frame, np.zeros((P - n, 4), dtype=np.float32)], axis=0)


def micro_doppler(points_seq, num_freq_bins: int = 64, doppler_max: float = 4.0) -> np.ndarray:
    """Per-frame Doppler histogram -> (T, num_freq_bins) micro-Doppler spectrum."""
    T = len(points_seq)
    md = np.zeros((T, num_freq_bins), dtype=np.float32)
    for t, pts in enumerate(points_seq):
        if pts is None or pts.size == 0:
            continue
        d = pts[:, 3]
        w = pts[:, 4] if pts.shape[1] >= 5 else np.ones(pts.shape[0], dtype=np.float32)
        idx = np.clip(((d + doppler_max) / (2 * doppler_max) * num_freq_bins).astype(int),
                      0, num_freq_bins - 1)
        np.add.at(md[t], idx, w.astype(np.float32))
    return md


def main():
    ap = argparse.ArgumentParser(description='Build the OctoNet subset for MAFN.')
    ap.add_argument('--octonet-root', required=True,
                    help='Root of the downloaded OctoNet release (contains the '
                         'node_1/ and imu/ recordings referenced by the index).')
    ap.add_argument('--index', default='data/octonet_index.csv',
                    help='Record index CSV shipped with this repository.')
    ap.add_argument('--out', default='data/octonet', help='Output directory.')
    ap.add_argument('--T', type=int, default=64, help='Common window length.')
    ap.add_argument('--P', type=int, default=128, help='Points per mmWave frame.')
    ap.add_argument('--sensor', type=int, default=6, help='IMU sensor index (wrist).')
    ap.add_argument('--min-mmwave', type=int, default=4,
                    help='Minimum mmWave frames for radar to count as present.')
    ap.add_argument('--max-coord', type=float, default=30.0, help='Glitch-filter bound.')
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    raw = Path(args.octonet_root)
    out = Path(args.out)
    proc = out / 'processed'
    if proc.exists():
        shutil.rmtree(proc)
    proc.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index)
    rng = np.random.default_rng(args.seed)
    rows = []
    for ri, rec in df.iterrows():
        subj, act = int(rec['subject']), rec['activity']
        split = split_of(subj)

        mm_ts, mm_pts = load_mmwave(raw / rec['mmwave_path'], max_coord=args.max_coord)
        with open(raw / rec['imu_path'], 'rb') as f:
            imu = pickle.load(f)
        imu_ts = np.array([pd.Timestamp(t).to_pydatetime() for t in imu['timestamps']])
        imu_data = np.asarray(imu['data'], dtype=np.float32)   # (N, 13, n_sensors)

        cuts = [parse_cut(c) for c in ast.literal_eval(rec['cut_timestamps'])]
        bounds = [parse_start(rec['start_time'])] + cuts
        for j in range(len(cuts)):
            t0, t1 = bounds[j], bounds[j + 1]
            if t1 <= t0:
                continue
            im_mask = (imu_ts >= t0) & (imu_ts < t1)
            n_imu = int(im_mask.sum())
            if n_imu < 2:
                continue
            imu_win = imu_data[im_mask][:, :, args.sensor][resample_idx(n_imu, args.T)]  # (T, 13)

            mm_in = (mm_ts >= t0) & (mm_ts < t1) if len(mm_ts) else np.zeros(0, bool)
            mm_sel = np.where(mm_in)[0]
            n_mm = int(len(mm_sel))
            radar_present = int(n_mm >= args.min_mmwave)
            if radar_present:
                frames = [mm_pts[k] for k in mm_sel[resample_idx(n_mm, args.T)]]
                pc = np.stack([fix_points(f, args.P, rng) for f in frames], axis=0)
            else:
                frames = [np.zeros((0, 4), np.float32)] * args.T
                pc = np.zeros((args.T, args.P, 4), dtype=np.float32)
            md = micro_doppler(frames, num_freq_bins=64)

            sample_id = f's{subj}_{act}_r{ri}_rep{j}'
            sdir = proc / sample_id
            sdir.mkdir(parents=True, exist_ok=True)
            np.save(sdir / 'radar_pc.npy', pc.astype(np.float32))
            np.save(sdir / 'radar_md.npy', md.astype(np.float32))
            np.save(sdir / 'imu.npy', imu_win.astype(np.float32))
            rows.append(dict(
                sample_id=f'processed/{sample_id}', split=split, subject_id=subj,
                activity=act, har_label=int(rec['har_label']), fall_label=int(rec['is_fall']),
                radar_present=radar_present, n_mmwave_frames=n_mm, n_imu_samples=n_imu,
            ))
        if (ri + 1) % 25 == 0 or ri + 1 == len(df):
            print(f'  [{ri + 1}/{len(df)}] recordings processed, {len(rows)} samples')

    pd.DataFrame(rows).to_csv(out / 'manifest.csv', index=False)
    print(f'\nWrote {out / "manifest.csv"} ({len(rows)} samples).')


if __name__ == '__main__':
    main()

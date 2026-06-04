"""OctoNet test-set loader (mmWave point cloud + micro-Doppler + IMU).

Reads the cached subset under ``data/octonet`` and returns the dict batch the
model consumes. One sample is one activity repetition.

Per-sample arrays (``data/octonet/<sample_id>/``):
    radar_pc.npy : (T, P, 4)   point cloud [x, y, z, velocity]   (kept raw)
    radar_md.npy : (T, F)      per-frame micro-Doppler spectrum  (instance-norm)
    imu.npy      : (T, C)      wearable IMU channels              (instance-norm)

Two tasks, selected by ``task``:
    'har'  : 12-class HAR over the structured activities (rows with har_label >= 0).
    'fall' : binary fall detection (falldown = positive; the 12 activities = negatives).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HAR_CLASSES = ['sit', 'walk', 'sleep', 'bow', 'pickup', 'squat', 'lunge',
               'legraise', 'stretchoneself', 'turn', 'jog', 'stagger']
FALL_CLASSES = ['non_fall', 'falldown']


def _instance_norm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6)


class OctoNetTestSet(Dataset):
    """OctoNet TEST split for one task. Exposes ``num_classes`` / ``class_names``
    and the data dimensions (``imu_channels``, ``pc_in_dim``) inferred from a sample."""

    def __init__(self, root: str, task: str = 'har', split: str = 'TEST'):
        self.root = Path(root)
        self.task = task.lower()
        manifest = self.root / 'manifest.csv'
        if not manifest.exists():
            raise FileNotFoundError(f'manifest not found at {manifest}')
        df = pd.read_csv(manifest)
        df = df[df['split'].str.upper() == split.upper()]

        if self.task == 'har':
            df = df[df['har_label'] >= 0]
            self.label_col, self.num_classes, self.class_names = 'har_label', 12, HAR_CLASSES
        elif self.task == 'fall':
            self.label_col, self.num_classes, self.class_names = 'fall_label', 2, FALL_CLASSES
        else:
            raise ValueError("task must be 'har' or 'fall'")

        self.manifest = df.reset_index(drop=True)
        if len(self.manifest) == 0:
            raise RuntimeError(f'{split} split for task={self.task} is empty.')

        s0 = self.root / self.manifest.iloc[0]['sample_id']
        pc0 = np.load(s0 / 'radar_pc.npy')
        imu0 = np.load(s0 / 'imu.npy')
        self.seq_len = int(pc0.shape[0])
        self.pc_in_dim = int(pc0.shape[-1])
        self.imu_channels = int(imu0.shape[-1])

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        sdir = self.root / row['sample_id']
        pc = torch.from_numpy(np.load(sdir / 'radar_pc.npy').astype(np.float32))           # raw
        md = _instance_norm(torch.from_numpy(np.load(sdir / 'radar_md.npy').astype(np.float32)))
        imu = _instance_norm(torch.from_numpy(np.load(sdir / 'imu.npy').astype(np.float32)))
        radar_present = float(int(row['radar_present']))
        return {
            'radar': {'point_cloud': pc, 'micro_doppler': md},
            'imu': {'data': imu},
            'modality_mask': torch.tensor([radar_present, 1.0]),
            'label': torch.tensor(int(row[self.label_col]), dtype=torch.long),
        }


def collate(batch: list) -> dict:
    """Stack a list of samples (all sequences share the same length T)."""
    return {
        'radar': {
            'point_cloud': torch.stack([s['radar']['point_cloud'] for s in batch], 0),
            'micro_doppler': torch.stack([s['radar']['micro_doppler'] for s in batch], 0),
        },
        'imu': {'data': torch.stack([s['imu']['data'] for s in batch], 0)},
        'modality_mask': torch.stack([s['modality_mask'] for s in batch], 0),
        'label': torch.stack([s['label'] for s in batch], 0),
    }

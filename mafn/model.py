# Author: Boya Zhang <by.zhang1@siat.ac.cn>
# Date: 2026.05.31
"""MAFN — Modality-Adaptive Fusion Network.

A lightweight network that fuses mmWave radar (point cloud, optionally with the
per-frame micro-Doppler velocity spectrum) and a wearable IMU for in-home human
activity recognition (HAR) and fall detection (FD), with robustness to a missing
modality at inference.

Pipeline (Fig. 1 of the paper):

    radar ─► LTCE ─┐
                   ├─► BMAF ─► MLP head ─► logits
    IMU   ─► LTCE ─┘

* **LTCE** (``mafn.ltce``): per-modality temporal encoder. The radar front-end is
  task-adaptive — point cloud + micro-Doppler for HAR (``radar_front='pcmd'``),
  point cloud only for FD (``radar_front='pc'``).
* **BMAF** (``mafn.bmaf``): bidirectional cross-attention, mean-max pooling, a
  learned ``[MASK]`` vector for an absent modality, and a presence-aware adaptive
  gate.

Trained with modality dropout, a single checkpoint operates under full-modality,
radar-only and IMU-only inputs; the modality present at inference is selected
through ``batch['modality_mask']`` without retraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .bmaf import BiDirectionalCrossAttn, MaskedFeatureMix, MeanMaxPool, ResidualAdaptiveGate
from .ltce import IMUEncoder, PointCloudEncoder, PointMDEncoder

_RADAR_COL, _IMU_COL = 0, 1


class MAFN(nn.Module):
    """Dual-modality (radar + IMU) MAFN.

    Args:
        num_class:     number of output classes (12 for HAR, 2 for FD).
        d_model:       temporal embedding width (192 for MAFN, 128 for MAFN-S).
        radar_front:   ``'pcmd'`` (point cloud + micro-Doppler, HAR) or
                       ``'pc'`` (point cloud only, FD).
        aux:           build auxiliary per-modality heads. These are used only
                       for deep supervision during training; they are inert at
                       inference but kept so a checkpoint trained with them loads.
        imu_channels:  number of IMU channels.
        pc_in_dim:     point-cloud feature dimension (x, y, z, velocity = 4).
        n_heads:       cross-attention heads.
        dropout:       dropout rate (inactive in eval mode).
        md_bins:       micro-Doppler velocity bins.
    """

    def __init__(self, num_class: int, d_model: int = 192, radar_front: str = 'pc',
                 aux: bool = False, imu_channels: int = 13, pc_in_dim: int = 4,
                 n_heads: int = 4, dropout: float = 0.1, md_bins: int = 64):
        super().__init__()
        self.d_model = d_model
        self.radar_front = radar_front

        # --- per-modality encoders (LTCE) ---
        if radar_front == 'pcmd':
            self.radar_enc = PointMDEncoder(pc_in_dim, d_model, dropout, md_bins)
        else:
            self.radar_enc = PointCloudEncoder(pc_in_dim, d_model, dropout)
        self.imu_enc = IMUEncoder(imu_channels, d_model, dropout)

        # --- fusion (BMAF) ---
        self.pool = MeanMaxPool()
        self.cross = BiDirectionalCrossAttn(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.masked_mix = MaskedFeatureMix(d_model=2 * d_model, n_slots=2)
        self.gate = ResidualAdaptiveGate(d_model=2 * d_model, n_slots=2)

        # --- classification head ---
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 2), nn.Linear(d_model, num_class))

        # Auxiliary per-modality heads (training-only deep supervision).
        self.aux = aux
        if aux:
            self.radar_aux = nn.Linear(2 * d_model, num_class)
            self.imu_aux = nn.Linear(2 * d_model, num_class)

    # ------------------------------------------------------------------ #
    def _encode_radar(self, batch: dict, mask: torch.Tensor) -> torch.Tensor:
        B, dev = mask.shape[0], mask.device
        if self.radar_front == 'pcmd':
            pc = batch['radar'].get('point_cloud')
            md = batch['radar'].get('micro_doppler')
            if pc is not None and md is not None and mask[:, _RADAR_COL].sum() > 0:
                return self.radar_enc(pc, md).transpose(1, 2)
            return torch.zeros(B, 1, self.d_model, device=dev)
        pc = batch['radar'].get('point_cloud')
        if pc is not None and mask[:, _RADAR_COL].sum() > 0:
            return self.radar_enc(pc).transpose(1, 2)
        return torch.zeros(B, 1, self.d_model, device=dev)

    def _encode_imu(self, batch: dict, mask: torch.Tensor) -> torch.Tensor:
        B, dev = mask.shape[0], mask.device
        x = batch['imu'].get('data')
        if x is not None and mask[:, _IMU_COL].sum() > 0:
            return self.imu_enc(x).transpose(1, 2)
        return torch.zeros(B, 1, self.d_model, device=dev)

    def forward(self, batch: dict) -> torch.Tensor:
        """batch: dict with radar.point_cloud, radar.micro_doppler, imu.data,
        and modality_mask (B, 2) of 0/1 over (radar, imu)."""
        mm = batch['modality_mask'].float()
        B = mm.shape[0]

        z_radar = self._encode_radar(batch, mm)   # (B, T, d)
        z_imu = self._encode_imu(batch, mm)        # (B, T, d)
        radar_kpm = (mm[:, _RADAR_COL:_RADAR_COL + 1] < 0.5).expand(B, z_radar.shape[1])
        imu_kpm = (mm[:, _IMU_COL:_IMU_COL + 1] < 0.5).expand(B, z_imu.shape[1])

        r_enh, i_enh = self.cross(z_radar, z_imu, a_kpm=radar_kpm, b_kpm=imu_kpm)
        radar_p = self.pool(r_enh.transpose(1, 2))   # (B, 2d)
        imu_p = self.pool(i_enh.transpose(1, 2))     # (B, 2d)

        feats = torch.stack([radar_p, imu_p], dim=1)  # (B, 2, 2d)
        feats = self.masked_mix(feats, mm)
        fused = self.gate(feats, mm)                  # (B, 2d)
        return self.head(fused)

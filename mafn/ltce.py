# Author: Boya Zhang <by.zhang1@siat.ac.cn>
# Date: 2026.05.31
"""Lightweight Temporal Context Encoder (LTCE).

Each modality is encoded by a modality-specific front-end followed by a stack of
dilated depthwise-separable temporal blocks that expand the temporal receptive
field at a low parameter budget:

* ``DSCBlock``         : depthwise + pointwise Conv1d with BatchNorm, ReLU,
                         dropout and a residual shortcut. Dilation grows
                         geometrically across the stack (1, 2, 4, ...).
* ``PointCloudEncoder``: per-frame PointNet-style set encoder over the radar
                         point cloud (used by fall detection).
* ``PointMDEncoder``   : cross-representation radar front-end that fuses the
                         point-cloud set encoding with the per-frame
                         micro-Doppler velocity spectrum (used by HAR).
* ``IMUEncoder``       : identity front-end over the (T, C) IMU stream.

All encoders return ``(B, d_model, T)`` — a temporal feature map ready for
mean-max pooling and cross-modal fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DSCBlock(nn.Module):
    """Depthwise-separable Conv1d with BatchNorm + ReLU + Dropout + residual.

    Supports dilation for a large temporal receptive field (TCN-style) and
    stride for optional temporal downsampling.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.downsample: nn.Module
        if stride > 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = self.downsample(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out + residual


class PointCloudEncoder(nn.Module):
    """Per-frame PointNet-style set encoder followed by a dilated DSC stack.

    Input  : (B, T, P, in_dim)
    Output : (B, d_model, T)
    """

    def __init__(self, in_dim: int = 4, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128),
        )
        self.tcn = nn.Sequential(
            DSCBlock(128, d_model, kernel_size=3, dilation=1, dropout=dropout),
            DSCBlock(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
            DSCBlock(d_model, d_model, kernel_size=3, dilation=4, dropout=dropout),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: (B, T, P, in_dim) -> per-frame max-pool over points -> (B, 128, T)
        x = self.point_mlp(pc)
        x = x.max(dim=2).values
        x = x.transpose(1, 2)
        return self.tcn(x)


class PointMDEncoder(nn.Module):
    """Cross-representation radar front-end fusing point cloud + micro-Doppler.

    The sparse point cloud (max-pooled over points) loses the per-frame velocity
    distribution that discriminates many fine-grained activities; the
    micro-Doppler spectrum restores it. Both are projected to a per-frame channel
    vector, concatenated and fused, then passed through the same dilated DSC stack
    as ``PointCloudEncoder``. Only the front-end changes.

    Input  : point_cloud (B, T, P, in_dim) + micro_doppler (B, T, F)
    Output : (B, d_model, T)
    """

    def __init__(self, in_dim: int = 4, d_model: int = 128, dropout: float = 0.1,
                 md_bins: int = 64):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True), nn.Linear(64, 128))
        self.md_mlp = nn.Sequential(
            nn.Linear(md_bins, 64), nn.ReLU(inplace=True), nn.Linear(64, 64))
        self.fuse = nn.Linear(128 + 64, 128)
        self.tcn = nn.Sequential(
            DSCBlock(128, d_model, kernel_size=3, dilation=1, dropout=dropout),
            DSCBlock(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
            DSCBlock(d_model, d_model, kernel_size=3, dilation=4, dropout=dropout))

    def forward(self, pc: torch.Tensor, md: torch.Tensor) -> torch.Tensor:
        p = self.point_mlp(pc).max(dim=2).values            # (B, T, 128)
        m = self.md_mlp(md)                                 # (B, T, 64)
        x = self.fuse(torch.cat([p, m], dim=-1)).transpose(1, 2)  # (B, 128, T)
        return self.tcn(x)


class IMUEncoder(nn.Module):
    """IMU temporal encoder. Input (B, T, C) -> (B, d_model, T)."""

    def __init__(self, in_channels: int = 13, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.tcn = nn.Sequential(
            DSCBlock(in_channels, d_model, kernel_size=5, dilation=1, dropout=dropout),
            DSCBlock(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
            DSCBlock(d_model, d_model, kernel_size=3, dilation=4, dropout=dropout),
            DSCBlock(d_model, d_model, kernel_size=3, dilation=8, dropout=dropout),
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        # imu: (B, T, C) -> (B, C, T) -> (B, d_model, T)
        x = imu.transpose(1, 2)
        return self.tcn(x)

"""Evaluate a trained MAFN checkpoint on the OctoNet test set.

One checkpoint is evaluated under three inference regimes selected purely by the
modality mask — full (radar + IMU), radar-only and IMU-only — to demonstrate
missing-modality robustness.

Examples
--------
    python test.py --task har               # MAFN (d=192), 12-class HAR
    python test.py --task fall              # MAFN (d=192), fall detection
    python test.py --task har  --variant mafn-s   # MAFN-S (d=128)
    python test.py --task fall --variant mafn-s
    python test.py --task har  --checkpoint path/to/checkpoint.pth
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from mafn import MAFN
from mafn.data import OctoNetTestSet, collate
from mafn.metrics import (
    accuracy, auroc_binary, f1_macro, f1_positive, f1_weighted,
    precision_binary, sensitivity_specificity, youden_threshold,
)

# variant + task -> model configuration and default checkpoint.
PRESETS = {
    ('mafn',   'har'):  dict(d_model=192, radar_front='pcmd', aux=True,  ckpt='mafn_har.pth'),
    ('mafn',   'fall'): dict(d_model=192, radar_front='pc',   aux=False, ckpt='mafn_fall.pth'),
    ('mafn-s', 'har'):  dict(d_model=128, radar_front='pcmd', aux=True,  ckpt='mafn_s_har.pth'),
    ('mafn-s', 'fall'): dict(d_model=128, radar_front='pc',   aux=False, ckpt='mafn_s_fall.pth'),
}
SCENARIOS = (('full', (1.0, 1.0)), ('radar-only', (1.0, 0.0)), ('IMU-only', (0.0, 1.0)))


@torch.no_grad()
def run_scenario(model, loader, device, force):
    model.eval()
    preds, targets, scores = [], [], []
    for batch in loader:
        batch['radar'] = {k: v.to(device) for k, v in batch['radar'].items()}
        batch['imu'] = {k: v.to(device) for k, v in batch['imu'].items()}
        batch['label'] = batch['label'].to(device)
        B = batch['label'].shape[0]
        batch['modality_mask'] = torch.tensor([list(force)] * B, dtype=torch.float32, device=device)
        logits = model(batch)
        preds.append(logits.argmax(-1).cpu().numpy())
        scores.append(torch.softmax(logits, -1).cpu().numpy())
        targets.append(batch['label'].cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser(description='Evaluate MAFN on OctoNet.')
    ap.add_argument('--task', choices=['har', 'fall'], required=True)
    ap.add_argument('--variant', choices=['mafn', 'mafn-s'], default='mafn',
                    help='mafn = d=192 (accuracy-oriented); mafn-s = d=128 (lightweight).')
    ap.add_argument('--checkpoint', default=None, help='Override the default checkpoint path.')
    ap.add_argument('--data-root', default='data/octonet')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cfg = PRESETS[(args.variant, args.task)]
    ckpt = args.checkpoint or os.path.join('checkpoints', cfg['ckpt'])

    ds = OctoNetTestSet(args.data_root, task=args.task)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = MAFN(num_class=ds.num_classes, d_model=cfg['d_model'],
                 radar_front=cfg['radar_front'], aux=cfg['aux'],
                 imu_channels=ds.imu_channels, pc_in_dim=ds.pc_in_dim).to(args.device)
    state = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())

    print(f'\nMAFN{"-S" if args.variant == "mafn-s" else ""}  '
          f'task={args.task}  d_model={cfg["d_model"]}  params={n_params/1e3:.0f}K')
    print(f'checkpoint: {ckpt}')
    print(f'test samples: {len(ds)}   classes: {ds.num_classes}\n')

    if args.task == 'har':
        print(f'{"scenario":<12}{"accuracy":>10}{"macro-F1":>10}{"weighted-F1":>13}')
        print('-' * 45)
        for name, force in SCENARIOS:
            p, t, _ = run_scenario(model, loader, args.device, force)
            print(f'{name:<12}{accuracy(t, p):>10.4f}{f1_macro(t, p, ds.num_classes):>10.4f}'
                  f'{f1_weighted(t, p, ds.num_classes):>13.4f}')
    else:
        # AUROC is threshold-free; the other metrics are at the Youden-J operating
        # point (sensitivity + specificity - 1 maximised), as reported in the paper.
        hdr = ['recall', 'spec.', 'prec.', 'F1', 'AUROC', 'FAR', 'MDR']
        print(f'{"scenario":<12}' + ''.join(f'{h:>9}' for h in hdr))
        print('-' * 75)
        for name, force in SCENARIOS:
            _, t, s = run_scenario(model, loader, args.device, force)
            pos = s[:, 1]
            yp = (pos >= youden_threshold(t, pos)).astype(t.dtype)
            sens, spec = sensitivity_specificity(t, yp)
            row = [sens, spec, precision_binary(t, yp), f1_positive(t, yp),
                   auroc_binary(t, pos), 1 - spec, 1 - sens]
            print(f'{name:<12}' + ''.join(f'{v:>9.4f}' for v in row))
        print('  (recall/spec/prec/F1/FAR/MDR at the Youden-J operating point; AUROC is threshold-free)')
    print()


if __name__ == '__main__':
    main()

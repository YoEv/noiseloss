# MusicEval 14-Experiment Code Map

This file is the fixed mapping between the 14 experiment families and the executable scripts in `phase7_release/`.

## f01 Loss Curve

- `f01_cnn`:
  - `training/loss_curve/train.py`
  - `training/loss_curve/predict.py`
- `f01_transformer`:
  - `training/curve_transformer/train.py --mode loss`
  - `training/curve_transformer/predict.py --mode loss`

## f02 Entropy Curve

- `f02_cnn`:
  - `scripts/features/extract_entropy_curves.py`
  - `training/entropy_curve/train.py`
  - `training/entropy_curve/predict.py`
- `f02_transformer`:
  - `scripts/features/extract_entropy_curves.py`
  - `training/curve_transformer/train.py --mode entropy`
  - `training/curve_transformer/predict.py --mode entropy`

## f03 SAE

- `f03_cnn`:
  - `scripts/features/extract_sae_features.py`
  - `training/hybrid/train_cnn.py --curve-mode none`
- `f03_transformer`:
  - `scripts/features/extract_sae_features.py`
  - `training/hybrid/train_transformer_pool.py --curve-mode none`

## f04 Loss + Entropy

- `f04_cnn`:
  - `training/loss_curve/train.py` (with entropy manifests)
  - `training/loss_curve/predict.py` (with entropy manifests)
- `f04_transformer`:
  - `training/curve_transformer/train.py --mode loss_entropy`
  - `training/curve_transformer/predict.py --mode loss_entropy`

## f05 Entropy + SAE

- `f05_cnn`:
  - `training/hybrid/train_cnn.py --curve-mode entropy`
- `f05_transformer`:
  - `training/hybrid/train_transformer_pool.py --curve-mode entropy`

## f06 Loss + SAE

- `f06_cnn`:
  - `training/hybrid/train_cnn.py --curve-mode loss`
- `f06_transformer`:
  - `training/hybrid/train_transformer_pool.py --curve-mode loss`

## f07 Loss + Entropy + SAE

- `f07_cnn`:
  - `training/hybrid/train_cnn.py --curve-mode loss_entropy`
- `f07_transformer`:
  - `training/hybrid/train_transformer_pool.py --curve-mode loss_entropy`

## Curve Channel Convention (Important)

- Loss curve = 4 codebook channels.
- Entropy curve = 4 codebook channels.
- `loss_entropy` fusion = `4 + 4 = 8` channels.
- Final input appends one shared mask channel.
- So channel counts are:
  - `loss`: 5
  - `entropy`: 5
  - `loss_entropy`: 9

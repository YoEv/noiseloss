# Handoff

## Summary

All 5 datasets (music_arena, musiceval, musicpref, aime, songeval) have been migrated to **Option C layout**:

```
phase7_release/features/
├── loss/<dataset>/clean/
│   ├── split_train.csv
│   ├── split_val.csv
│   ├── split_test.csv
│   ├── loss_manifest_train.csv
│   ├── loss_manifest_val.csv
│   ├── loss_manifest_test.csv
│   └── train|val|test/*.csv (loss curve files)
├── entropy/<dataset>/clean/
│   ├── split_train.csv
│   ├── split_val.csv  
│   ├── split_test.csv
│   ├── entropy_manifest_train.csv
│   ├── entropy_manifest_val.csv
│   ├── entropy_manifest_test.csv
│   └── train|val|test/*.npy (entropy curve files)
└── sae/<dataset>/clean/
    ├── sae_features_train_meta.pt
    ├── sae_features_val_meta.pt
    ├── sae_features_test_meta.pt
    └── train|val|test/*.pt (SAE latent features)
```

## Validation Results

| Dataset | Loss | Entropy | SAE |
|---------|------|---------|-----|
| music_arena | ✅ 4896/612/612 | ⚠️ 3915/58/61 (partial coverage) | ⚠️ 4897/612/613 |
| musiceval | ✅ 2198/274/276 | ✅ 2198/274/276 | ✅ 2198/274/276 |
| musicpref | ✅ 4024/503/503 | ✅ 4024/503/503 | ✅ 4024/503/503 |
| aime | ✅ 1040/130/130 | ✅ 1040/130/130 | ✅ 1040/130/130 |
| songeval | ✅ 1919/239/241 | ✅ 1919/239/241 | ✅ 1919/239/241 |

**Known Issues:**
- **music_arena entropy**: Only ~80% of samples have entropy features extracted (3915/4896 train). The entropy was extracted from a different audio file set than `full_splits`. Re-extraction needed for complete coverage.
- **music_arena SAE**: +1 extra row on train/test (4897 vs 4896, 613 vs 612).

## Scripts

- `phase7_release/scripts/data/migrate_option_c_features.py` — Migrates features to Option C layout
- `phase7_release/scripts/data/validate_option_c_features.py` — Validates feature integrity

## Config

`phase7_release/config/paths.yaml` updated to point to:
```yaml
token_loss_root: "phase7_release/features/loss"
features_loss: "phase7_release/features/loss"
features_entropy: "phase7_release/features/entropy"
sae.output_dir: "phase7_release/features/sae"
```

## Data Sources

- **Splits**: `phase7_release/data/full_splits/<dataset>/{train,val,test}.csv`
- **Loss curves**: Extracted from audio via MusicGen, stored as per-token loss CSV
- **Entropy curves**: Extracted from codebook activations
- **SAE features**: Latents from MusicGen-small SAE (16384-dim)
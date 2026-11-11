# Experiment protocol (committed before test-set evaluation)

Frozen before any test-set metric was computed. Hashes of the frozen artifacts are recorded in `artifacts/logs/` (`data_audit.json`, `split_summary.json`, `hair_frozen_config.json`).

## Cohort

- C0 = the 10,015 HAM10000 images listed by Harvard Dataverse metadata and present in ISIC collection 212. Recomputed class counts match the expected counts exactly: akiec 327, bcc 514, bkl 1099, df 115, mel 1113, nv 6705, vasc 142.
- C0-extra = 1,705 quarantined images, never used for confirmatory work.

## Splits

- Groups = `lesion_id` merged with pixel-confirmed near-duplicates (21 pairs).
- Class-stratified 70/15/15, seed 42: train 7,011 / val 1,502 / test 1,502.
- Frozen in `splits.csv` with its SHA-256 recorded in `split_summary.json`; zero group, lesion and exact-hash overlap verified.

## Arms

| Arm | Input |
|---|---|
| A | original HAM10000 image |
| B | DullRazor-style morphology plus Telea inpainting on detected hair |
| C | DM-AHR diffusion hair removal (native 128px) |
| A* | original down/up-sampled to 128px, controlling the resolution confound |

Hair-absent images are identical to the original in every arm. Hair present means detected coverage >= 1% of pixels, using the frozen screening config.

## Model and training (identical across arms, seeds 42/123/2026)

- torchvision `vit_b_16`, `IMAGENET1K_V1`, 224px, 7 logits in the order akiec, bcc, bkl, df, mel, nv, vasc.
- AdamW, lr 3e-5, weight decay 0.05, 5-epoch warm-up then cosine decay, batch 32, up to 100 epochs, inverse-frequency class weights from train labels only, early stopping on validation macro-F1 with patience 10, gradient clipping 1.0, bf16 autocast.
- Augmentation identical across arms: RandomResizedCrop(scale 0.75-1.0, ratio 0.9-1.11), horizontal flip 0.5, vertical flip 0.5, rotation 15 deg, ColorJitter(0.15, 0.15, 0.1, 0.02). Validation and test transforms are deterministic: resize 256, centre crop 224, ImageNet normalization.

## Diffusion (Arm C)

- Pinned DM-AHR commit `0aee1bb`, SR3 U-Net (in 6, out 3, 64 inner, multipliers 1/2/4/8/8), 2,000-step linear beta schedule from 1e-6 to 1e-2, 128x128, L1 loss, Adam 1e-4, batch 4.
- Trained only on pairs whose sources are in the train split (654 pairs) and validated only on val-split pairs (182). Test sources never train or select a learned component; leakage is re-checked programmatically and recorded in `training_manifest.json`.
- The released official checkpoint is used for the smoke test only, because its training data is undocumented.

## Primary endpoints (fixed)

1. Test macro-F1 on the full test split, comparing arms A, B and C.
2. Paired PSNR and SSIM against hairless references on held-out synthetic test pairs (186 pairs).
3. X1 lesion-overlap at top 20% salience for Grad-CAM and for attention rollout on the test split.

Planned comparisons are A-B, A-C and B-C with Holm correction across these three per endpoint. Uncertainty uses a lesion-group bootstrap with 10,000 replicates, the same resample shared across arms and seeds, percentile 95% CIs, plus paired group-aggregated Wilcoxon tests.

## Locked test evaluation

Each frozen checkpoint is evaluated on the test split once. Predictions, metrics, the split hash and the checkpoint hash are written together so results cannot be silently regenerated against a different split.

## What would falsify the hypotheses

If C is not better than B or A on the primary endpoints given the reported CIs, the honest conclusion is that DM-AHR hair removal does not improve seven-class classification on this cohort. A null result is a valid outcome.


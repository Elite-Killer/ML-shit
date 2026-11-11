---
name: vit-training-and-eval
description: Train the ViT-B/16 classifiers for all arms and seeds with a matched recipe, then compute the locked-test classification metrics and per-image predictions. Use for Phases 6-8.
---

# ViT-B/16 training and classification evaluation

## Preparation
1. Build one cached uint8 array (or shard set) per arm at 224×224 from the frozen split: A (original), B (classical, hair-present only changed), C (DM-AHR, hair-present only changed), A\* if required. All four share the same row order and ids. Verify each arm has identical id lists and labels.
2. Compute class weights from the **train split only** (inverse frequency, normalized so the mean weight is 1; record the formula and values). Save to the run config.
3. Transform config: ImageNet mean/std from `ViT_B_16_Weights.IMAGENET1K_V1.transforms()`. Train-only augmentation applied identically across arms, on the GPU: flips, rotation ±90°, zoom/shift ±15%, mild brightness. Val/test deterministic.

## Model
`torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`; replace `model.heads.head` with `Linear(768, 7)`. Class order `[akiec, bcc, bkl, df, mel, nv, vasc]` is saved in every checkpoint and asserted when loading.

## Recipe (same for every arm; record real values)
AdamW (record weight decay), lr 3e-5, linear warm-up then cosine decay (record warm-up length), batch 32 (use accumulation if the physical batch is smaller), ≤ 100 epochs, class-weighted cross-entropy, early stopping on validation macro-F1 with patience 10, gradient clipping (default max-norm 1.0), bf16 autocast (fallback fp16+GradScaler or fp32), seeds 42, 123, 2026. Seed Python, NumPy, torch, CUDA; enable deterministic algorithms where possible and note any non-determinism.
No hyperparameter tuning on the test set. Do not tune per arm. Any recipe change applies to all arms and is logged before re-running.
Do not combine class weights with balanced sampling unless running it as a labelled ablation on all arms.

## Runs
- 3 arms × 3 seeds = 9 (+3 for A\*). Launch in parallel Modal containers (see `modal-gpu-pipeline`). Output dir `artifacts/checkpoints/vit/{arm}/{seed}/` with `best.pt`, `final.pt`, `config.yaml`, `class_weights.json`, `train_log.csv`, `final.json`.
- Per-epoch log: train/val loss, accuracy, macro-F1, balanced accuracy, per-class precision/recall, lr, epoch seconds, peak GPU memory, early-stop counter.
- Resumable from the last epoch checkpoint (model, optimizer, scheduler, scaler, RNG, best-metric state).
- Best epoch = max validation macro-F1.

## Test evaluation (once, after freezing)
Before the first test evaluation: commit `reports/experiment_protocol.md` with primary endpoints and the hashes of `splits.csv`, configs, and checkpoints.
For each arm × seed, on the identical test examples, full test set and hair-present subset:
- accuracy, balanced accuracy, macro-F1, weighted-F1;
- per-class precision, recall (sensitivity), F1, specificity (one-vs-rest); macro specificity;
- ROC-AUC one-vs-rest per class and macro, where defined; undefined values recorded as `NaN` plus a reason;
- raw and row-normalized confusion matrices; class supports.
Save `artifacts/metrics/per_image_predictions_{arm}_seed{seed}.csv` (image_id, lesion_id, group_id, true, pred, 7 probabilities, hair_present, arm, seed, checkpoint path), `classification_metrics_by_seed.csv`, `classification_summary_mean_sd.csv`, confusion-matrix and ROC figures, training curves, and the test-manifest hash.

## Checks
- Checkpoint reload gives the same logits within a documented tolerance.
- Output shape `(B, 7)`, finite, softmax sums to 1 within 1e-5.
- Metric code verified on a tiny hand-computed example (including a class with no positives).
- Arms use identical config hashes except the data path/arm tag.
- Do not call "macro accuracy" a metric.

## Interpretation notes for the report
Rare classes have tiny test supports (df, vasc: tens of images), so per-class numbers are noisy. Say so beside the table. If A ≈ B ≈ C within seed variation, report that.

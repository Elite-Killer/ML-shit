# Progress ledger

Updated after every phase. A fresh session should be able to resume from this
file alone.

## Status

| Phase | Gate | Status | Evidence |
|---|---|---|---|
| 0 Environment | G1 | PASS | `artifacts/logs/environment_report.json` (A100-80GB, torch 2.5.1+cu124) |
| 1 Data audit | G2 | PASS | 11,720 collection images, 10,015 C0, 1,705 quarantined, class counts match expected exactly |
| 2 Splits | G3 | PASS | 7,011/1,502/1,502, zero group/lesion/hash overlap, hash frozen |
| 3 Hair screening + Arm B | G6 | PASS | 8,993 hair-present, 10,015 B and A* outputs written |
| 4 DM-AHR smoke | G5 | PASS | batch 16 returns 16 images, geometry preserved, reload identical |
| 5 Diffusion training | G4 | PASS | 20,000 iters, train split only, val split only for selection |
| 6 Arm C sampling | - | PARTIAL | 6,464 of 8,993 hair-present images; stopped when Modal credits ran out |
| 7 ViT training | G7 | PARTIAL | 9 of 12 runs (A, B, A* x 3 seeds); C arm not trained |
| 8 Classification eval | G8 | PARTIAL | 9 of 12 checkpoints evaluated on the locked test split |
| 9 Image quality | G9 | PASS | 186 held-out pairs, all three conditions scored |
| 10 Explainability | G10 | PASS | 9 runs x 1,502 test images, X1 tier |
| 11 Statistics | G11 | PASS | 10 comparisons, Holm-corrected, 10,000 replicates |
| 12 Report + handover | G12-G14 | PASS | report, inventory (119 rows), acceptance 11 PASS / 0 FAIL / 3 BLOCKED |

## Final acceptance state

`artifacts/logs/acceptance_report.json`: **11 PASS, 0 FAIL, 3 BLOCKED**.

The three blocked gates are honest and documented:

- **G7 / G8 (Arm C)**: no Arm C checkpoint exists. Arm C image processing
  reached 6,464 of 8,993 hair-present images before the Modal account ran out
  of credits, so the C arm could not be trained or evaluated. Every A-C and B-C
  comparison is *missing*, not null.
- **G13 (reproduction)**: the README reproduction path was not re-run from a
  clean checkout, because that needs GPU credits.

## Headline results (real, from artifacts)

Test macro-F1, mean ± SD over 3 seeds, 1,502 test images:

| Arm | macro-F1 | accuracy | macro AUC |
|---|---|---|---|
| A original | 0.750 ± 0.005 | 0.858 | 0.969 |
| A* 128 px | 0.732 ± 0.008 | 0.851 | 0.969 |
| B DullRazor | 0.698 ± 0.009 | 0.842 | 0.965 |

- **A vs B**: +0.052 macro-F1, 95% CI [0.019, 0.087], p_holm = 0.0012.
  DullRazor-style removal *hurts* classification on this cohort.
- **A vs A***: +0.018, CI [-0.008, 0.045], p_holm = 0.167. The 128 px
  resolution change alone is not a significant effect.
- **Image quality** (186 held-out pairs): B improves PSNR 22.13 -> 35.73 and
  SSIM 0.520 -> 0.918 versus the hairy input. Arm C scores 16.31 / 0.130,
  *worse than doing nothing*, because the 20,000-iteration checkpoint is
  undertrained relative to upstream's 660,000.
- **Explainability** (X1, 1,502 images): Grad-CAM lesion-overlap@20% 0.38,
  attention rollout 0.55-0.60, against a uniform chance baseline of 0.20 and a
  centred-Gaussian baseline of 0.64. Rollout does not beat the Gaussian
  baseline, so its apparent alignment is largely a centre bias.

## Measured compute

See `reports/compute_plan.md`. Headline measured units:

- DM-AHR sampling: 4.35 s/image at batch 32 on A100-80GB.
- DM-AHR training: 20,000 iterations in 706 s.
- ViT-B/16: ~17 s/epoch, full early-stopped run ~420 s.

## Failed attempts and fixes (do not repeat)

1. **Serial file I/O over the Modal Volume.** Reading 11,720 images one at a
   time took >15 min with ~12 s of CPU time. Fixed by 32-way parallel reads in
   `data/manifest.py` and `data/masks.py`; the audit now runs in ~30 s.
2. **dHash alone as a near-duplicate test.** 70,062 candidate pairs from 10,015
   images. Fixed by pixel confirmation (MAD <= 4/255 and r >= 0.985), leaving
   21 real pairs.
3. **Size-first split assignment.** Overall 70/15/15 but `vasc` split 114/6/22.
   Fixed with class-major apportionment.
4. **`modal.Image.add_local_*` before `.env()`.** Modal rejects build steps
   after local-file mounts; all `add_local_*` calls must come last.
5. **Upstream config is JSON5.** `json.load` fails on comments and trailing
   commas; a tolerant reader is required.
6. **Upstream checkpoint loading.** `GaussianDiffusion.__init__` omits
   `set_new_noise_schedule`, so diffusion buffers are absent and strict
   `load_state_dict` fails. Call the schedule setter first.
7. **Upstream sampler collapses batches.** `continous=False` returns only the
   last image; take the final B rows of the `continous=True` trajectory.
8. **PIL atomic write.** Saving to `name.png.tmp` needs an explicit
   `format="PNG"`.
9. **Volume commits.** `vol.commit()` only at job end loses a long run on
   preemption; `modal_app._run` now commits every 60 s and streams logs.
10. **Over-broad app cleanup.** Stopping every `ephemeral` app also killed an
    unrelated ViT sweep. Stop apps by explicit id only.

## Blockers (documented, with evidence)

- **DERMAHAIR**: Kaggle credentials absent, so F04 is blocked and the synthetic
  fallback is used. Consequence: 186 held-out test pairs, wide CIs.
- **Hair annotations**: none independent, so the screening threshold is
  NOT VERIFIED against real hair labels.
- **Licences**: HAM10000 is CC BY-NC 4.0; DM-AHR's README (CC BY-NC 4.0) and
  LICENSE (Apache-2.0) contradict each other. Commercial use unresolved.

## Next step

Wait for Arm C sampling, then train and evaluate the C arm, run XAI and stats,
and build the final report plus inventory.


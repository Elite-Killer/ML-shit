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
| 6 Arm C sampling | - | PASS | 8,993/8,993 hair-present images at 1,000 steps (labelled ablation) |
| 7 ViT training | G7 | PASS | 12 of 12 runs (A, B, A*, C x 3 seeds) |
| 8 Classification eval | G8 | PASS | 12 of 12 checkpoints evaluated on the locked test split |
| 9 Image quality | G9 | PASS | 186 held-out pairs, all three conditions scored |
| 10 Explainability | G10 | PASS | 9 runs x 1,502 test images, X1 tier |
| 11 Statistics | G11 | PASS | 10 comparisons, Holm-corrected, 10,000 replicates |
| 12 Report + handover | G12-G14 | PASS | report, inventory, acceptance 14 PASS / 0 FAIL / 0 BLOCKED |

## Final acceptance state

`artifacts/logs/acceptance_report.json`: **14 PASS, 0 FAIL, 0 BLOCKED**.

**G13 closed.** Arm A, seed 42 was re-run from scratch into a fresh directory
through the same entrypoint. Recorded and reproduced validation macro-F1 are
identical to 16 significant figures (0.7783361, absolute difference 0.0 against
a 0.02 tolerance). The log is `artifacts/logs/reproduction_log.json`.

One functional item remains open, and it is a scope blocker rather than a defect:
**F04 (DERMAHAIR provenance)**. No Kaggle credentials were configured, so the
kit's required synthetic-hair fallback was used. It is recorded as blocked with
evidence; the DERMAHAIR route was never silently substituted.

## Arm C is a labelled ablation, not the paper's method

Arm C ran at **1,000 sampling steps instead of the reference 2,000**, with bf16
autocast, because the account balance could not cover the full-step run. This is
recorded in decision log D12, in `configs/dmahr.json`, in every per-image CSV
(`num_timesteps` column) and in the report. The A-C and B-C comparisons answer
"does a half-budget DM-AHR help?", not "does DM-AHR help?".

## Headline results (real, from artifacts)

Test macro-F1, mean ± SD over 3 seeds, 1,502 test images:

| Arm | macro-F1 | accuracy | macro AUC |
|---|---|---|---|
| A original | 0.750 ± 0.005 | 0.858 | 0.969 |
| A* 128 px | 0.732 ± 0.008 | 0.851 | 0.969 |
| B DullRazor | 0.698 ± 0.009 | 0.842 | 0.965 |
| C DM-AHR (1,000-step ablation) | 0.426 | — | — |

- **A vs B**: +0.052 macro-F1, 95% CI [0.019, 0.087], p_holm = 0.0012.
  DullRazor-style removal *hurts* classification on this cohort.
- **A vs A***: +0.018, CI [-0.008, 0.045], p_holm = 0.167. The 128 px
  resolution change alone is not a significant effect.
- **A vs C**: +0.324, CI [0.283, 0.366], p_holm < 0.0001. Arm C is far worse,
  consistent with the half-budget sampling producing over-smoothed images.
- **B vs C**: +0.272, CI [0.225, 0.320], p_holm < 0.0001.
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

## Post-delivery alignment pass

The client supplied the proposal (`DM-AHR-ViT_Skin_Lesion_Classification_(2).docx`)
and asked for the results to be checked against it, plus presentation material for
a supervisor discussion.

Found and fixed two gaps where a required deliverable was silently absent:

- **Wilcoxon tests were never computed.** `wilcoxon_group_paired` existed and was
  unit-tested, but `scripts/09_stats.py` hard-coded `wilcoxon_p_raw` to NaN. All
  endpoints now carry group-aggregated Wilcoxon p-values with Holm correction
  (decision log D17).
- **Explainability endpoints never reached the statistics.** The XAI CSVs had no
  `group_id`, so `xai_rows()` returned nothing. Added at generation and backfilled
  from `mask_manifest_x1.csv`. The stats table grew from 14 rows / 3 endpoints to
  20 rows / 5 endpoints.

Also added: a "Compute environment" section (Modal, real versions) and a
"How the results bear on the three hypotheses" section, both generated from
artifacts; decision log entries D14-D16 for the Colab, augmentation and
class-balancing divergences; `docs/CODE_GUIDE.md`; and two notebooks.

Verified: 28 CPU tests pass, notebook 1 executes with zero errors, 14/14 gates
still pass (G11 now rows=20, endpoints=5).

## Next step

Nothing outstanding. A faithful Arm C requires the full training and sampling
budget (decision log D1, D12); everything else is reproducible from
`notebooks/02_pipeline_reproduction.ipynb`.


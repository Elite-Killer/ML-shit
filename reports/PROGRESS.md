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
| 6 Arm C sampling | - | RUNNING | 4 shards over 8,993 hair-present images |
| 7 ViT training | G7 | RUNNING | 12 runs, resumable |
| 8 Classification eval | G8 | PENDING | |
| 9 Image quality | G9 | PENDING | 186 held-out synthetic test pairs |
| 10 Explainability | G10 | PENDING | |
| 11 Statistics | G11 | PENDING | |
| 12 Report + handover | G12-G14 | PENDING | |

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


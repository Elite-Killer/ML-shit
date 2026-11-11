---
name: dm-ahr-diffusion
description: Audit, smoke-test, train and run the DM-AHR diffusion hair-removal model (Arm C) without test leakage, handle DERMAHAIR source mapping and synthetic hair pairs, and control the resolution/blur confound. Use for Phase 5 and any diffusion question.
---

# DM-AHR diffusion (Arm C)

## 1. Audit before integrating
- `git clone https://github.com/AnasHXH/DM-AHR third_party/DM-AHR`; record commit hash and date. Keep LICENSE and README untouched.
- Facts when this skill was written: root-level `sr.py`, `infer.py`, `sample.py`, `eval.py`, folders `config/ core/ data/ dataset/ misc/ model/`, `requirement.txt`; README (credits SR3) says `python main/sr.py -p val -c main/config/akiec.json` although the files appear to sit at the repo root. README covers validation only. Weights are in a Google Drive folder. README says CC BY-NC 4.0, LICENSE file says Apache-2.0. **Verify all of this on the real checkout.**
- Read the configs: image size (names like `akiec_16_128` hint at SR3-style 128 px; verify), channels, value range, normalization, timesteps, noise schedule, sampler, conditioning, checkpoint file names (SR3 code usually saves `*_gen.pth` / `*_opt.pth`; verify), dependency versions. Install in an isolated image if versions conflict.
- Write `reports/license_audit.md` (both statements verbatim, no legal conclusion).

## 2. Smoke tests (on a Modal GPU)
- Download the Drive weights (`gdown`; if blocked or rate-limited, ask the user to download and `modal volume put`).
- Load the checkpoint: does it load with no architecture change? Run 1 image, then a batch of 5–20. Save input | output | reference contact sheet, stdout/stderr, package versions, seconds per image, peak VRAM.
- Check: output finite, in range, same geometry as input, hair actually removed, lesion preserved. Do not claim exact reproduction if you changed anything.
- Provenance: official checkpoint → smoke tests and exploratory comparison only unless you can prove its training sources exclude val/test images of this project.

## 3. Training data (leakage-controlled)
- Inputs: `splits.csv`, `dermahair_source_mapping.csv`, hair screening results.
- **Allowed training pairs**: (a) DERMAHAIR pairs whose mapped source is in the train split; (b) own synthetic pairs: your seeded hair generator applied to **train-split hair-absent** images (the clean image is then a genuine hairless target). Do not overlay synthetic hair on images that already contain hair and call the original "clean".
- **Validation pairs**: val-split sources only, for checkpoint selection.
- **Never**: test-split sources, test labels, test metrics.
- Write `diffusion_training_manifest.csv` listing every training/validation pair with source id, split, origin (DERMAHAIR/own), and generator seed. A pytest asserts no val/test source appears in the training list.
- Synthetic hair generator (`hair_removal/synthetic_hair.py`): thin curved strands (Bezier/random-walk), dark/brown/light variants, thickness, opacity, density, occasional crossings, soft antialiasing; seeded per image; parameters saved. Check examples against DERMAHAIR hairy images for visual similarity.

## 4. Training
- Reference settings from the paper (verify against repo configs): 2,000 timesteps, 20,000 iterations, Adam lr 1e-4. Use the repo's architecture and sampler. Log the actual values.
- First an overfit test on 8–16 pairs (loss falls; sample looks right).
- Resumable: checkpoint every N iterations to the volume with `commit()`. Launch with `modal run --detach`. Validate on val pairs periodically (PSNR/SSIM against the hairless val target) and keep the best by validation PSNR/SSIM.
- If the repo's training path is missing or broken, document the exact failure, then implement a faithful conditional-DDPM reproduction from the paper under `src/dm_ahr_vit/hair_removal/` with its own provenance note.

## 5. Sampling Arm C images
- Only hair-present images (from screening), all splits, so every arm has the same coverage. Hair-absent images are copied from the originals.
- Benchmark one batch first; write the compute plan; stay under the spend cap (see `modal-gpu-pipeline`).
- Shard + batch + `.map` across containers; per-shard manifest; skip finished shards on restart.
- Automated output checks (kit §7.5), then an agent-reviewed sample of ≥ 100 stratified images. Record failure modes with example paths.
- Do not reduce sampling steps silently. A reduced-step or DDIM run is an extra, labelled variant.

## 6. Resolution/blur confound control (required)
Find the model's native resolution. If it is below 224 or the output is resampled:
- Build **Arm A\***: original images passed through the same down/up-sampling round trip (same interpolation) as Arm C. Train it exactly like the other arms. A vs C then mixes hair removal and blur; A\* vs C isolates hair removal.
- Optional ablation: masked composite (diffusion pixels only inside the dilated hair mask).
State which of these you did and why in the decision log.

## 7. Image-quality evaluation inputs
Paired tiers (kit §2.2–2.3): DERMAHAIR pairs with test-split sources (primary), own synthetic pairs from test-split hair-absent images (secondary, labelled). Run Arm B and Arm C on the hairy input; compare each with the hairless reference at the same size and range. Never use test pairs to train, tune, or select.

## Outputs
`artifacts/checkpoints/dm_ahr/` (with provenance JSON), `artifacts/cleaned_images/diffusion/`, `reports/dm_ahr_audit.md`, smoke-test logs/panels, `artifacts/metrics/diffusion_output_checks.csv`, reviewed-sample notes, `diffusion_training_manifest.csv`.

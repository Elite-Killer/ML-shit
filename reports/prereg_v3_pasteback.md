# Stage D pre-registration: full-resolution paste-back of the diffusion fill (arm Cv2p)

Written and committed **before** any Cv2p image is built or scored. Everything below is fixed at the commit that adds this file. Parameters are chosen on validation pairs only; the test split is not read until the validation gates pass and the parameters are frozen.

## Why

Stage C (`reports/stageC_results.md`) showed the rebuilt diffusion arm Cv2 removes synthetic hair (+7 dB) but cannot beat the originals as built, for two reasons visible in existing numbers:

- **Resolution.** Shrinking the originals to 128 px alone costs macro-F1 0.018 (A 0.750, A* 0.732) and Cv2 scores 0.720, so a 128 px arm is capped near A*.
- **Whole-image regeneration.** Cv2 scores 29.2 dB on the test pairs against 35.7 dB for DullRazor, which leaves every non-hair pixel unchanged.

## Design status (read this first)

Cv2p was designed **after** the Stage C test results were seen. It is therefore **exploratory**, not part of the original protocol's primary family. Its statistics are reported in their own Holm family and labelled exploratory in every document. The test pairs are scored a second time for the same diffusion weights with a different post-processing step; this is declared here and in the decision log.

## Method (frozen description)

For each hair-present image (frozen screening, 8,993 of 10,015), with the existing 128 px Cv2 output as the fill:

1. Downsample the original to 128 px with the same bicubic squash the model saw. Flag pixels where the fill is **brighter than the input by more than `tau`** in gray level (dark hair replaced by skin). Dark hair only, as in the harvest; pale hair is not addressed and is listed as a limitation.
2. Dilate the flag `dilate` times with a 3x3 ellipse at 128 px; upsample to full resolution bilinearly (soft edge).
3. Optionally (`refine`) multiply by the frozen full-resolution hair detector mask (`configs/hair_screen.json`) dilated by 2 px, so only pixels that both the diffusion model changed and the detector calls hair are replaced.
4. Output = original x (1 - mask) + bicubic-upsampled fill x mask. Pixels with mask 0 are identical to the original (unit test).
5. Hair-absent images are the original, as in every arm.

No training and no new sampling: the existing `dmahr_v2/final_gen.pth` outputs (sha256 dd06c216...) are reused.

## Parameter selection (validation only)

Grid: `tau` in {8, 16, 24, 32}, `dilate` in {0, 1, 2}, `refine` in {off, on} = 24 settings. Score on the 182 **validation** synthetic pairs with the protocol's scorer (`07b_image_quality.score`: PSNR and SSIM after bicubic resize to 128 px). Choose the setting with the highest PSNR; if two are within 0.05 dB choose the one with the smaller mask. Also report full-resolution PSNR.

## Validation gates (all must pass, else stop and report; no test use)

| Gate | Rule |
|---|---|
| D1 | Best setting's validation PSNR at least 3.0 dB above Cv2 alone (29.02 dB in `image_quality_val_Cv2.json`), i.e. at least 32.02 dB |
| D2 | Best setting's validation SSIM at least 0.90 (Cv2 alone 0.858, DullRazor 0.915) |
| D3 | On the 40 validation real-hair photos (`reports/prereg_v2_diffusion.md`, ids sha256 7243bcc1...) and on the 12 dense-hair validation photos, the median share of frozen-detector hair coverage remaining at **full resolution** is at most 0.5 of the original |
| D4 | Agent-reviewed contact sheet of the 12 dense photos (original, DullRazor, Cv2p, full-resolution tiles): hair visibly removed, no lesion altered, no visible halo or seam; written verdict. Fails if lesion pixels are visibly changed |
| D5 | Unit tests pass, including that pixels outside the mask equal the original exactly |

Reading D3 with care: the detector also fires on lesion texture, which paste-back leaves in place, so the number is conservative.

## After the gates pass (test split opened, once)

1. Freeze `tau`, `dilate`, `refine`, record them and their hash in `artifacts/logs/stageD_params.json`.
2. Build Cv2p for all 10,015 images and for the 182 validation and 186 test pairs (`scripts/48_pasteback.py build`).
3. Score the 186 test pairs once (`07b_image_quality.py --c-arm Cv2p`).
4. Train and evaluate three ViT seeds (42, 123, 2026), recipe unchanged; each checkpoint evaluated once on the test split; explanation overlap as before.
5. Statistics with `09_stats.py --c-arm Cv2p`: planned-style family A-B, A-Cv2p, B-Cv2p, secondary Cv2p-A* and A-A*, Holm within each (endpoint, subset, family). Labelled **exploratory (designed after Stage C results)**.

## How the outcome will be read

- Cv2p macro-F1 above A with a 95% CI excluding zero and Holm p below 0.05: evidence that clean hair removal helps this classifier on this cohort.
- Cv2p within noise of A, A* or B: no evidence that hair removal helps or hurts; with a clean full-resolution remover this is the answer to the headroom question and is reported as a null result.
- Cv2p below A with a CI excluding zero: hair removal hurts, reported as such.
- Three seeds cannot resolve differences below about 0.02 macro-F1 (Cv2 seed SD 0.021). A pooled analysis with more seeds for every arm is **not** part of this stage and would need its own pre-registration.

## Cost

CPU only for parameter selection, building and statistics; three ViT trainings of about 8 to 9 minutes each on A100-80GB (measured in Stage C). Roughly 0.6 GPU-hours. Modal prices unverified.

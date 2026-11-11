---
name: hair-screening-and-dullrazor
description: Implement the hair-presence screening detector and the classical DullRazor-style removal (Arm B), with frozen parameters tuned on train/val only. Use for Phase 4 and whenever hair masks, inpainting, or hair-present/absent labels are needed.
---

# Hair screening and the classical arm

## Purpose
1. Decide which images are `hair_present` (they get hair removal in Arms B and C; hair-absent images pass through unchanged in all arms).
2. Produce Arm B images.
3. Supply hair-absent images as clean targets for synthetic-hair generation (used by the diffusion skill).

## Detector (DullRazor-style)
Pipeline, all parameters in `configs/hair_screening.yaml` and `configs/dullrazor.yaml`:
1. RGB → grayscale.
2. Black-hat morphology with line structuring elements at several orientations (e.g., 0°, 45°, 90°, 135° and intermediates); take the per-pixel maximum response. Consider a light-hair variant (top-hat) and decide from train/val examples whether it is needed.
3. Threshold the response; remove small blobs and large compact regions (lesion borders); keep thin elongated components (use aspect ratio, area, solidity).
4. Dilate the mask to cover hair edges.
5. Output `hair_mask_coverage` = mask pixels / total pixels, and `hair_present = coverage >= tau`.

## Freezing rule (no test leakage)
- Choose kernel sizes, threshold, dilation radius, `tau`, and inpainting radius using **train and val images only**.
- Build contact sheets (original | mask | cleaned) of stratified samples: each class, dense and sparse hair, dark and light hair, images with rulers, gel bubbles, vignettes, dark borders, and lesion borders touching hair. Open them as images and write what you see (agent-reviewed).
- If the Lama et al. ChimeraNet hair masks are obtainable, use them to measure detector precision/recall on train/val images only. If not, say "NOT VERIFIED: no independent hair annotations".
- Freeze the YAML, store its SHA-256 in the decision log, and only then process the test split.

## Arm B inpainting
OpenCV `cv2.inpaint` with `INPAINT_TELEA` by default; `INPAINT_NS` as an option. Process each hair-present image at original resolution and save:
- `hair_mask.png` (0/255), `cleaned.png`, a metadata row (parameters, coverage, timing, status), and error log entries.
Name it "DullRazor-style morphology and inpainting" everywhere.

## Quality checks
- Deterministic: the same config gives byte-identical masks (test with two runs).
- Mask and image dimensions agree; outputs decode, 3 channels, same size as input.
- No silent failures: any exception is logged with the image id and counted; the script's final line prints processed, skipped, failed counts.
- Visual failure catalogue: borders erased, pigment-network lines removed, ruler marks flagged, residual hair left. Save examples for the report.

## Outputs
`artifacts/hair_masks/`, `artifacts/cleaned_images/classical/`, `artifacts/metrics/hair_screening.csv` (one row per image: id, split, coverage, hair_present), panels in `artifacts/figures/`.
Report hair-present counts per class and split.

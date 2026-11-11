# DM-AHR-ViT Development Kit (v2, corrected)

Explainable Vision Transformer for skin-lesion classification with diffusion hair removal.
Scope authority: `docs/proposal.docx`. Persistent rules: `AGENTS.md`. Completion tests: `docs/ACCEPTANCE_CRITERIA.md`.
Compute: Modal GPUs. Stack: Python, PyTorch, torchvision, OpenCV, scikit-learn, pandas, matplotlib, pytest.
Status words used below: **MUST** = required for acceptance. **SHOULD** = do unless blocked, log if skipped. **VERIFY** = unverified claim you must check against the real file/page before relying on it.

---

## 1. Objective, questions, hypotheses

Build a reproducible research pipeline that tests whether DM-AHR diffusion hair removal changes:

- **RQ1 image quality:** is DM-AHR closer to a hairless reference than a DullRazor-style classical method?
- **RQ2 classification:** does a ViT-B/16 trained and tested on DM-AHR-cleaned images differ from one on original or classically cleaned images?
- **RQ3 explainability:** do Grad-CAM and attention-rollout maps overlap lesion masks differently across the three conditions?

Directional hypotheses from the proposal, to be pre-registered and not assumed true:
- **H1** DM-AHR has higher PSNR/SSIM than the classical method on held-out paired data.
- **H2** The DM-AHR arm scores higher on accuracy, sensitivity, specificity, F1, AUC than arms A and B.
- **H3** The DM-AHR arm has higher lesion-overlap and pointing-game scores.

A null or negative result reported honestly meets the project goal.

Out of scope: clinical validation, regulatory work, a diagnostic app, new annotation of the whole corpus, guaranteed accuracy gains, publication.

---

## 2. Data

### 2.1 Primary classification data: HAM10000 via the client's ISIC collection

The client supplied `https://api.isic-archive.com/collections/212/`.

What was verified when this kit was written (re-verify yourself):
- The collection is named HAM10000, lists **11,720 images**, and states **CC BY-NC 4.0**. The page showed images from `ISIC_0024306` onward. It shows "8,838 specified lesions", "0 patients".
- The proposal and the published HAM10000 paper say **10,015** images. 10,015 + 193 + 1,512 = 11,720, which matches the ISIC 2018 Task 3 training set plus its validation (193) and test (1,512) sets. **This is a hypothesis (VERIFY)**: confirm by metadata and by ID ranges before relying on it.

Required handling:
1. Download image files and metadata for collection 212. Use the ISIC API v2 (`/api/v2/images/search/?collections=212`, paginated) or the `isic-cli` package (needs Python ≥ 3.12; run `isic --help`, `isic image download --help`, `isic metadata download --help` to find the real collection flags; do not trust remembered syntax). Anonymous access is expected to work (VERIFY). If the API is unreachable from Modal, record the failure and ask the user for the files.
2. Obtain the authoritative list of the original 10,015 images and their `lesion_id`, `dx`, `dx_type`, `age`, `sex`, `localization`. Preferred source: `HAM10000_metadata.csv` from the Harvard Dataverse record (doi:10.7910/DVN/DBW86T). If the Dataverse is not reachable, derive from the ISIC metadata and document how.
3. Define cohorts and write them to the manifest:
   - **C0** = the 10,015 original HAM10000 images. This is the only cohort for training, validation, test, and the confirmatory classification result.
   - **C0-extra** = every other image in collection 212 (expected ≈ 1,705). Quarantined. Not used in training, validation, model selection, or any confirmatory metric. Optional use: a clearly labelled secondary external test, only after full duplicate/near-duplicate checks against C0 and only if its labels are verified.
4. Compare the recomputed class counts with the expected ones below and record any difference. Do not hard-code them.

| Class | Expected in C0 |
|---|---:|
| nv | 6,705 |
| mel | 1,113 |
| bkl | 1,099 |
| bcc | 514 |
| akiec | 327 |
| vasc | 142 |
| df | 115 |
| total | 10,015 |

Fixed class order everywhere: `[akiec, bcc, bkl, df, mel, nv, vasc]`.

### 2.2 DERMAHAIR (hair-removal training and paired evaluation)

- Source: `https://www.kaggle.com/datasets/riotulab/skin-cancer-hair-removal` (Kaggle credentials needed, Modal secret `kaggle`). Folder layout, naming and size are unknown to this kit: VERIFY by inspection.
- Per the DM-AHR paper (VERIFY): about 700 HAM10000 source images (100 per class) with artificial hair added. It is **synthetic hair on clinical images, not natural hairy/hairless pairs.**
- Source mapping MUST be done before any use in a split-sensitive experiment: match each hairless image to C0 by exact filename, SHA-256, then perceptual-hash plus pixel similarity; resolve ambiguity by a documented rule or agent-reviewed visual check; join to `lesion_id` and split. Write `data/manifests/dermahair_source_mapping.csv` (columns: `pair_id, hairy_path, hairless_path, ham_image_id, lesion_id, class, split, match_method, match_score, confidence, review_status`). Quarantine unmatched or ambiguous pairs from the confirmatory experiment.
- Use rules: pairs whose source is in **train** may train the diffusion model; **val** sources are for diffusion validation/selection; **test** sources are used only for the final image-quality evaluation.

### 2.3 Own synthetic pairs (required fallback and recommended supplement)

Implement a seeded synthetic hair generator (curved thin dark/light/brown strands, varied thickness, density, opacity, occasional crossing; follow the DM-AHR paper's concept, VERIFY its description). Apply it only to **hair-absent** C0 images (per the screening in §6) so the clean image is a true hairless target. Record the seed and parameters per image.
- For diffusion training: use train-split hair-absent sources only.
- For image-quality evaluation if DERMAHAIR mapping leaves too few held-out pairs, or as a secondary table: test-split hair-absent sources, labelled "own synthetic pairs", reported separately from DERMAHAIR.

### 2.4 Lesion masks for explainability (three tiers, never pooled)

The proposal says masks come from "a standard segmentation network". ChatGPT's original kit claimed HAM10000 has no lesion masks. That is not accurate: a "HAM10000 Lesion Segmentations" dataset of binary masks by Tschandl exists on Kaggle (`kaggle.com/datasets/tschandl/ham10000-lesion-segmentations`) (and is believed to be on the Dataverse record). Its creation method (fully automatic, semi-automatic, or manually corrected), coverage and quality are **VERIFY**.

| Tier | Cohort | Masks | Label in all outputs |
|---|---|---|---|
| X1 (primary XAI cohort) | C0 test split (the same images as classification) | Released HAM10000 lesion masks, after QA | `released-masks (provenance: <what you verified>)`; call them "expert" only if the source says expert-traced or manually reviewed |
| X2 (external, SHOULD) | ISIC 2017 Task 1 images, deduplicated against C0, DERMAHAIR, C0-extra | Expert binary masks (0/255) | `expert-masks-external` |
| X3 (fallback only) | C0 test split | Masks predicted by a U-Net trained on ISIC 2017 masks | `pseudo-masks` |

QA for X1 masks (MUST): coverage vs C0, mask/image dimension match, binary values, non-empty rate, and an agent-reviewed overlay of at least 100 stratified samples (all classes). Record failures. If X1 fails QA, fall back to X3 and say so.
Rules: masks never tune any model; X1/X2/X3 are reported in separate tables. A lesion mask must align with every arm's input: confirm Arm C output has no geometric shift before reusing the mask.
Primary XAI endpoint is on X1 because it uses the identical test images and supports lesion-cluster statistics. X2 is the external robustness check; if X2 is blocked (download, licence, dedup), mark it blocked. That does not fail the project.

### 2.5 Optional: hair masks from Lama et al. (ChimeraNet)

The proposal cites manually annotated hair masks on 1,333 HAM10000 images as an independent reference for hair screening. Availability is unknown (VERIFY). If obtainable, use them only to evaluate hair-detector precision/recall on train/val images. If not, state that and use agent-reviewed samples.

### 2.6 Models and code

- Classifier: `torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`.
- DM-AHR: `https://github.com/AnasHXH/DM-AHR` (paper: Cancers 16(17):2947, 2024). Weights: Google Drive folder linked from its README. See §7.
- Classical baseline: your own documented DullRazor-style implementation (§6).

---

## 3. Experimental arms

| Arm | Input to train, val, test | Hair removal |
|---|---|---|
| A | original image | none |
| B | classically cleaned | DullRazor-style morphology + inpainting |
| C | DM-AHR cleaned | conditional diffusion |
| A* (control; MUST if DM-AHR runs below 224 px or its output is resampled) | original image after the same down/up-sampling round trip as Arm C | none |

Why A*: DM-AHR in this family of SR3-style code may run at 128 px or similar (VERIFY from its config). If Arm C is upsampled to 224 it will be blurrier than A. A classifier difference between A and C could then come from resolution, not hair removal. A* isolates that. A* costs no diffusion compute; add 3 more ViT runs.

For images the hair screening marks hair-absent, all arms use the original image. Report every metric on the full test set and on the hair-present subset.
Hair removal is applied once per image and cached; it is never re-run during ViT training.

---

## 4. Data integrity and leakage control

### 4.1 Raw zone
`data/raw/` is immutable (on the Modal data volume). For each file store SHA-256, size, width, height, channels, decode status. Every excluded file has a reason in `artifacts/logs/exclusions.csv`.

### 4.2 Manifest
`data/manifests/data_manifest.csv` with at least:
`image_id, image_path, dataset_name, cohort, class_label, lesion_id, group_id, split, source_pair_id, hair_status, hair_mask_coverage, sha256, phash, width, height, channels, decode_ok, exclusion_reason`.
Separate manifests for DERMAHAIR pairs and for ISIC 2017.

### 4.3 Group definition and split
1. Check `lesion_id` is non-null for all C0 images; never invent one.
2. Compute SHA-256 and perceptual hashes (e.g., pHash + dHash). Flag exact duplicates and near-duplicate pairs under a Hamming-distance threshold you choose by inspecting train/val examples. Confirm candidate near-duplicates by pixel similarity or agent-reviewed view.
3. `group_id` = connected component of the graph whose edges are "same `lesion_id`" or "confirmed near-duplicate". Every group goes to exactly one split.
4. Deterministic, class-stratified, group-aware split with seed 42. A simple approach: `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`; one fold = test, the next fold = validation, the remaining five = train (≈ 71/14/14). Any other group-aware method is fine if deterministic. Report actual counts and per-class counts. Do not split a group to hit exact ratios.
5. Write `splits.csv` and `split_summary.json` (with SHA-256 of `splits.csv`) before any training. Freeze. All arms read the same file.
6. Automated test: group, `lesion_id`, and exact-hash intersection across splits are zero; near-duplicate collisions are zero or explained.

### 4.4 Diffusion leakage rule
The diffusion model is a learned component. Only train-split sources may train it. Val sources only select checkpoints. Test sources are never used for training, selection, or tuning. If the official checkpoint's training sources cannot be established, it may be used only for smoke tests and an exploratory comparison; the confirmatory Arm C uses a checkpoint you train from train-split sources only. If you cannot train one, say the strict-independence claim is not available.

### 4.5 External cohorts
ISIC 2017 images are checked against C0, C0-extra and DERMAHAIR by exact and perceptual hash. Duplicates are removed from X2 and counted.

---

## 5. Common preprocessing and ViT input

- Decode RGB, reject corrupt images with a logged reason.
- Resize to 224×224 (bicubic; the proposal squashes aspect ratio, 600×450 → 224×224; keep that and say so). Apply the identical geometric transform to masks (nearest-neighbour).
- Scale to [0,1], then normalize with the ImageNet mean/std from `ViT_B_16_Weights.IMAGENET1K_V1.transforms()`; record the values.
- Training augmentation (all arms identical, on GPU, online, training split only): horizontal and vertical flip, rotation ±90°, zoom/shift ±15%, mild brightness. Validation and test are deterministic.
- Pre-resize and cache each arm's images as a uint8 array (or shards) on the Modal volume so epochs do not decode JPEGs.
- Save the exact transform config and software versions with every run.

---

## 6. Hair screening and Arm B

### 6.1 Screening (decides who gets hair removal)
Fixed DullRazor-style detector: grayscale → black-hat (and a light-hair variant if you can justify it) with multi-orientation line structuring elements → threshold → connected-component/line-shape filtering → mask. Output `hair_mask_coverage` (fraction of pixels) and a binary `hair_present` using a coverage threshold τ.
- Choose τ and detector parameters using **train and val images only**, with agent-reviewed contact sheets of stratified samples (each class, dense/sparse, dark/light hair, with rulers/gel bubbles/dark borders) and, if available, the Lama masks. Freeze them in `configs/hair_screening.yaml` and hash the file before processing the test split.
- Report detector failures you found (e.g., lesion borders or rulers flagged as hair).

### 6.2 Arm B
Hair mask → dilation → inpainting (OpenCV Telea default; Navier-Stokes as a separate option). Every kernel, threshold, dilation, and inpaint radius is configurable and stored. Call it "DullRazor-style morphology and inpainting", not DullRazor, unless exactly reproduced from Lee et al. 1997.
Outputs per image: mask PNG, cleaned PNG, metadata row; plus side-by-side panels for samples and failures. Deterministic for a fixed config. Output size equals input size.

---

## 7. Arm C: DM-AHR

### 7.1 Repository audit gate (before integrating)
- Clone into `third_party/DM-AHR/`, pin the commit hash and date, keep the notices.
- Known facts when this kit was written: small repo (about 15 commits), files at the root (`sr.py`, `infer.py`, `sample.py`, `eval.py`, `config/`, `core/`, `data/`, `dataset/`, `model/`, `requirement.txt`), README credits SR3, README tells you to run `python main/sr.py -p val -c main/config/akiec.json` but the files appear to be at the repo root: **the README path is probably wrong. Check the real tree.** README documents validation only; no training instructions. Weights are on Google Drive, not in the repo; Drive folder downloads (e.g., `gdown`) may be rate-limited. If blocked, ask the user to download and upload via `modal volume put`.
- Determine from configs: image size, channels, pixel range, normalization, noise schedule, timesteps, sampler, conditioning, checkpoint file format (SR3-derived code usually saves `*_gen.pth` and `*_opt.pth`: VERIFY), required package versions.
- Check whether the provided checkpoint loads with no architecture change. Run a one-image smoke test and a small batch test on Modal; save input/output/contact sheet, stdout/stderr, package versions, elapsed time, peak GPU memory.
- Do not silently change timesteps, sampler, or architecture and still call it DM-AHR. Any change (e.g., fewer sampling steps) is a labelled variant.

### 7.2 Licence gate
README says CC BY-NC 4.0; the `LICENSE` file is Apache-2.0. Record both verbatim in `reports/license_audit.md`. No legal conclusion. Keep the code and weights isolated, notices intact. Flag as unresolved for any commercial redistribution.

### 7.3 Training route (confirmatory)
Reference settings reported by the paper (VERIFY against repo configs; do not assume the public code matches): 2,000 diffusion timesteps, 20,000 training iterations, Adam lr 1e-4.
1. Test the official checkpoint on 5–20 images first.
2. If its training sources are unknown or may include held-out images, do not use it as the only confirmatory model.
3. Train a leakage-controlled checkpoint on: (a) mapped DERMAHAIR pairs with train-split sources, plus (b) own synthetic pairs from train-split hair-absent images (§2.3). Record the mix. Validate/select on val-source pairs only.
4. Sanity check first: overfit 8–16 pairs; loss falls, output looks right.
5. Log noise schedule, steps, image size, optimizer, lr, iterations, batch size, checkpoint cadence, sampler, seeds. Resumable.
6. If the official code cannot run, stop that route, document the exact error, then reimplement faithfully from the paper with separate provenance.

### 7.4 Inference at scale
Sampling with ~2,000 steps is slow. Benchmark first (§12 on compute plan). Process only hair-present images. Batch the sampling, shard across many Modal containers (`.map`), write results per shard, skip finished shards on restart. Do not reduce steps unless you add the reduced-step run as a labelled ablation and the compute plan justifies it.

### 7.5 Output checks (per image, automated)
File exists and decodes; RGB; required size; all finite and in range after decoding; no crop/shift versus input (e.g., low-frequency alignment check or mask-IoU of lesion segmentation between input and output, record the method). Save input, output, paired reference when available, and inference metadata.
Agent-reviewed sample of ≥ 100 images (all if fewer), stratified by class and hair density, including rare classes: flag hallucinated structures, erased borders, colour shifts, blur, checkerboard artefacts, hairs left in place. Do not equate "looks plausible" with "faithful".

### 7.6 Optional ablation (SHOULD if time allows)
Masked composite: use the diffusion output only inside the dilated hair mask and keep original pixels elsewhere. It removes the resolution/blur confound differently from A*. Label it clearly; it is not the paper's method.

---

## 8. ViT-B/16 classifier

- Architecture: ViT-B/16, 224×224, 14×14 = 196 patches, hidden 768, 12 layers, 12 heads, MLP 3072, class token. Replace the head with 7 logits in the fixed class order.
- Config (matched across arms; record the real values): AdamW (record weight decay), lr 3e-5, linear warm-up (record length) then cosine decay, batch 32 (use gradient accumulation if the physical batch is smaller), max 100 epochs, class-weighted cross-entropy with weights from the training split only, early stopping on validation macro-F1 with patience 10, gradient clipping (record max-norm, default 1.0), BF16 autocast on A100 if stable (else FP16+GradScaler or FP32), seeds 42, 123, 2026.
- Imbalance: class-weighted loss is primary. The proposal also mentions class-balanced sampling; do not combine it with weights by default. If run, it is a separate labelled ablation applied identically to all arms.
- Runs: 3 arms × 3 seeds = 9, plus A* × 3 seeds if required. They are independent: run them in parallel Modal containers.
- Select the best epoch by validation macro-F1 only. Save best and final checkpoints, optimizer/scheduler state, config, seed, class weights, class order, git commit.
- Per-epoch log: train/val loss, accuracy, macro-F1, balanced accuracy, per-class precision/recall, lr, epoch time, peak GPU memory, early-stopping state.
- Test evaluation happens once per run after the checkpoint and protocol are frozen.

---

## 9. Explainability

### 9.1 Grad-CAM (ViT adaptation; primary per proposal)
- Target the **patch-embedding feature grid**: output of `model.conv_proj`, shape `(B, 768, 14, 14)`. Register forward hook and gradient hook; backpropagate the logit of the predicted class (also support a given target class for tests). Channel weights = spatial mean of gradients; map = ReLU(Σ weights × activations); upsample bilinearly to 224×224; min–max normalize per image.
- Sensitivity analysis (SHOULD): final encoder block tokens (drop the class token, reshape to 14×14), reported separately. Do not replace the primary method after seeing results; log any change.
- Unit tests: shape, finite values, different target classes give different maps, determinism.

### 9.2 Attention rollout
- torchvision's ViT calls attention with `need_weights=False`, so a plain forward does not return weights. Wrap each encoder block's self-attention to call `nn.MultiheadAttention` with `need_weights=True, average_attn_weights=False` and store the `(B, 12, 197, 197)` tensor, or use a verified alternative. Prove the weights are real (rows sum to 1; recomputed output matches).
- Rollout (Abnar & Zuidema 2020): average over heads → add identity (`0.5·A + 0.5·I`) → row-normalize → multiply layers 1…12 → take the class-token row → drop the class token → 196 values → 14×14 → upsample to 224 → min–max normalize. Same procedure for all arms.
- Unit tests: tensor dims, CLS index 0, finite values, rows sum ≈ 1.

### 9.3 Localization metrics (pre-register before evaluation)
- **Lesion-overlap ratio:** fraction of the top-k% most salient pixels that lie inside the lesion mask. Primary **k = 20%**. Optional sensitivity at 10% and 30%, fixed in advance. Break ties deterministically.
- **Pointing game:** 1 if the maximum-salience pixel is inside the mask, else 0.
- Report chance baselines on the same masks: uniform random map, and a centered Gaussian map. Also report the mean lesion area fraction. (Large lesions raise overlap by chance.)
- Compute for predicted class on all images, and also on the correctly classified subset. Average over the 3 seeds per image for the arm-level estimate; keep per-seed rows too.
- Save raw maps (npy), overlays, per-image metric rows (`mask_id, image_id, lesion_id, arm, seed, method, k, overlap, pointing, mask_tier`), plus representative successes and failures including maps that sit on hair, rulers, gel bubbles, borders, or background.
- Interpretation guardrail: these metrics measure spatial alignment only. They do not show clinically correct reasoning.

---

## 10. Evaluation metrics

### 10.1 Classification (locked HAM test split, per arm × seed, full test and hair-present subset)
Accuracy, balanced accuracy, macro-F1, weighted-F1; per-class precision, recall/sensitivity, F1, specificity (one-vs-rest); macro specificity; macro one-vs-rest ROC-AUC and per-class AUC where defined; confusion matrices (raw and row-normalized); class supports. Mark undefined metrics explicitly (`NaN` + reason), never fill them. Use standard names; do not report "macro accuracy". Ground the class supports next to every per-class figure: the test split has only tens of images for df and vasc.

### 10.2 Image quality (held-out paired data only)
Compare, against the hairless reference: (1) hairy input, (2) Arm B output, (3) Arm C output. PSNR and SSIM with one declared policy: same size, `data_range=1.0` (or 255), colour space (RGB, SSIM with `channel_axis`), no resizing mismatch. One row per pair with `pair_id, ham_image_id, lesion_id, class, split, tier (DERMAHAIR/own-synthetic), paths, PSNR, SSIM`. Report mean ± SD overall and per class. Never compute PSNR/SSIM against a guessed hairless version of a naturally hairy image; use qualitative panels for natural hair.

---

## 11. Statistics

- **Seeds:** report mean ± SD over 3 seeds per arm and metric.
- **Paired comparisons** A–B, A–C, B–C (and A*–C if A* exists, as a labelled extra) on identical examples. Holm correction across the three planned comparisons for each primary endpoint.
- **Primary endpoints (fix before test evaluation, write into `reports/experiment_protocol.md`):** classification macro-F1 (full test); PSNR and SSIM (held-out pairs); lesion-overlap ratio at k=20% for Grad-CAM and for rollout, separately (X1 cohort). Everything else is secondary and labelled so.
- **Lesion-cluster bootstrap** (resample `group_id` with replacement, keep all images of each sampled group), 10,000 replicates, fixed seed. In each replicate use the same resampled groups for every arm and seed, compute the metric per seed, average over seeds, then take the arm difference. Efficient method: precompute a per-group confusion matrix per arm/seed, sum the sampled groups. Percentile 95% CI; two-sided bootstrap p-value `2·min(P(d≤0), P(d≥0))` with the +1 correction; store seed and number of successful replicates.
- **Proposal-faithful test:** Wilcoxon signed-rank on per-group paired differences in per-image correctness (aggregate to group level first, to avoid counting images of one lesion as independent), also Holm-corrected. Report beside the bootstrap result.
- Image-quality pairs: bootstrap over source `group_id`. External X2 images: bootstrap over images.
- Report effect size, CI, raw p, adjusted p, test name. If a test cannot be applied, document why and use a justified alternative.

---

## 12. Modal execution architecture

Full recipes are in the `modal-gpu-pipeline` skill. Summary:

- `modal_app.py`: `modal.App`, one pinned image (CUDA-capable PyTorch + `requirements.lock`, source added via `add_local_python_source`/`add_local_dir`), two Volumes (`dmahr-data`, `dmahr-artifacts`), secrets (`kaggle`, optional `wandb`), one function per phase.
- CPU functions (data audit, hashing, DullRazor, metrics, bootstrap): high `cpu=` count, no GPU. GPU functions: ViT runs, DM-AHR training and sampling, explanation generation.
- Resumability: write a checkpoint on a fixed cadence → `volume.commit()` → on start, `volume.reload()` and resume. Use `modal.Retries`. Launch with `modal run --detach`. Poll with `modal app logs`.
- Compute plan: benchmark one unit of each job on Modal (one ViT epoch, one DM-AHR batch of N images at full steps, one training step block) → `reports/compute_plan.md` → compare with `MAX_SPEND_USD` → proceed or ask.
- Pull small results (metrics CSVs, reports, figures, logs) back to the local repo with `modal volume get`. Checkpoints and image caches stay on the volume; provide `scripts/export_handover.py` and the exact `modal volume get` commands in the README.
- Everything configurable by YAML + env vars; no hard-coded paths.

---

## 13. Phases and scripts

Every phase has a script, a log, and a gate. Run in order; fail fast.

| # | Phase | Script(s) | Gate |
|---|---|---|---|
| 0 | Resource, licence, repo audit; env report from a Modal GPU function | `00_environment_check.py`, `00b_resource_audit.py` | G1 |
| 1 | Download collection 212, define C0/C0-extra, hash, decode, dedup, manifests | `01_validate_datasets.py` | G2 |
| 2 | DERMAHAIR + mask downloads and mapping | `02_match_dermahair_sources.py`, `02b_prepare_masks.py` | G4, G10 |
| 3 | Split + leakage audit | `03_build_splits.py` | G3 |
| 4 | Hair screening, Arm B | `04_screen_hair.py`, `05_run_dullrazor.py` | G6 |
| 5 | DM-AHR audit, smoke test, own synthetic pairs, train, sample | `06_smoke_test_dm_ahr.py`, `06b_make_synthetic_pairs.py`, `07_train_dm_ahr.py`, `08_generate_dm_ahr_outputs.py` | G5, G4 |
| 6 | Compute plan; cache arms (A, A*, B, C) | `08b_cache_arms.py` | – |
| 7 | Train ViTs | `09_train_all_vit_arms.py` | G7 |
| 8 | Classification evaluation | `10_evaluate_classification.py` | G8 |
| 9 | Image quality | `11_evaluate_image_quality.py` | G9 |
| 10 | Explanations + localization | `12_generate_explanations.py`, `13_evaluate_localization.py` | G10 |
| 11 | Statistics | `14_run_statistics.py` | G11 |
| 12 | Report, inventory, handover, acceptance check | `15_build_final_report.py`, `check_acceptance.py`, `export_handover.py` | G12–G14 |

---

## 14. Repository layout

```text
dm-ahr-vit/
├── AGENTS.md  CLAUDE.md  README.md  pyproject.toml  requirements.lock
├── modal_app.py
├── configs/            data, split, hair_screening, dullrazor, dm_ahr, vit, explainability, experiments (.yaml)
├── docs/               proposal.docx, DEVELOPMENT_KIT.md, ACCEPTANCE_CRITERIA.md
├── data/               manifests/ (in git); raw/ interim/ processed/ live on the Modal volume
├── third_party/DM-AHR/ # isolated, commit-pinned, notices kept
├── src/dm_ahr_vit/
│   ├── data/            download, validate, deduplicate, split, datasets
│   ├── hair_removal/    screening, dullrazor, synthetic_hair, dm_ahr_adapter
│   ├── models/          vit_classifier
│   ├── training/        train_classifier, train_diffusion, losses
│   ├── explainability/  gradcam_vit, attention_rollout
│   ├── evaluation/      classification, image_quality, localization, statistics
│   └── utils/           config, reproducibility, logging, hardware, modal_io
├── scripts/            numbered phase scripts + check_acceptance.py + export_handover.py
├── tests/              data_integrity, splits, preprocessing, dm_ahr, vit, explainability, metrics, statistics
├── artifacts/          checkpoints/ cleaned_images/ hair_masks/ explanations/ figures/ metrics/ logs/   (volume; metrics/figures/logs mirrored locally)
└── reports/            PROGRESS.md decision_log.md resource_audit.md license_audit.md data_audit.md
                        compute_plan.md experiment_protocol.md results.md final_research_report.md artifact_inventory.csv
```

---

## 15. Risks and required responses

| Risk | Response |
|---|---|
| Client collection has 11,720 images, not 10,015 | Define C0 vs C0-extra; never train/test on extras; document |
| `lesion_id` missing or different in ISIC metadata | Join with Dataverse metadata; never invent IDs |
| Near-duplicate images across lesion IDs | Merge into `group_id` components |
| DERMAHAIR source mapping incomplete | Quarantine uncertain pairs; own synthetic pairs from train hair-absent images |
| Official checkpoint provenance unknown | Smoke test only; train a leakage-controlled checkpoint |
| README path/config mismatch in DM-AHR | Audit the real tree; write and test an adapter |
| Diffusion output at lower resolution than A | Arm A* control; masked-composite ablation |
| Diffusion too slow/expensive | Benchmark, shard, hair-present only, compute plan vs spend cap; labelled reduced-step ablation only |
| Diffusion hallucination | Agent-reviewed samples; PSNR/SSIM; no fidelity claim beyond evidence |
| Mask provenance unclear | Label tier honestly; QA; X2 as external check |
| Class imbalance, tiny rare-class test support | Macro-F1, balanced accuracy, per-class with supports; wide CIs expected |
| Test-set tuning | Freeze configs and hash them before test runs |
| Licence conflicts (CC BY-NC data and code) | Record; do not conclude; flag for client |
| Preempted/crashed Modal jobs | Checkpoint + resume; retries; ledger |

## 16. What counts as a blocker

Stop the affected phase, write the exact evidence in `reports/PROGRESS.md`, continue anything unaffected, and never fabricate: dataset cannot be downloaded or verified; metadata cannot be joined to images; DERMAHAIR provenance cannot be established for pairs used in confirmatory training; diffusion code/checkpoint cannot run and a faithful reimplementation also fails; masks cannot be paired to images; a checkpoint is corrupt and cannot be resumed; spend cap would be exceeded; credentials or a rights decision are needed.

## 17. References

1. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset. Sci Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161 · data https://doi.org/10.7910/DVN/DBW86T
2. Benjdira B et al. DM-AHR. Cancers 16(17), 2947 (2024). https://www.mdpi.com/2072-6694/16/17/2947 · repo https://github.com/AnasHXH/DM-AHR · DERMAHAIR https://www.kaggle.com/datasets/riotulab/skin-cancer-hair-removal
3. ISIC collection supplied by client: https://api.isic-archive.com/collections/212/ · ISIC 2017 Task 1: https://challenge.isic-archive.com/landing/2017/42/ · ISIC data portal https://challenge.isic-archive.com/data/
4. HAM10000 lesion segmentations: https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations
5. Lee T et al. DullRazor. Comput Biol Med 27, 533–543 (1997). Dosovitskiy A et al. ViT. ICLR 2021. Selvaraju RR et al. Grad-CAM. ICCV 2017. Abnar S, Zuidema W. Quantifying attention flow in transformers. ACL 2020. Gessert N et al. MethodsX 7, 100864 (2020). Lama N et al. ChimeraNet. J Digit Imaging 36, 526–535 (2023). Ho J et al. DDPM. NeurIPS 2020.
6. torchvision ViT-B/16: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html · Modal GPU docs: https://modal.com/docs/guide/gpu

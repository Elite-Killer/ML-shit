---
name: ham10000-data-and-splits
description: Download and audit the client's ISIC collection 212 (HAM10000), define the 10,015-image confirmatory cohort versus extras, build manifests, deduplicate, and create the frozen lesion-grouped 70/15/15 split. Use for Phases 1-3 and any data-integrity question.
---

# HAM10000 data, manifests, and leakage-safe splits

## The key trap
The client's link `https://api.isic-archive.com/collections/212/` is HAM10000 with **11,720 images** (CC BY-NC 4.0), not 10,015. 10,015 + 193 + 1,512 = 11,720, which suggests the original training set plus the ISIC 2018 Task 3 validation and test images. Treat that as a hypothesis and verify it.

## Steps
1. **Fetch**
   - Preferred: ISIC API v2 `GET /api/v2/images/search/?collections=212` (paginated); read image URLs and metadata fields from the response. Or `isic-cli`: run `isic --help`, `isic image download --help`, `isic metadata download --help` and use the real flags. Anonymous access is expected; if not, ask the user.
   - Save raw files and the raw metadata JSON/CSV unmodified under `data/raw/isic_collection_212/`.
2. **Authoritative C0 list**: get `HAM10000_metadata.csv` (Harvard Dataverse, doi:10.7910/DVN/DBW86T) with `lesion_id, image_id, dx, dx_type, age, sex, localization`. If unreachable, derive from ISIC metadata and document it. ID ranges (`ISIC_0024306` upward) are a hint, not proof.
3. **Cohorts**: C0 = the 10,015 images in the authoritative list. C0-extra = everything else in collection 212. Write the counts in `reports/data_audit.md`. Confirm: images without metadata, metadata without images, label set equals the seven classes, `lesion_id` non-null, one `dx` per `lesion_id`.
4. **Validate each file**: decode, RGB, non-zero size, SHA-256, size, width, height. Corrupt → `exclusions.csv` with reason.
5. **Duplicates**: exact (SHA-256) and perceptual (pHash and dHash; compute within C0 and also against C0-extra, DERMAHAIR, ISIC 2017). Choose the Hamming threshold by inspecting examples from train-candidate images, not by looking at test. Confirm candidates by pixel similarity or agent-reviewed side-by-sides. Record every decision.
6. **Group ids**: build a graph over C0 images; edges = same `lesion_id`, or confirmed near-duplicate. `group_id` = connected component. Many HAM10000 lesions have several images; this prevents leakage.
7. **Split**: seed 42; e.g. `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)` on `dx` with `groups=group_id`; fold 0 = test, fold 1 = validation, folds 2–6 = train (≈71/14/14). Report the actual numbers and per-class counts. Do not break groups to hit exact ratios. If a class has too few groups for the method, handle it explicitly and log it.
8. **Freeze**: write `data/manifests/splits.csv` and `split_summary.json` (with SHA-256 of `splits.csv`); commit them. Every later script loads this file and verifies the hash.
9. **Tests** (pytest, run on CPU): zero overlap of `group_id`, `lesion_id`, and exact hashes across splits; near-duplicate collisions zero or explained; extras never appear in a split; class order fixed; metadata join correct on a hand-built mini-example.
10. **Outputs**: `data_manifest.csv` (schema in the kit §4.2), `exclusions.csv`, `reports/data_audit.md` (class table with expected vs actual, discrepancies, exclusions), audit JSON, class-count table, contact sheets of random samples per class.

## DERMAHAIR and masks (also handled here, Phase 2)
- DERMAHAIR: Kaggle `riotulab/skin-cancer-hair-removal` via the `kaggle` secret. Inspect structure first; write what you find. Map each hairless image to C0 (filename → SHA-256 → pHash plus pixel similarity). Write `dermahair_source_mapping.csv` (kit §2.2). Unmatched/ambiguous pairs are quarantined.
- HAM10000 lesion masks: Kaggle `tschandl/ham10000-lesion-segmentations`. Check coverage against C0, dimensions, binary values, non-empty rate. Read the dataset description for how the masks were created and write exactly that in the decision log. Agent-review ≥ 100 stratified overlays.
- ISIC 2017 Task 1 (expert masks 0/255): download from the ISIC challenge page or archive, record the file list and licence, dedupe against C0, C0-extra, DERMAHAIR. This is the external X2 cohort and is non-blocking.

## Stop conditions
Cannot download collection 212; metadata cannot be joined; `lesion_id` missing for many C0 images and the Dataverse CSV unreachable → write a blocker and ask the user for the files.

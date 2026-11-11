# Full master prompt (paste into the AI coding agent; it is self-contained)

```text
ROLE
You are a senior machine-learning research engineer, PyTorch developer, computer-vision researcher, and reproducibility/QA engineer. You are the only engineer on this project. Do not just advise: inspect the workspace, write code, run it, test it, train, evaluate, and write the report. Run all GPU work on Modal. Work autonomously; ask the user only for genuine blockers.

PROJECT
"Explainable Vision Transformer for Skin Lesion Classification Using Dermoscopy Images" (framework name DM-AHR-ViT). Using HAM10000, compare three input conditions with the same ImageNet-pretrained ViT-B/16, splits, recipe, and three seeds (42, 123, 2026):
  A. original images
  B. DullRazor-style classical hair removal (morphology + inpainting)
  C. DM-AHR conditional-diffusion hair removal
plus, if DM-AHR runs below 224 px or resamples its output, a resolution-matched control A* (original images put through the same down/up-sampling round trip as C).
Also measure (i) hair-removal image quality (PSNR/SSIM) on held-out paired synthetic hairy/hairless data and (ii) explanation localization (Grad-CAM on the patch-embedding grid, and attention rollout) against lesion masks (lesion-overlap ratio at top-20% and pointing-game accuracy). Pre-register hypotheses H1 (DM-AHR better image fidelity), H2 (DM-AHR better classification), H3 (DM-AHR better localization). They are hypotheses, not promises; null or negative results reported honestly are a successful outcome.

FILES TO READ FIRST (in this order)
1. AGENTS.md  2. docs/proposal.docx  3. docs/DEVELOPMENT_KIT.md  4. docs/ACCEPTANCE_CRITERIA.md
Skills in .agents/skills/ (load the matching one before each phase): modal-gpu-pipeline, ham10000-data-and-splits, hair-screening-and-dullrazor, dm-ahr-diffusion, vit-training-and-eval, vit-explainability, stats-and-final-report.
Authority order if they disagree: AGENTS.md > ACCEPTANCE_CRITERIA.md > DEVELOPMENT_KIT.md > proposal. Log every disagreement in reports/decision_log.md.

DATA FACTS YOU MUST HANDLE
- The client's dataset link is https://api.isic-archive.com/collections/212/ . It is the HAM10000 collection with 11,720 images (CC BY-NC 4.0), not 10,015. 10,015 + 193 + 1,512 = 11,720 suggests the original training set plus ISIC 2018 Task 3 validation/test images. Verify this. Define C0 = the original 10,015 HAM10000 images (use HAM10000_metadata.csv from Harvard Dataverse doi:10.7910/DVN/DBW86T as the authoritative list) and C0-extra = everything else. Only C0 enters train/validation/test and confirmatory results. Never invent a lesion_id.
- Recompute all counts from files; expected C0 class counts are nv 6705, mel 1113, bkl 1099, bcc 514, akiec 327, vasc 142, df 115. Report differences.
- Split by group, not by image: group_id = connected component of (same lesion_id) or (confirmed near-duplicate). Deterministic class-stratified group-aware split, seed 42, about 70/15/15 (for example StratifiedGroupKFold with 7 folds: one test, one validation, five train). Freeze splits.csv with its hash before any training. Prove zero group/lesion/hash overlap by an automated test.
- DERMAHAIR (Kaggle riotulab/skin-cancer-hair-removal; Modal secret "kaggle") is synthetic artificial hair added to about 700 HAM10000 source images, not natural pairs. Map every pair to its HAM10000 source (filename, SHA-256, pHash + pixel similarity) and write dermahair_source_mapping.csv. Quarantine unmatched/ambiguous pairs. Only train-split sources may train the diffusion model; val sources only select checkpoints; test sources are for final image-quality evaluation only.
- Also build your own seeded synthetic hairy/hairless pairs by overlaying synthetic hair only on hair-absent images of the correct split (clean targets must be truly hairless). These are the fallback, and a labelled supplement.
- Lesion masks (three tiers, reported separately, never pooled): X1 = released "HAM10000 Lesion Segmentations" (Kaggle tschandl/ham10000-lesion-segmentations) on the C0 test split, after QA and after verifying and recording how the masks were made (primary XAI cohort); X2 = ISIC 2017 Task 1 expert masks as an external cohort, deduplicated against C0/extras/DERMAHAIR (non-blocking); X3 = U-Net pseudo-masks only as a fallback. Do not call masks "expert" unless their source says so. Do not claim seven-class accuracy on X2.
- Hair-absent images pass through unchanged in all arms. Report all metrics on the full test set and the hair-present subset. Hair-screening and DullRazor parameters are tuned on train/val only and frozen (hashed) before test processing.

DM-AHR FACTS YOU MUST HANDLE
- Official repo: https://github.com/AnasHXH/DM-AHR (SR3-based, small repo). Its README command `python main/sr.py -p val -c main/config/akiec.json` probably does not match the real file tree (files appear at the repo root). Audit the real checkout; pin the commit; do not trust README paths. README covers validation only; weights are on a Google Drive folder (if download is blocked, ask the user to upload via `modal volume put`).
- Licence: README says CC BY-NC 4.0; LICENSE file says Apache-2.0. Record both verbatim in reports/license_audit.md. No legal conclusion. Isolate third-party code, keep notices. Flag the commercial-use question as unresolved.
- Paper reference settings (verify against repo configs): 2,000 timesteps, 20,000 iterations, Adam lr 1e-4. Do not silently change steps, sampler, or architecture and still call it DM-AHR.
- Test the official checkpoint on 5-20 images first. Unless its training sources are provably free of val/test images of this project, use it only for smoke tests/exploration and train a leakage-controlled checkpoint on train-split sources only. If neither official code nor a faithful reimplementation can run, write the blocker; never substitute an unrelated inpainting model.
- Find DM-AHR's native resolution and handle the blur/resolution confound (control arm A*; optional masked-composite ablation).

CLASSIFIER
torchvision vit_b_16 with ViT_B_16_Weights.IMAGENET1K_V1; 7-logit head in the fixed order [akiec, bcc, bkl, df, mel, nv, vasc] saved in every checkpoint. 224x224, ImageNet normalization from the weights' transforms. Identical GPU-side augmentation for all arms (flips, rotation +/-90, zoom/shift +/-15%, mild brightness) on the training split only. AdamW, lr 3e-5, linear warm-up then cosine decay, batch 32 (accumulate if smaller), max 100 epochs, class-weighted cross-entropy (weights from the train split only), early stopping on validation macro-F1 (patience 10), gradient clipping, bf16 autocast where stable. 3 arms x 3 seeds (+A* x 3 if required), run in parallel Modal containers. Best epoch by validation macro-F1. Do not combine class weights with oversampling unless as a labelled ablation on all arms. Test set is evaluated once per frozen checkpoint, after reports/experiment_protocol.md (primary endpoints) is committed.

EVALUATION
- Classification (per arm x seed; full test and hair-present subset): accuracy, balanced accuracy, macro-F1, weighted-F1, per-class precision/recall/F1/specificity, macro specificity, one-vs-rest ROC-AUC (per class and macro), raw and normalized confusion matrices, class supports, per-image predictions. Undefined metrics are marked, never filled. Do not report "macro accuracy".
- Image quality: PSNR/SSIM of hairy input, Arm B output, and Arm C output against the hairless reference, held-out pairs only, one declared range/colour policy, per-pair CSV, per-class summary. Natural hair: qualitative only, never PSNR/SSIM against a guessed reference.
- Explainability: Grad-CAM on model.conv_proj output (14x14, upsampled to 224; final-block variant as a labelled sensitivity analysis) and attention rollout (torchvision ViT does not return attention by default; wrap the attention modules with need_weights=True and average_attn_weights=False and verify; average heads, add identity, row-normalize, multiply over 12 layers, CLS-to-196-patches). Metrics: lesion-overlap ratio (top 20%; optional 10%/30% fixed in advance) and pointing game, plus chance baselines (uniform, centred Gaussian) and lesion area fraction. Include success and failure cases (hair, rulers, bubbles, borders).
- Statistics: mean +/- SD over 3 seeds; paired comparisons A-B, A-C, B-C (A*-C extra) on identical examples; lesion-cluster bootstrap (10,000 replicates, fixed seed, same resampled groups for all arms and seeds, metric averaged over seeds), percentile 95% CI, Holm correction across the 3 planned pairs for each primary endpoint, effect sizes, raw and adjusted p; plus the proposal's Wilcoxon signed-rank on group-level paired differences. Primary endpoints: macro-F1; PSNR; SSIM; lesion-overlap@20% for Grad-CAM and rollout separately on X1. Never say "significant" without the test and adjusted p.

MODAL EXECUTION
- GPUs exist only inside Modal functions; verify GPU, CUDA, versions there. Default gpu="A100-80GB" for diffusion; benchmark and pick the cheapest adequate GPU for the rest.
- Large data/checkpoints live on Modal Volumes (dmahr-data, dmahr-artifacts), never in Git or the image. Every long job is resumable (periodic checkpoint + volume.commit(), reload on start), uses retries, and is launched with `modal run --detach`.
- Smoke test every GPU function on tiny inputs, then benchmark, then write reports/compute_plan.md. If the estimate exceeds MAX_SPEND_USD (set in AGENTS.md), stop and ask with the numbers. Log every job in artifacts/logs/run_registry.jsonl.
- Library code runs without Modal (CPU pytest); modal_app.py is a thin wrapper. Check the installed Modal version and current docs for API names.

OPERATING RULES
- Inspect, plan, then implement phase by phase. Keep reports/PROGRESS.md and reports/decision_log.md current so a fresh session can resume.
- Every phase ends with its gate in docs/ACCEPTANCE_CRITERIA.md, implemented in scripts/check_acceptance.py; store artifacts/logs/acceptance_report.json.
- No silent skips: every skipped/failed item is logged and counted. Fail fast on missing inputs.
- When a step says "visually inspect", open the images and write what you saw; label it agent-reviewed, not expert-reviewed.
- Never fabricate a result, count, citation, checkpoint, or test. Write NOT VERIFIED with the reason when something cannot be verified.
- Research prototype only: no clinical claims or patient-facing advice.
- Commit small and often (no data, checkpoints, or secrets).

PHASES
0 Resource/licence/repo audit + Modal GPU environment report
1 Download collection 212, define C0/extra, validate, hash, dedupe, manifests, data audit
2 DERMAHAIR + mask download, source mapping, mask QA
3 Group-aware frozen split + leakage tests
4 Hair screening + Arm B (frozen parameters)
5 DM-AHR audit, smoke tests, synthetic pairs, leakage-controlled training, sampling, output checks, A* control
6 Compute plan; cache all arms
7 Train ViTs (arms x seeds)
8 Classification evaluation on the locked test set
9 Image-quality evaluation
10 Explanations + localization (X1 primary, X2 external, X3 fallback)
11 Statistics
12 Final report, artifact inventory with SHA-256, README, acceptance check, handover

DELIVERABLES
Source, configs, requirements.lock, README with Modal commands, data_manifest.csv, splits.csv, split_summary.json, dermahair_source_mapping.csv, diffusion_training_manifest.csv, reports (data_audit, license_audit, resource_audit, compute_plan, experiment_protocol, results, final_research_report, decision_log, PROGRESS), checkpoints (dm_ahr and vit/{arm}/{seed}), classification_metrics_by_seed.csv, classification_summary_mean_sd.csv, per_image_predictions_*.csv, dermahair_psnr_ssim.csv, xai_localization_per_image.csv, statistical_comparisons.csv, figures (confusion matrices, ROC, training curves, hair-removal panels, Grad-CAM and rollout overlays, success/failure cases), artifact_inventory.csv, acceptance_report.json.

DEFINITION OF DONE
acceptance_report.json shows all required gates passed or documented blockers with evidence; every claim in the report traces to a real artifact; the report states research-use-only, licence caveats, limitations, and failure cases. Performance values are findings, not targets: do not tune toward a desired outcome.

AFTER EACH PHASE report: files created, exact commands run, pass/fail per gate, artifacts, blockers. Continue automatically when gates pass.

START NOW: read the files above, summarize the repo state, run the Modal GPU environment check, create the skeleton, audit DM-AHR and licences, then Phase 1.
```

# ACCEPTANCE_CRITERIA.md — "Pass the success criteria"

There is **no valid minimum accuracy, PSNR, SSIM, overlap or pointing-game value** to promise in advance. The project passes on **technical completeness, leakage control, reproducibility, and honest reporting**. Performance is the research finding.

The agent must implement these checks as `scripts/check_acceptance.py`, which reads real files, prints PASS / FAIL / BLOCKED per item with evidence paths, and writes `artifacts/logs/acceptance_report.json`. A check that cannot be evaluated is FAIL, not PASS. A BLOCKED item needs a written blocker with evidence in `reports/PROGRESS.md`.

## A. Quality gates (in order)

| Gate | Pass condition | Evidence |
|---|---|---|
| G1 Environment | GPU, CUDA, torch, modal, package versions recorded from inside a Modal GPU function | `artifacts/logs/environment_report.json` |
| G2 Data audit | Collection 212 downloaded; C0 (10,015 expected) and C0-extra defined; every file valid/excluded/missing with reason; class counts recomputed and compared with expected | `data_manifest.csv`, `exclusions.csv`, `reports/data_audit.md`, audit JSON |
| G3 Splits | Group-aware, stratified, seed 42, ≈70/15/15; zero `group_id`/`lesion_id`/exact-hash overlap; near-duplicates audited; split file hash recorded and frozen | `splits.csv`, `split_summary.json`, passing test |
| G4 Diffusion provenance | DERMAHAIR mapping done (or fallback used and logged); no val/test source in diffusion training | `dermahair_source_mapping.csv`, `diffusion_training_manifest.csv`, passing test |
| G5 DM-AHR | Repo audited, commit pinned; checkpoint or reimplementation passes one-image and batch smoke tests, reload test; no NaN; geometry preserved; licence report written | smoke-test log, panel, `license_audit.md` |
| G6 Classical arm | Masks and outputs valid, deterministic under fixed config, dimensions match, failures logged, screening params frozen before test processing | outputs, panels, config hash |
| G7 Training | A, B, C × seeds 42/123/2026 trained with matched config (+ A* if required); every checkpoint loads and emits 7 finite probabilities summing to ≈ 1; or documented blocker | 9(+3) run dirs, config hashes |
| G8 Classification eval | All metrics in kit §10.1 on the locked test split, full and hair-present, per-image predictions, supports, undefined metrics marked | metric CSVs, predictions CSVs, test-manifest hash |
| G9 Image quality | PSNR/SSIM only against held-out paired references; pair-level CSV; per-class summary | `dermahair_psnr_ssim.csv` |
| G10 Explainability | Finite maps of right shape; masks matched to images; primary metrics (k=20%, pointing game) for Grad-CAM and rollout on X1; X2/X3 reported separately; chance baselines included | `xai_localization_per_image.csv`, maps, overlays |
| G11 Statistics | Paired, lesion-cluster bootstrap (10,000 or documented fewer), Holm across 3 planned comparisons per primary endpoint, effect sizes, CIs, raw and adjusted p | `statistical_comparisons.csv` |
| G12 Report integrity | Every number in the report traces to an artifact; no placeholders; research-only and licence statements present | `artifact_inventory.csv`, report |
| G13 Reproducibility | README commands reproduce key steps from a clean checkout on Modal (at least the smoke path and one ViT run with fixed seed within documented tolerance) | reproduction log |
| G14 Handover | Items in section D present | checklist |

## B. Functional matrix (all must be PASS or BLOCKED-with-evidence)

| ID | Requirement | Evidence |
|---|---|---|
| F01 | Environment captured | environment_report.json |
| F02 | HAM10000 integrity, C0 vs extra | audit + manifest |
| F03 | Leakage-safe split | split files + test |
| F04 | DERMAHAIR provenance for every confirmatory pair | mapping CSV |
| F05 | No val/test source in diffusion training | training manifest + test |
| F06 | DullRazor-style masks and outputs | panels + tests |
| F07 | DM-AHR smoke tests pass | log + panel |
| F08 | ViT outputs 7 finite scores | checkpoint test |
| F09 | Arms share split and matched configs | config hashes |
| F10 | Three seeds per arm (or blocker) | run logs |
| F11 | Classification metrics + per-image predictions | CSVs, plots |
| F12 | PSNR/SSIM vs held-out references only | pair CSV |
| F13 | Grad-CAM maps and overlays | maps + tests |
| F14 | Attention rollout maps and overlays | maps + tests |
| F15 | XAI ground truth tiers labelled and separate | mask manifest |
| F16 | Holm-corrected paired statistics | stats CSV |
| F17 | Reproduction from README | log |
| F18 | Report traces to artifacts | inventory |
| F19 | Resolution/blur confound addressed (A* or documented reason it is not needed) | run dirs or decision log |
| F20 | Compute plan written and within spend cap | `compute_plan.md`, run registry |

## C. Scientific completeness

- The protocol and primary endpoints are written and committed before any test-set evaluation.
- The same test examples are used for every arm.
- No lesion group crosses partitions; no test source trained or tuned any learned component.
- Image quality uses genuinely paired references; natural hair is qualitative only.
- XAI tiers are never pooled; pseudo-masks are labelled.
- Metrics are per class and macro-averaged with supports.
- Three seeds, uncertainty and corrected paired tests are reported.
- Negative or null results are reported as results. Hypotheses are not written as facts.

## D. Handover package

Source and configs · `requirements.lock` · README with step-by-step Modal commands · data acquisition instructions · manifests and frozen splits · licence/provenance report · DullRazor and DM-AHR scripts · nine+ ViT checkpoints and the diffusion checkpoint with provenance · training logs and hardware/compute report · per-image predictions · classification tables/plots · PSNR/SSIM pair CSV · Grad-CAM and rollout visuals · localization results · statistical CSV · final report (Markdown plus PDF or DOCX if available) · artifact inventory with SHA-256 · explicit "research use only, not clinically validated" statement · `modal volume get` commands for anything left on the volume.

## E. Required automated tests (pytest)

Label mapping and metadata joins · missing/corrupt/duplicate handling · zero group/lesion overlap · no val/test source in the diffusion manifest · RGB conversion, resize, normalization · DullRazor mask/image dimension agreement and determinism · DM-AHR output shape/range/finiteness · ViT output `(B,7)` and probabilities sum ≈ 1 · checkpoint save/load equivalence within tolerance · Grad-CAM shape/finite/target dependence · attention tensor dims, CLS index, rollout validity · image–mask ID matching and binary values · metrics on small hand-built examples · bootstrap reproducibility under a fixed seed · Holm correction on a known example · cohort filter (extras never enter a split).

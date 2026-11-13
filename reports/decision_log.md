# Decision log

Authority order applied: `AGENTS.md` > `docs/ACCEPTANCE_CRITERIA.md` > `docs/DEVELOPMENT_KIT.md` > the proposal. Every source disagreement is recorded below with the decision taken.

## D1 - DM-AHR reference iteration count

- Disagreement: the kit states reference settings "2,000 timesteps, 20,000 iterations, Adam lr 1e-4". The shipped upstream `config/akiec.json` has `"n_iter": 660000`, and the released checkpoint is `I725000_E1083` (725,000 iterations).
- Decision: keep 2,000 timesteps, Adam 1e-4 and 128x128 exactly as verified in the config, but train for 20,000 iterations as the kit specifies, and report this divergence.
- Reason: AGENTS.md outranks the kit; the kit's number is the requirement, the config's number is the evidence of what upstream actually shipped. 20,000 iterations is a compute-bounded deviation and is not presented as reproducing the released checkpoint's training budget.

## D2 - Upstream config is not valid JSON

- Finding: `config/akiec.json` contains `//` and `/* */` comments and trailing commas, so `json.load` rejects it.
- Decision: parse it with a small JSON5-tolerant reader that strips comments outside string literals and removes trailing commas. Parsed values are written to `artifacts/logs/dmahr_smoke.json` for audit.

## D3 - Upstream sampler and schedule bugs

- Finding: `GaussianDiffusion.super_resolution(x, continous=False)` returns `ret_img[-1]`, so a batch of N images yields only the last one. Separately, `GaussianDiffusion.__init__` never calls `set_new_noise_schedule` (the call is commented out), so the diffusion buffers do not exist until an explicit schedule call.
- Decision: call `set_new_noise_schedule` explicitly before loading weights, so strict `load_state_dict` validates parameters and buffers together; then take the final B rows of the `continous=True` trajectory to recover the whole batch.
- Verification: the smoke test confirms batch 16 returns 16 images, geometry is preserved, output is finite, and reloading the checkpoint with the same seed reproduces identical pixels.
- Reason: AGENTS.md rule 11 - document the exact failure, then work around it without substituting a different model.

## D4 - Near-duplicate merging

- Finding: dHash at Hamming distance <= 6 produced 70,062 candidate pairs among 10,015 images, far too many to be real duplicates; similar dermoscopic backgrounds collide.
- Decision: re-check every candidate on a common 96x96 grayscale grid and keep it only if mean absolute difference <= 4/255 and Pearson r >= 0.985. 21 pairs survive. Groups merge by `lesion_id` plus these pixel-confirmed matches, giving 7,470 groups for 10,015 images, with 0 groups spanning two diagnoses.
- Reason: the kit requires confirmed near-duplicates; an unconfirmed hash match must not merge lesions.

## D5 - Split construction

- Finding: a size-first greedy splitter hit 70/15/15 overall but put 80% of `vasc` in train and only 3% in val.
- Decision: apportion each class independently under a global size cap. Result: 7,011/1,502/1,502 images with every class at 0.148-0.150 of its total in each split.
- Reason: the kit requires class-stratified splits, and the rare classes have only tens of test images.

## D6 - C0-extra quarantine

- Finding: collection 212 holds 11,720 images; Dataverse metadata lists 10,015. The 1,705 remainder are quarantined as `C0-extra`.
- Decision: all confirmatory work uses C0 only, proven by `test_extras_never_enter_a_split`.

## D7 - Arm A* is required

- Finding: DM-AHR is native 128x128, below the 224px ViT input.
- Decision: include Arm A* (originals round-tripped through the same 128px bicubic down/up-sample) for all three seeds, as the kit mandates when Arm C is below 224px.

## D8 - Hair-screening threshold frozen before test scoring

- Finding: the black-hat/top-hat detector separates synthetic hairy from synthetic clean images perfectly (balanced accuracy 1.0) at coverage >= 0.01, so synthetic pairs do not constrain the threshold sharply. Real-world prevalence at that threshold is 89.8%.
- Decision: freeze 0.01, tuned on train/val sources only. No independent hair annotations exist, so this is recorded as NOT VERIFIED: no independent hair annotations. The threshold is not changed after any test-set metric was computed.
- Reason: changing a frozen parameter after inspecting test-set coverage statistics would be test-set tuning. The high prevalence is reported as a limitation and a cost driver, not hidden.

## D9 - DERMAHAIR availability

- Finding: DERMAHAIR is distributed on Kaggle and needs Kaggle API credentials, which are not configured here (`KAGGLE_USERNAME`/`KAGGLE_KEY` absent).
- Decision: use the kit's required synthetic-hair fallback, generating pairs only from hair-absent C0 images with a recorded seed, and mark the DERMAHAIR route BLOCKED: Kaggle credentials unavailable. F04 is reported blocked-with-evidence, not passed.
- Consequence: test pairs number 186 (every hair-absent C0 test image), so paired image-quality confidence intervals will be wide.

## D10 - Spend cap

- Finding: `AGENTS.md` requires `MAX_SPEND_USD` to be set by the user; no value was supplied.
- Decision: record measured GPU-hours only and set a provisional `max_spend_usd = 250` in `configs/protocol.json`. Modal prices are not guessed.

## D11 - Sharded Arm C sampling

- Finding: Arm C must process 8,993 hair-present images at ~4.35 s/image on A100-80GB, about 10.9 GPU-hours.
- Decision: shard by image across containers with per-shard manifests, skip already-written outputs on restart, and commit the volume periodically. Reduced-step sampling is not used (it would need a separately labelled ablation).

## D12 - Arm C run as a 1,000-step ablation (budget forced)

- Finding: the account held $23. Full 2,000-step sampling of 8,993 images costs
  about $21 on A100-40GB, leaving nothing for the three ViT C trainings,
  evaluations and XAI passes. Measured throughput was 4.38 s/image at 2,000
  steps and 2.18 s/image at 1,000 steps.
- Decision: run Arm C at **1,000 sampling steps with bf16 autocast**, which
  halves the cost and fits the balance. This is recorded as a labelled ablation
  in the report, the config and every per-image CSV (`num_timesteps` column).
- Reason: the kit permits reduced-step runs only as a labelled, justified
  ablation. A complete four-arm comparison at reduced steps is more informative
  than a missing arm, provided the deviation is stated plainly and never
  presented as the paper's method.
- Consequence: Arm C image quality and classification are expected to be worse
  than a full-step run, and the measured results confirm that. The A-C and B-C
  comparisons therefore answer "does a half-budget DM-AHR help?" and not "does
  DM-AHR help?".

## D13 - Arm C passthrough for hair-absent images

- Finding: the sampler only processes hair-present images, but the ViT dataset
  for Arm C reads every split image from `processed/C`. Hair-absent images were
  therefore missing and training failed with `FileNotFoundError`.
- Decision: copy the original image into `processed/C` for every hair-absent
  image, matching the protocol rule that hair-absent images are identical in
  every arm. Implemented in `scripts/05b_fill_c_passthrough.py`; all 1,022
  images filled, giving 10,015 Arm C inputs.

## D14 - Implementation environment: Modal, not Google Colab

- Disagreement: the proposal states four times that the pipeline is
  "implemented in Python with PyTorch in Google Colab with GPU acceleration",
  and section 5.2 makes reporting that environment a deliverable. `AGENTS.md`
  mandates that all GPU work runs on Modal.
- Decision: run every GPU job on Modal, and state the divergence here and in
  the report's "Compute environment" section. Record the real versions:
  Python 3.11.12, torch 2.5.1+cu124, CUDA 12.4, NVIDIA A100-80GB PCIe.
- Reason: `AGENTS.md` outranks the proposal. The proposal's own workload also
  does not fit the described environment: Arm C needs 8,993 images at about
  2.18 s each, roughly 5.4 A100-hours, which exceeds a free Colab session and
  cannot be relied on with Colab's availability-dependent GPU allocation.
- Consequence: Colab is used only to retrieve artifacts. It is not the
  execution environment and the report must not claim otherwise.

## D15 - Augmentation differs from the plus/minus 90 degree rotation proposed

- Finding: the proposal and dev kit specify rotation up to +/-90 degrees and
  zoom/shift +/-15%. `configs/vit.json` applies `RandomRotation(15)` and
  `RandomResizedCrop` (scale 0.75-1.0, ratio 0.9-1.11) with no explicit shift.
- Decision: keep the implemented recipe, which is applied identically to all
  four arms, and disclose the difference in the report's limitations.
- Reason: matched arms preserve the internal validity of the comparison, which
  is what the study measures. Re-running twelve ViT trainings to change
  augmentation would spend the remaining balance without changing the
  conclusion about the hair-removal condition. Disclosed rather than hidden.

## D16 - Class balancing uses loss weights only

- Finding: the proposal mentions "class-balanced sampling with loss weighting".
  The dev kit makes inverse-frequency loss weights primary and warns against
  combining sampling with weights unless reported as a separate ablation.
- Decision: use inverse-frequency class weights only, computed from the train
  split, identical across arms.
- Reason: the dev kit's guidance replaces the proposal's looser wording.
  Recorded here because the proposal implies both were used.

## D17 - Wilcoxon tests added; explainability endpoints now enter the table

- Finding: the proposal and dev kit require a Wilcoxon signed-rank test reported
  beside the bootstrap CI. `wilcoxon_group_paired` existed with unit tests but
  was never called; `scripts/09_stats.py` hard-coded `wilcoxon_p_raw` to NaN.
  Separately, the XAI CSVs carried no `group_id`, so `xai_rows()` returned
  nothing and the two primary explainability endpoints never reached the table.
- Decision: compute group-aggregated Wilcoxon signed-rank tests for every
  endpoint, Holm-corrected within the same family as the bootstrap, and write
  `wilcoxon_p_raw`, `wilcoxon_p_holm`, `wilcoxon_statistic` and
  `wilcoxon_n_groups`. Values are collapsed to one per lesion group before
  testing. Add `group_id` to the `scripts/08_xai.py` output, and backfill it in
  `09_stats.py` from `mask_manifest_x1.csv` for XAI CSVs written before the fix.
- Verification: the table grew from 14 to 20 rows; all 20 carry a non-NaN
  Wilcoxon p-value and the two XAI endpoints are present.
- Reason: both were required deliverables that were silently absent. Computed
  from existing artifacts on CPU, so no GPU spend is involved.

## D18 - External review corrections to the reports

- Finding: two independent reviews of the results documents surfaced four
  claims that the artifacts do not support. Each was checked against the
  source before being changed.
- Corrections:
  1. **Mask provenance.** The report said the source "does not state they were
     expert-traced". The Harvard Dataverse record for doi:10.7910/DVN/DBW86T
     states that a single dermatologist (Tschandl P) created binary masks for
     all 10,015 images, network-initialised and then verified, corrected or
     replaced by hand in FIJI (Tschandl et al., Nature Medicine 2020). The
     report now describes them as single-rater and dermatologist-corrected,
     and notes that no inter-rater agreement exists.
  2. **Duplicate merge was credited as leakage control.** All 21
     pixel-confirmed pairs already shared a `lesion_id`, so the merge changed
     no grouping: 7,470 groups equals the unique `lesion_id` count. The step is
     now described as verifying that no unlinked duplicates exist, not as
     extending the grouping.
  3. **Centre prior beats every arm.** A centred-Gaussian baseline scores
     0.644 on overlap and 0.953 on the pointing game, exceeding every arm on
     both. The report previously said it explained "part of that margin"; it
     now states that no arm beats a trivial attend-to-the-centre prior, so the
     explainability numbers do not establish lesion localisation.
  4. **Wilcoxon mislabelled.** The test uses per-group mean correctness for the
     classification endpoint, so it measures per-lesion accuracy, not
     macro-F1. Columns are relabelled and the test is described explicitly.
     The Holm family column and the four analysis subsets are now visible in
     the table, and A* is rendered as `A*` rather than `Astar`.
- Also corrected: the 660,000-iteration figure now cites its source
  (`third_party/dmahr/config/akiec.json` `n_iter`, commit `0aee1bb`, consistent
  with the released checkpoint name `I725000_E1083`), and the Lama et al. hair
  masks are described as a skipped validation step rather than as annotations
  that do not exist.
- Not changed: Arm C's verdict. A reviewer proposed "failed reproduction,
  H1-H3 untested" as an alternative to "did not help". The reports now say the
  two possible causes (training budget, inference path) have not been
  separated and that the released checkpoint was never scored through this
  adapter, which is a stronger statement than either label.
- Verification: `scripts/16_verify_report_claims.py` recomputes all 42
  report claims from the artifacts; all pass. The three Word documents were
  regenerated and re-rendered to PDF.

## D19 - Arm C diagnosis: undertrained, not an adapter defect

- Question: Arm C scored PSNR 16.3 against a hairy-input floor of 22.1, with
  SSIM 0.130 +/- 0.010 and 185 of 186 pairs below 0.15. A near-constant
  failure of that kind is consistent with either an undertrained model or a
  defect in the inference path, and the two had not been separated.
- Method: `scripts/17_arm_c_diagnose.py` runs on validation synthetic pairs
  only, so no test-set choice depends on it. Three checks: an identity test
  (hairless reference through the full inference path), a head-to-head between
  the released checkpoint and ours at both step counts, and defect checks
  (value range, channel means, spatial alignment, high-frequency energy).
- Result: **outcome (b), undertrained.** The identity test returns PSNR 34.62 /
  SSIM 0.990 at 1,000 steps and 41.72 / 0.992 at 2,000. The released checkpoint
  reaches PSNR 26.80 / SSIM 0.684 (2,000) and 24.66 / 0.641 (1,000), both above
  the 22.11 floor. Our checkpoint reaches 14.99 / 0.079 and 14.12 / 0.073, both
  below it. No flip, no offset beyond 1 px, no range or channel error.
- Consequence 1: the reports' description of Arm C as "over-smoothed" is
  withdrawn. Our output carries 0.269 mean absolute Laplacian against the
  reference's 0.018, with global pixel std only falling from 0.166 to 0.127.
  That is a model that has not learned the mapping, not one that blurs.
- Consequence 2: Arm C stays a labelled under-resourced ablation. The repair
  needs $12.85 (ablation budget) to $24.38 (reference settings); the remaining
  balance is $7.35 after a $1.02 diagnosis, so it does not fit.
- Consequence 3: H1 and H2 are reported as untested for DM-AHR at a realistic
  training budget rather than as evidence against the method, because the
  released weights show this adapter can produce sensible output.
- Cost and evidence: 29.1 min on A100-40GB, $1.02. See
  `reports/arm_c_diagnosis.md` and `artifacts/logs/arm_c_*.csv`.

## D20 - H3 reported as not supported and uninformative

- Finding: a centred-Gaussian prior scores 0.644 on lesion overlap and 0.953 on
  the pointing game, exceeding every arm on both measures. No arm beats a
  trivial attend-to-the-centre prior.
- Decision: H3 is reported as **not supported, and the measurement is
  uninformative as specified**, rather than "partly supported". The distinction
  matters: "partly supported" would imply the saliency measures carry
  information about lesion localisation, and they do not.
- Reason: reporting a direction-of-difference between two methods that both
  lose to a centre prior would overstate what the data show.

## D21 - Test-set reuse disclosed

- Finding: regenerating Arm C would require scoring the 186 test pairs a second
  time. The first scoring is void if the inference path is ever repaired.
- Decision: no repair is affordable, so the test pairs were scored once and the
  diagnosis used validation pairs only. If a future run repairs Arm C, the
  first test scoring must be recorded as void in the report rather than
  presented as an independent second look.

## D22 - Released-checkpoint arm (Creleased) added as exploratory

- Finding: the authors' released DM-AHR checkpoint, run through our pipeline at
  a 25-step respaced schedule, produces usable hair removal (PSNR 27.08 /
  SSIM 0.703 on the 186 held-out pairs, +4.95 dB over the hairy input) and
  classifies at macro-F1 0.713 ± 0.028 over three seeds.
- Decision: report it as an **exploratory** arm, not as a pre-registered
  condition. `scripts/09_stats.py`'s `ARMS` list and the frozen planned family
  are unchanged; Creleased comparisons live in
  `artifacts/logs/statistical_comparisons_exploratory.csv` with their own Holm
  family and an explicit label.
- Reason: the arm was created after the protocol was frozen, and it deviates
  from the paper in two ways - 25 sampling steps instead of 2,000, and a
  classifier trained on our budget rather than the paper's. Folding it into the
  planned family would retroactively change a committed protocol.
- Result: A still beats Creleased on the full test split (difference +0.038,
  95% CI +0.005 to +0.071, Holm p=0.048) but the effect is negligible
  (Cohen's dz 0.05). B vs Creleased is inconclusive (difference -0.015, CI
  -0.049 to +0.019, Holm p=0.400). The released checkpoint does not beat
  classical removal, and diffusion removal remains 8.66 dB behind it on pixel
  fidelity.
- Cost: three evaluations and three XAI runs on A100-80GB, plus one CPU image
  quality pass. No retraining.

## D23 - Eval transform corrected to resize 255 + centre crop 224

- Finding: the reconstructed data package squashed images to a square at eval,
  which changed predictions. Sweeping resize sizes against the recorded A/42
  run showed that resize 255 (int(224 * 1.14)) followed by a 224 centre crop
  reproduces the recorded run exactly: 0 of 1,502 prediction mismatches, maximum
  probability difference 0.0, macro-F1 identical to 16 significant digits.
- Decision: adopt that transform, matching `reports/experiment_protocol.md`
  ("resize 256, centre crop 224" - the recorded runs used 255, the nearest
  integer to 224 * 1.14).
- Reason: the reconstruction gate must prove the data code is faithful before
  any new arm is trusted. It now does, and the check is automated in
  `scripts/21_verify_reconstruction.py`.
- Consequence: the first Creleased evaluation used the wrong transform and was
  discarded and re-run. Only the re-run numbers appear in the report.

## D24 - Two XAI generations were being averaged together

- Finding: `artifacts/logs/` held two sets of explainability CSVs -
  `xai_localization_<arm>_...` (current `08_xai.py`, with explicit chance
  baselines) and `xai_localization_v2_<arm>_...` (an earlier diagnostic with
  different columns). Both matched the `xai_localization_*.csv` glob, so the
  statistics averaged two different measurements of the same arm. The
  regenerated report also showed `nan` in the chance columns for three arms.
- Decision: exclude `_v2_` files from the aggregation in `09_stats.py` and
  `14_results_report.py`, and regenerate the current-format XAI outputs for
  A, A*, B and C so every arm is measured by the same code.
- Reason: an endpoint must come from one measurement, not an average of two.
  The `_v2_` files are kept on the volume, not deleted.
- Second defect fixed: the `group_id` backfill only ran when the column was
  entirely absent. With a mixed file set the column existed but was empty for
  the older runs, so Arm C silently dropped out of the A-C and B-C
  comparisons. The backfill now fills per row.
- Result: XAI endpoints are restored to the values the earlier report carried
  (A Grad-CAM 0.363 vs 0.350, C 0.412 vs 0.412; A roll-out 0.555 vs 0.557).
  Conclusions are unchanged: the two saliency methods still disagree, and no
  arm beats the centred-Gaussian prior (0.644 overlap).

## D25 - Arm C trainer used an inverted noise-level encoding

- Finding: the U-Net's `NoiseLevelEncoder` scales a frequency ramp by a scalar
  noise level. The sampler (`p_sample`, diffusion.py:154) passes
  `sqrt_alphas_cumprod_prev[t+1]`, which decreases with `t` (0.997 at t=50,
  0.011 at t=1900). Our trainer passed `t / num_timesteps`, which increases with
  `t` (0.025 at t=50, 0.950 at t=1900). The two encodings are inverted, so the
  model was trained to read a small value as "clean" and sampled with a small
  value meaning "almost pure noise".
- Evidence: epsilon-prediction error measured under both encodings on 50
  validation pairs. The released checkpoint is 2.9x to 20.5x better under the
  sampler encoding at every timestep. Ours is about 10x better under `t/T` at
  every timestep, the encoding the sampler never uses. See
  `artifacts/logs/noise_convention_check.json`.
- Corroborating: zeroing the conditioning image changes the released model's
  error by 3.3x to 4.8x but ours by only 0.97x to 1.8x, and at t=200 removing
  the condition slightly helps. A model that ignores its condition produces
  output unrelated to the input, matching SSIM 0.074 on test and 0.287 on the
  8-pair overfit attempt.
- Decision: fix the trainer to mirror upstream `p_losses` exactly - per-sample
  continuous `sqrt_alphas_cumprod_prev` uniform between `t-1` and `t`, passed as
  a `(B,1)` tensor. `tests/test_noise_level_convention.py` pins it, so the bug
  cannot silently return.
- Consequence for the budget question: **training longer on the old trainer
  would have reproduced this defect at any iteration count.** The answer to
  "would more iterations help" is therefore no, until this fix. Any future
  budget proposal must retrain from scratch with the fixed trainer, not resume
  the existing checkpoint, whose weights encode the wrong convention.
- Second defect fixed in the same pass: `PairedDataset` opened manifest paths
  verbatim, so training failed on any volume other than the one that wrote the
  manifests. It now re-roots on the `/data/` marker like the rest of the code.

## D26 - The 25-step sampling routine is not in the repository

- Finding: the `Creleased` images were produced at 25 timesteps and score
  26.91 dB on validation pairs, but the committed code cannot reproduce that.
  Schedule truncation to 25 steps gives 11.44 dB. A DDIM respacing I wrote gives
  23.23 dB. The upstream `jsd` schedule diverges to non-finite values. The
  recorded values also decrease with more steps, while mine do not, so it is a
  different routine rather than a different parameter.
- Decision: do not claim the repository reproduces the Creleased sampling. The
  Creleased images and every measurement taken from them stand, because they are
  real artefacts on the volume. Regeneration is a different matter.
- Consequence: the earlier "$1 to regenerate 8,993 images at 25 steps" figure
  is withdrawn. Until the original routine is recovered, regeneration is priced
  at 2,000 steps, about $26. The report and this log now say so.
- Kept anyway: the DDIM routine is a real improvement over truncation (23.23
  versus 11.44 dB at 25 steps) and is used only where labelled as such, never
  described as reproducing the recorded pipeline.
- Evidence: `artifacts/logs/respacing_check.json`,
  `scripts/28_respacing_check.py`.


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


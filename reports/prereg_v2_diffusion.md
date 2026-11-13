# Pre-registration: building a diffusion hair remover that works on real hair

Written and committed before any experiment below is run. Validation pairs and train-split images
only; the 186 test pairs and all test images stay unread until a checkpoint is frozen.

## Scope (from `docs/proposal.docx`, section 3.3)

A self-supervised conditional diffusion hair remover (DM-AHR). Training pairs are made by compositing
*realistic* hair artefacts onto clean dermoscopic images. The proposal also says the gap between synthetic
training pairs and real hairy images must be quantified. Our real-hair check (`reports/hair_check_prereg.md`)
quantified it: the released model leaves real hair in place (change ratio 1.13). Our own model was trained
on 654 pairs with simple curved hair, which is not "realistic hair" in the proposal's sense.

## Evidence this plan follows from

- Our fixed trainer denoises well at low and middle noise (one-step error below the released model's up to
  t=1000) and is weak at high noise (t=1500 and above: 0.0086 against 0.0048 at 50k iterations).
- Sampling from pure noise is unstable across noise draws (one pair ranged 6.5 to 20.0 dB).
- Released-model respacing (DDIM, 25 steps) scored 23.2 dB on validation pairs, so strided sampling works.

## Stages, each with a gate

### Stage A: can the sampler be fixed at inference? (this document's first experiment)

Method: DDIM, 50 strided steps, started from the noised *hairy input* at schedule index t0 instead of from
pure noise (`start_t` in `respaced_super_resolution`; tests in `tests/test_partial_noise_sampler.py`).
Script `scripts/38_partial_noise_eval.py`, 24 validation pairs, 4 noise seeds, t0 in {600, 1000, 1500, 1999}.
Models: fixed-trainer snapshot at 50,000 iterations (`gen_iter050000.pth`) and, as the positive control
for the script, the released checkpoint at t0 in {1000, 1999}. 1999 means a pure-noise start.

Predictions:
- Released at t0=1999: mean margin above the hairy floor of at least +2.0 dB (earlier respacing check gave
  23.2 dB against a floor of about 22 dB). If it does not, the script is wrong and nothing else counts.
- Fixed model at t0=1999: fails, as pure-noise sampling did (margin below 0).
- Fixed model at t0=1000: passes.

Pass for the sampler (fixed model, any t0 at or below 1500): mean margin at least +2.0 dB **and** at most
10% of the 96 (pair, seed) runs below that pair's hairy-input PSNR **and** the standard deviation of the
four per-seed means at most 0.5 dB.

If it fails at every t0, the sampler problem is in training at high noise and Stage B must address that
before hair realism matters.

This stage measures only synthetic hair of our own style. It says nothing yet about real hair.

### Stage B: realistic hair and more targets (not run yet; will be pre-registered separately)

Composite real hair layers, harvested from real hairy train-split images with a thin-structure detector,
onto low-hair train-split images, regenerated every iteration. Warm-start from the Stage A model.
Gate: the real-hair check on 40 validation-split real-hair images, pass if the inside/outside change ratio
is at least 2 and a visual review shows hair gone and lesions intact.

### Stage C: process all hairy images, retrain three ViT seeds, score the frozen checkpoint once

Only if Stage B passes. Estimated cost with strided sampling is a few dollars, not the earlier $26 figure for
2,000-step sampling (unverified).

## Budget

No spend cap was set by the owner. Stage A costs well under $1. Stages B and C are expected at a few
dollars each and will be launched only after their gates, with the estimate stated first.

## Stage A, first run (commit 7cd1505, 24 validation pairs, 4 seeds, DDIM 50 steps)

| Model | t0 | PSNR | margin over hairy floor (21.3) | runs below floor | sd of seed means |
|---|---:|---:|---:|---:|---:|
| fixed, 50k iterations | 600 | 30.25 | +8.95 | 0% | 0.10 |
| fixed, 50k iterations | 1000 | 30.55 | +9.25 | 0% | 0.29 |
| fixed, 50k iterations | 1500 | 26.74 | +5.43 | 13% | 0.64 |
| fixed, 50k iterations | 1999 (pure noise) | 16.30 | -5.01 | 92% | 0.41 |
| released (control) | 1000 | 25.68 | +4.38 | 0% | 0.10 |
| released (control) | 1999 (pure noise) | 23.26 | **+1.96** | 28% | 0.25 |

Artifacts: `artifacts/diagnosis/stageA_*_partial_noise.json` (checkpoint hashes and commit recorded).

**The positive control missed its bar by 0.04 dB.** The pre-registered bar was a margin of at least +2.0 dB for the released model at t0=1999, and it scored +1.96. By my own rule that means the script is not yet validated. The independent evidence points the other way: an earlier, separate check (`artifacts/logs/respacing_check.json`, different code path, 25 pairs) gave the released model 23.23 dB at 25 steps and 24.23 dB at 50 steps, and this run gives 23.26 dB. A 0.04 dB miss on a 24-pair mean is inside what a different pair set changes. I am treating the first run as indicative only.

**Replication, rules fixed before running it.** Same script on 48 validation pairs the first run never saw (pairs 24 to 71, `--offset 24`), for the fixed model at all four t0 and the released model at t0 in 1000 and 1999. The bar is unchanged: released at t0=1999 needs a margin of at least +2.0 dB. If it misses again, Stage A is recorded as not validated by the strict bar. The fixed-model sampler counts as passing only if the pass criteria above hold on both pair sets.

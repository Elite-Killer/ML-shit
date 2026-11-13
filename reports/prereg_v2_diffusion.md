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

# One-step reconstruction diagnostic: does more training explain Arm C?

**Answer: yes for image quality, no for your result.** Training longer should
improve Arm C substantially, but it will not change the finding, because the
authors' own fully-trained model already caps at the baseline.

Evidence and reasoning below. Every number traces to
`artifacts/logs/oneshot_diagnostic.json`, produced by
`scripts/25_oneshot_diagnostic.py` on 100 validation pairs.

## What was measured

The question was whether Arm C failed because the weights never learned the
mapping, or because the sampling loop is broken. A sampling loop can only work
if the model can denoise correctly at a single step, so I tested exactly that.

For each checkpoint and timestep `t`: take a clean hairless reference `x0`,
noise it with a fixed epsilon to get `x_t`, ask the model to predict that
epsilon, and recover `x0` in one step. This bypasses the sampling loop entirely.
It uses the same conditioning arrangement and the same noise-level embedding as
the upstream sampler, verified line by line against
`third_party/dmahr/model/sr3_modules/diffusion.py`.

Single-step reconstruction from heavy noise is ill-posed, so absolute values at
high `t` are not meaningful on their own. What matters is the comparison between
checkpoints at the same `t`, and whether either beats the do-nothing floor.

Validation pairs only. The test set was not read.

## Results

One-step x0 reconstruction PSNR in dB, 100 validation pairs, identical noise
seed for both checkpoints.

| t | Released (725k) | Ours (20k) | Gap |
|---:|---:|---:|---:|
| 50 | 38.56 | 31.54 | 7.02 |
| 200 | 33.90 | 24.58 | 9.32 |
| 500 | 31.00 | 23.44 | 7.56 |
| 1000 | 29.51 | 21.38 | 8.13 |
| 1500 | 25.38 | 8.25 | 17.13 |
| 1900 | 14.61 | 6.36 | 8.26 |

SSIM at the same points:

| t | Released | Ours |
|---:|---:|---:|
| 50 | 0.931 | 0.678 |
| 200 | 0.856 | 0.307 |
| 500 | 0.787 | 0.271 |
| 1000 | 0.737 | 0.289 |
| 1500 | 0.484 | 0.021 |
| 1900 | 0.095 | 0.013 |

The do-nothing floor, meaning the unprocessed hairy image scored against the
same hairless reference, is **22.11 dB / 0.519 SSIM**.

## What this shows

**1. The sampling loop is not the defect.** The released checkpoint reaches
26.80 dB through the full loop using the same adapter, and a hairless reference
round-trips through it at 34.62 dB. Both facts were established earlier and
stand. A broken loop cannot produce either.

**2. Our model is not untrained. It is under-trained.** At `t=50` our checkpoint
reaches 31.54 dB, which is 9.4 dB above the do-nothing floor. It has learned to
denoise. This rules out the strongest version of the "never learned the mapping"
hypothesis that an earlier report implied.

**3. The deficit is uniform across noise levels, which is the signature of
incomplete convergence.** Our checkpoint is behind the released one at every
single timestep, by 7 to 9.3 dB across the low-to-mid range. It is not failing at
one specific stage, which is what a structural bug would produce. It is
uniformly less converged, which is what fewer iterations produces.

**4. Our model has learned denoising better than it has learned the
conditioning.** Zeroing the hairy conditioning image changes our output by only
0.0234 at `t=200`, against 0.1076 for the released model. At `t=200` our output
is actually marginally *better* with the condition removed (24.72 versus 24.58).
That is a model that is reconstructing generic skin texture and largely ignoring
the hair mask it is supposed to be conditioning on.

**5. The sampling loop amplifies our model's weakness at high noise.** The
released model converts 38.56 dB single-step quality into 26.80 dB through the
full loop, a loss of 11.8 dB. Ours converts 31.54 into 14.99, a loss of 16.5 dB.
The extra loss is where ours is weakest: at `t=1500` it is 17.1 dB behind the
released model, and a sampling loop spends many of its steps in that region. So
the loop is not broken, but it does propagate the high-noise deficit.

This fourth point is the one that complicates a simple "just train longer"
answer. Learning a conditional mapping is exactly the part that requires many
iterations, so more training is the most likely fix, but it is also the part
where a training-setup problem would show up. I cannot separate those two from
this evidence alone.

## Why this supports "yes" on more training

The model learns, it is uniformly behind at every noise level, and the released
checkpoint proves this architecture and this code path reach good quality. Those
three facts together make incomplete convergence the most likely explanation,
and convergence is what iterations buy.

Expected size of the gain: our one-step reconstruction at low noise is 7 dB
behind the released model. The full-loop gap is 12 dB (26.80 versus 14.99). If
training closes most of the one-step deficit, most of the loop deficit should
close with it. The high-noise rows are the clearest target, because that is
where ours is furthest behind and where the loop spends much of its time.

## Why this still supports "no" on changing your result

The released checkpoint is the authors' own fully-trained model at 725,000
iterations. It scores macro-F1 0.713. That is:

- statistically tied with classical DullRazor: difference −0.015, 95% CI −0.049
  to +0.019, Holm p = 0.40;
- below original images: difference +0.038, 95% CI +0.005 to +0.071, Holm
  p = 0.048, though with a negligible effect size of Cohen's dz 0.05.

So the realistic outcome of training our model to full budget is Arm C moving
from 0.426 to roughly 0.70, which matches classical removal and remains behind
the untouched baseline of 0.750. A better-trained diffusion arm does not make
diffusion hair removal win.

## Recommendation

If you need the diffusion arm at published budget for the write-up, the training
is justified: it removes the "under-trained" caveat and makes the arm a fair
test. Expect Arm C around 0.70, not above 0.750.

If the goal is a result that beats baseline, do not spend it. Nothing in this
evidence, or in the authors' own results, supports that outcome.

The intermediate option is the one I would take: train to 100,000 iterations,
save checkpoints, and re-run this same single-step test plus a 50-pair sample.
If the one-step gap at `t=50` and `t=200` closes materially, continue to full
budget. If it does not move, stop and report what you have.

## Limitations of this diagnostic

This test measures single-step reconstruction, not final image quality. A model
that reconstructs well at one step can still sample poorly, and vice versa.

Single-step reconstruction from `t=1900` is close to pure noise for both
checkpoints, so those rows should be read as "both are guessing", not as a
quality ranking.

Two checkpoints give two points on a curve. This diagnostic strongly implicates
convergence but cannot measure a learning trajectory, because no intermediate
checkpoints exist.

An earlier diagnostic in `artifacts/logs/arm_c_v2_noise_mse.json` reported that
our checkpoint and the released one prefer different noise-prediction
conventions. Its generating script is not in the repository, so its "legacy" and
"upstream" modes cannot be verified from source and I do not rely on it here.
The one-step test above is the stronger evidence, and it shows our checkpoint
behaving sensibly under the standard convention, which argues against a
wholesale convention mismatch.

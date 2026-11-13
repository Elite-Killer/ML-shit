# DM-AHR-ViT: plain-English summary

Research prototype only. Not clinically validated and not intended for diagnosis or patient care.

## The question

Does removing hair from skin-lesion photos help a computer program classify the lesion? We compared
kinds of input on the 10,015 HAM10000 photos: the original photos, photos cleaned by a classical
hair-removal tool (DullRazor), and photos cleaned by an AI hair-removal model (DM-AHR).

## The scores

How well each version classified the held-out test photos (macro-F1, higher is better, average of
three runs; about 0.02 of run-to-run spread for the AI arm):

| Input | Score |
|---|---:|
| Original photos | **0.750** |
| Original photos shrunk to 128 px (a fairness control) | 0.732 |
| AI hair removal, rebuilt and retrained by us | 0.720 |
| AI hair removal, authors' released model (exploratory) | 0.713 |
| Classical hair removal | 0.698 |
| AI hair removal, our first training run (had a bug) | 0.426 (failed) |

## What this shows

- **The AI model now works.** Our first run failed because of a bug in our training script (it fed the
  network the noise level backwards). We fixed it, retrained, taught the model with real hair strands
  cut from photos and pasted onto clean ones, and it now removes most of the hair on 12 heavily hairy
  validation photos while keeping the lesion's outline and colour.
- **Hair removal did not change the classification score in a way we can measure.** The rebuilt AI arm
  (0.720) is statistically indistinguishable from the originals (0.750), from the shrunk-photo control
  (0.732) and from the classical tool (0.698). Only "originals beat the classical tool" is clear
  (0.052 higher). With three runs per arm, differences of a few hundredths cannot be resolved.
- **On picture quality the classical tool wins on our test.** On 186 test photos where we drew hair
  ourselves, the classical tool recovered the clean picture better (35.7 dB against 29.2 dB for the
  AI model; the hairy photo itself is 22.1 dB). The AI model rebuilds the whole 128-pixel picture and
  loses fine detail, while the classical tool only touches the hair. Our drawn hair is simple, so this
  says little about real hair, where the classical tool smeared lesions.
- **Where the model looks.** When we compare with the shrunk-photo control, the AI model's attention
  falls on the lesion about as much (one measure) or slightly more (the other measure, 0.60 against
  0.58). The second gain is small and depends on only two of three runs.

## What is not proven

- **The check we set for "removes hair from real photos" was not met.** Our planned test (a ratio of
  change inside versus outside the hair) came out at 1.95 and 1.57 against a bar of 2.0. We continued
  anyway, knowing the test is a poor ruler for a model that redraws the whole picture, and we say so
  in every report. Evidence that the model removes real hair is a visual look at 12 photos and a test
  on pasted real hair, not a passed pre-registered test.
- **No claim that hair removal helps or hurts diagnosis.** The study found no measurable difference.

## Why we could not do better

- The dataset of real hairy and hairless photo pairs (DERMAHAIR) was not available to us. We used real
  hair cut from other photos and artificial hair, which give a true reference picture but are not the
  same as real hairy-versus-clean pairs.
- Our AI model works at 128 pixels, so lesions lose fine texture.
- The authors' released model very probably saw our test photos during its own training (their paper
  says it trained on HAM10000). It used no labels, so this mainly affects image-quality numbers.
- The exact procedure that produced the released model's cleaned images is not in the code we have.

## What is solid

- No test photo was used to choose settings or to tune any model. Photos were split by lesion, so the
  same lesion never appears in both training and test. The rebuilt model and the procedure were frozen
  and written down before its test photos were processed, and the test set was used once for it.
- All planned checks pass, earlier results reproduce, and every number in the reports traces to a saved
  file. Earlier claims that turned out wrong (an "undertrained" diagnosis, a failed validation gate that
  was really an 8-pair smoke run) were retracted and corrected in every report.
- Cost of the whole rebuild was a few dollars of cloud compute (price not verified).

## Optional next step

Train the model at higher resolution or add a step that puts the original sharp pixels back outside the
hair, so lesions keep their texture, then rerun the comparison with more than three runs per arm to
see whether a small gain exists. Costs would be a few dollars to tens of dollars; the outcome is
uncertain.

## Licensing

HAM10000 is CC BY-NC 4.0. The DM-AHR repository's README and LICENSE contradict each other (CC BY-NC
4.0 against Apache-2.0), so commercial use is unresolved and needs a rights decision.

# DM-AHR-ViT: plain-English summary

Research prototype only. Not clinically validated and not intended for diagnosis or patient care.

## The question

Does removing hair from skin-lesion photos help a computer program classify the lesion? We compared
three kinds of input on the 10,015 HAM10000 photos: the original photos, photos cleaned by a
classical hair-removal tool (DullRazor), and photos cleaned by an AI hair-removal model (DM-AHR).

## The scores

How well each version classified the held-out test photos (macro-F1, higher is better, average of
three runs):

| Input | Score |
|---|---:|
| Original photos | **0.750** |
| Original photos shrunk to 128 px (a fairness control) | 0.732 |
| Classical hair removal | 0.698 |
| AI hair removal, authors' released model (added later, exploratory) | 0.713 |
| AI hair removal, our own training run | 0.426 (failed) |

## What this does and does not show

Original photos scored best. But this is not a fair test of hair removal, because none of the
hair-removal arms cleanly removed real hair:

- **The classical tool damaged lesions.** On real-hair images it removed the hair but smeared the
  lesion and skin texture, and in one image it erased the lesion. The likely cause is our settings.
- **The authors' AI model left real hair in place.** It was trained on artificial hair drawn onto
  photos, and on real hair it mostly did nothing.
- **Our own AI run failed because of a bug in our training script.** We found and fixed the bug, but
  the fixed version still did not produce usable results with the time and data we had.

So the study did **not** show that hair removal helps, and it did **not** show that it hurts. A
fair answer needs hair removal that works on real hair.

## Why we could not do better

- The dataset of real hairy and hairless photo pairs (DERMAHAIR) was not available to us. We used
  artificial hair, which gives a true reference image but does not look like real hair.
- Our AI model trained on 654 images; the authors' used about 9,300.
- The authors' released model very probably saw our test photos during its own training (their paper
  says it trained on HAM10000). It used no labels, so this mainly affects image-quality numbers.
- The exact procedure that produced the released model's cleaned images is not in the code we have.

## What is solid

- No test photo was used to choose settings or to tune any model. Photos were split by lesion, so the
  same lesion never appears in both training and test.
- All planned checks pass (14 of 14), the three-run results reproduce, and every number in the reports
  traces to a saved file.
- An earlier statement that our AI run was "undertrained" was wrong. It has been retracted and
  corrected in every report.

## Optional next step

Re-tune the classical tool so it stops damaging lesions, then rerun the comparison. Estimated
cost is about $2 to $3 of compute (price not verified). Getting an AI model that removes real hair
would need real hairy photos and far more training images, at roughly $35 to $40 with an uncertain
outcome.

## Licensing

HAM10000 is CC BY-NC 4.0. The DM-AHR repository's README and LICENSE contradict each other (CC BY-NC
4.0 against Apache-2.0), so commercial use is unresolved and needs a rights decision.

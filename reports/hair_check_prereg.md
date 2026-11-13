## Pre-registration for scripts/37_real_hair_check.py (written before the script is run)

Question: does the released DM-AHR checkpoint remove REAL hair, or only the synthetic
overlays it was trained on? Data: 40 train-split images with detector hair coverage of at
least 0.05, chosen by a hash order, so no test image is read. Instrument: the frozen hair
detector at 128 px, with DullRazor (Arm B) as the positive control.

Predictions, fixed in advance:
- Control: DullRazor leaves a median of 30% or less of the detector's hair coverage and
  edits hair-mask pixels much more than the rest (inside/outside ratio above 3). If it does
  not, the detector at 128 px is not a usable instrument and the experiment is void.
- If the released model removes real hair: median fraction remaining 50% or less and an
  inside/outside ratio above 2.
- If it mostly leaves real hair: median fraction remaining 80% or more and a ratio near 1.
- Between 50% and 80%: reported as partial removal.

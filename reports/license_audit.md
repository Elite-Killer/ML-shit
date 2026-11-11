# Licence and provenance audit

No legal conclusions are drawn here. Statements are recorded verbatim from the
pinned sources, and unresolved questions are flagged for the client.

## HAM10000 (ISIC collection 212)

- Source: ISIC Archive collection 212, `https://api.isic-archive.com/api/v2/collections/212/`.
- Licence as stated in the collection description (verbatim): "Data is provided under the CC BY-NC 4.0 license, attribution should be made by referencing the data descriptor manuscript: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018)."
- Per-image `copyright_license` for collection 212 images: CC-BY-NC.
- Metadata and released lesion masks: Harvard Dataverse, DOI `10.7910/DVN/DBW86T`.
- Commercial use is not permitted by the CC BY-NC 4.0 terms as stated. That is a client decision, not one this project can resolve.

## DM-AHR (`AnasHXH/DM-AHR`)

- Pinned commit: `0aee1bbabf788a59be5e1eecddc20f533b5006d8` (2024-01-08), vendored in `third_party/dmahr/` with Python caches removed and no source edits.
- `README.md`, section "License" (verbatim): "This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. You are free to use, share, and adapt this material for non-commercial purposes, as long as you provide attribution to the original author(s) and the source."
- `LICENSE`, opening lines (verbatim): "Apache License / Version 2.0, January 2004 / http://www.apache.org/licenses/"
- The repository contradicts itself: the README states CC BY-NC 4.0, the bundled LICENSE is Apache-2.0. These have materially different redistribution and commercial-use terms.
- Status: UNRESOLVED. Commercial use cannot be assumed from either statement without the authors' or a lawyer's decision.
- Upstream states "The codes are based on SR3" (`Janspiry/Image-Super-Resolution-via-Iterative-Refinement`). SR3's own licence has not been re-verified here and is marked NOT VERIFIED.

## Weights

- Official DM-AHR weights `I725000_E1083_gen.pth` and `_opt.pth` were downloaded from the Google Drive folder linked in the upstream README.
- The download succeeded; the generator SHA-256 is recorded in `artifacts/logs/dmahr_smoke.json` under `checkpoint_sha256`.
- The authors do not document the weights' training data, so their provenance relative to this project's validation and test splits is unknown. They are used for exploratory smoke testing only.
- Confirmatory Arm C uses a checkpoint trained here on train-split sources only; see `artifacts/checkpoints/dmahr/training_manifest.json`.

## Dependencies

- Third-party Python packages retain their own licences.
- torchvision ViT-B/16 `IMAGENET1K_V1` weights follow their published terms. The licence of the ImageNet training data itself is NOT VERIFIED here.

## Practical consequence for the client

HAM10000 is distributed non-commercially. Any commercial deployment of a model trained on it, or redistribution of DM-AHR code or weights while the README/LICENSE conflict is unresolved, needs a rights decision this project does not make.


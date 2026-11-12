"""DERMAHAIR source-pair mapping (blocked: no Kaggle credentials).

The DM-AHR paper's DERMAHAIR dataset is distributed via Kaggle. Kaggle
credentials were never provided on this project, so genuine hair/hairless
dermoscopy pairs could NOT be downloaded. Image-quality scoring instead uses
synthetic hair pairs rendered over hair-absent HAM10000 images (186 held-out
test pairs), which is a documented deviation with wide confidence intervals.
Every result that would have used DERMAHAIR says so explicitly.

This module intentionally provides no silent fallback: importing the mapping
raises, so no pipeline step can mistake synthetic pairs for DERMAHAIR.
"""

from __future__ import annotations

DERMAHAIR_KAGGLE_DATASET = "defolie/dermahir"
DERMAHAIR_AVAILABLE = False
BLOCKED_REASON = "Kaggle credentials unavailable; synthetic-hair fallback used instead"


def fetch_dermahir_pairs(*args, **kwargs):  # noqa: ANN002, ANN003
    raise NotImplementedError(
        f"DERMAHAIR download is blocked on this project: {BLOCKED_REASON}. "
        "Do not substitute synthetic pairs without documenting the deviation."
    )

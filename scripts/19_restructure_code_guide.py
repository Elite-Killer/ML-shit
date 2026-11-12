"""Reorganise docs/CODE_GUIDE.md as data -> methodology -> pipeline -> cells.

Splits the guide on its level-2 headings, inserts the new Data and Methodology
sections, and renumbers everything. Idempotent: refuses to run twice.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "CODE_GUIDE.md"

DATA = """## 2. The data

### What the corpus is

HAM10000 is a public dermoscopy collection: **10,015 images** of skin lesions, each with one of
seven diagnoses, published through the ISIC Archive. The client's collection (ISIC collection 212)
exposes **11,720** images — the 10,015 labelled ones plus 1,705 that carry no diagnosis in this
corpus. Only the labelled set is used; the extras are quarantined and cannot enter a split.

### Verified properties, recomputed rather than assumed

| Property | Value | Why it matters |
|---|---|---|
| Images in collection 212 | 11,720 | the download is complete, paged to exhaustion |
| Labelled HAM10000 cohort | 10,015 | every reported metric comes from this set |
| Quarantined extras | 1,705 | unlabelled; excluded from every split |
| Pixel dimensions | 600 × 450, every image | uniform geometry, so no per-image resize rules needed |
| Colour mode / encoding | RGB / JPEG | no greyscale or palette images to special-case |
| Decode status | 11,720 / 11,720 ok | no corrupt or truncated files |
| Median file size | ~27 KB | consistent with lossy JPEG at this resolution |
| Lesion groups | 7,470 | the unit of splitting and of every statistic |
| Diagnoses | 7 classes | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` |

### The imbalance problem

`nv` (melanocytic naevus) is roughly two thirds of the corpus; `df` and `vasc` number in the tens.
This single fact dictates several later choices: macro-F1 rather than accuracy as the primary
endpoint, class-weighted loss, and a stratified split so the rare classes keep a non-zero test
presence. It also means per-class numbers for `df` (17 test images) and `vasc` (21) are noisy and
must not be read individually.

### The leakage problem

HAM10000 contains several photographs of the same lesion, taken in different sessions. If those
photographs were scattered across training and test, the classifier would be scored on lesions it
had already seen, and the result would measure memorisation. The corpus therefore has to be split
by **lesion**, not by image. A dHash sweep proposed 70,062 candidate duplicate pairs; pixel
confirmation on a common 96×96 grid reduced that to 21 genuine pairs — and all 21 already shared a
lesion identifier, so grouping equals the metadata's own lesion field (7,470 groups). The check's
value is that it rules out *unlinked* duplicates, not that it changes the grouping.

### Reference material the study also needs

| Resource | Status | Use |
|---|---|---|
| HAM10000 lesion masks | available, 10,015 | reference for explanation overlap |
| Released DM-AHR weights | available (Google Drive) | smoke test only; training data undocumented |
| Synthetic hairy/hairless pairs | generated here, 1,022 | the only source of a true hairless reference |
| DERMAHAIR paired data | **blocked** (Kaggle-gated) | would have given real paired references |
| Lama et al. hair masks | **not obtained** | the proposal's independent screening reference |

The last two rows are gaps, not findings. They are recorded as skipped steps in the decision log,
not described as properties of the data.
"""

METHOD = """## 3. Methodology

### The question and the three hypotheses

The study asks whether **generative (diffusion) hair removal** helps a skin-lesion classifier more
than **classical hair removal**, or than leaving the image alone. Three hypotheses were stated
before any result existed:

- **H1 (image quality):** diffusion removal preserves lesion boundaries and texture better than
  classical inpainting. Measured by PSNR and SSIM against a true hairless reference.
- **H2 (classification):** a ViT trained on diffusion-cleaned images outperforms the original and
  classically cleaned arms. Measured by macro-F1 on the locked test set.
- **H3 (explainability):** Grad-CAM and attention maps focus more on the lesion after diffusion
  removal. Measured by lesion overlap and a pointing game against the released masks.

These are hypotheses, not expectations. A null or negative result, honestly reported, is a
successful outcome.

### The experimental design

Four input conditions, everything else held identical:

| Arm | Input the classifier sees | What it isolates |
|---|---|---|
| **A** | the original image | baseline: no hair removal |
| **B** | DullRazor-style morphological detection + Telea inpainting | classical hair removal |
| **C** | DM-AHR conditional diffusion inpainting | generative hair removal |
| **A\\*** | the original passed through a 128 px round-trip | Arm C's resolution change, alone |

Arm A\\* exists because DM-AHR works natively at 128 px while the ViT expects 224. Without it, a
drop in Arm C could be blamed on diffusion artefacts when the resolution round-trip alone caused
it. A\\* is the control that makes that separable.

### Why each design choice was made

| Choice | Reason |
|---|---|
| Split by lesion, frozen and hashed before training | a split chosen after seeing results cannot be trusted |
| Cohort limited to the 10,015 labelled images | the extras have no ground truth to score against |
| Hair threshold frozen before test scoring | otherwise the detector is tuned on the test set |
| Diffusion model trained here, on train/val sources | the released weights have undocumented training data, so leakage cannot be excluded |
| Test set scored once per frozen checkpoint | repeated scoring turns the test set into a tuning set |
| Three seeds (42, 123, 2026) | reduces the chance that one lucky initialisation drives the comparison |
| Identical recipe across arms | the hair-removal condition must be the only difference |

### How the results are judged

Two tests per endpoint, as the protocol specifies. A **lesion-group bootstrap** resamples whole
lesions 10,000 times, sharing the same resample across arms and seeds, giving a percentile 95%
confidence interval. A **Wilcoxon signed-rank** test is run on values collapsed to one per lesion
group, so the several photographs of one lesion are not counted as independent evidence. Holm
correction is applied across the three planned comparisons within each endpoint, separately for each
test. A confidence interval spanning zero is called **inconclusive**, not "no difference".

Effect size matters as much as the p-value: with 1,437 test groups a trivial difference can reach
significance. Results whose standardised effect is negligible are flagged as such.
"""

PIPELINE_INTRO = """## 4. The ML pipeline, phase by phase

The pipeline has thirteen phases. Each one reads files, writes artifacts and logs what it did, so
any later number can be recomputed from disk. Phases 1–3 and 12 are CPU work; 4–11 need a GPU.

Three properties hold throughout. Nothing is skipped silently — every excluded image is counted and
written to a CSV with a reason. Long jobs are resumable, so a preempted container continues rather
than restarting. And no phase may consume a later phase's output, which is what keeps the test set
out of every design decision.
"""

NOTEBOOKS = """## 7. The notebooks, cell by cell

Three notebooks ship with the repository. Each is annotated: a markdown cell above every code cell
states what that cell does and why it exists. This section is the written index.

| Notebook | Purpose | Needs |
|---|---|---|
| `01_results_walkthrough.ipynb` | reads the committed artifacts and explains the results | CPU only, no Modal, no download |
| `02_pipeline_reproduction.ipynb` | runs every phase on Modal | Modal token |
| `03_local_gpu_pipeline.ipynb` | runs every phase natively on a local or Colab GPU | a CUDA GPU, no Modal |

### `01_results_walkthrough.ipynb` — presenting the study

| Cell | What it does |
|---|---|
| Setup | resolves the artifact directories and defines JSON-loading helpers |
| Cohort accounting | rebuilds the headline counts from the audit artifact |
| Split sizes | prints per-class split counts, showing the stratification worked |
| Leakage check | independently recomputes that no lesion group spans two splits |
| Hair screening | reports how many images were marked hair-present |
| Arm definitions | a reference table of what each arm is |
| Classification results | macro-F1, accuracy, balanced accuracy and AUC per arm |
| Result figures | the same numbers as bars with error bars |
| Per-class detail | where the arms differ by diagnosis |
| Image quality | PSNR/SSIM against hairless references, with the hairy input as a floor |
| Explanation metrics | overlap and pointing per arm with both chance baselines |
| Explanation figure | saliency against the uniform-chance line |
| Comparison table | every endpoint with both tests and Holm correction |
| Significance summary | the decision a reader cares about, per comparison |
| Hypothesis verdicts | maps H1/H2/H3 onto the comparisons that bear on them |
| Visual panels | the contact sheets written during processing |
| Acceptance gates | the project's own automated checks |

### `03_local_gpu_pipeline.ipynb` — running it yourself

| Cell | What it does |
|---|---|
| Environment check | reports the GPU and torch build before anything assumes one |
| Dependencies | installs the pinned versions the results were produced with |
| Get the code | clones the repository and enters it |
| Where the data lives | sets `DATA_ROOT` and `ARTIFACT_ROOT`; in Colab these point at Drive |
| Script runner | a wrapper that invokes each phase with those variables set |
| Download + verify | fetches all 11,720 images and masks, then reads the report back |
| Audit + split, and verify | rebuilds the frozen split and confirms the counts and leakage check |
| Hair pipeline | screening, Arm B, Arm A\\* and the synthetic pairs |
| Released checkpoint | fetches the DM-AHR authors' weights, needed for the smoke test |
| Smoke test | confirms the vendored upstream code runs and a batch round-trips deterministically |
| Benchmark | measures sampling throughput, to size the Arm C run |
| Diffusion sanity + training | overfits a few pairs, then trains on train/val sources only |
| Arm C, one shard | generates one eighth of the images and reports the timing |
| Arm C, remaining shards | finishes the rest; safe to re-run because finished images are skipped |
| Arm C, held-out pairs | generates the pairs that make PSNR/SSIM measurable |
| Passthrough | copies hair-absent images into Arm C, where they are identical |
| Arm inventory | counts what exists on disk before training |
| One ViT run, then all | confirms the recipe works, then trains the remaining eleven |
| Evaluation | scores each frozen checkpoint once on the test set |
| Image quality, XAI | PSNR/SSIM, then Grad-CAM and attention roll-out with baselines |
| Statistics, report, gates | the comparison table, generated report, inventory and acceptance check |
| Outputs | counts the files produced, and how to explore them |

`02_pipeline_reproduction.ipynb` follows the same phases, but each cell shells out to `modal run`.
It needs no local GPU; the runtime selected in Colab is irrelevant to it.
"""


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split on level-2 headings, returning the preamble and (heading, body) pairs."""
    parts = re.split(r"^## ", text, flags=re.MULTILINE)
    preamble = parts[0]
    sections = []
    for chunk in parts[1:]:
        lines = chunk.splitlines()
        sections.append((lines[0].strip(), "\n".join(lines[1:])))
    return preamble, sections


def main() -> int:
    text = GUIDE.read_text()
    if "## 2. The data" in text:
        print("already restructured; nothing to do")
        return 0

    preamble, sections = split_sections(text)
    by_title = {t: b for t, b in sections}

    def find(fragment: str) -> str:
        for t in by_title:
            if fragment.lower() in t.lower():
                return t
        raise KeyError(f"no section matching {fragment!r}")

    # (new number and title, existing section to reuse, or a full new block)
    order = [
        ("1. The study in one page", find("study in one page"), None),
        ("2. The data", None, DATA),
        ("3. Methodology", None, METHOD),
        ("4. The ML pipeline, phase by phase", find("pipeline, phase by phase"),
         PIPELINE_INTRO),
        ("5. What was actually used", find("what was actually used"), None),
        ("6. Repository map", find("repository map"), None),
        ("7. The notebooks, cell by cell", None, NOTEBOOKS),
        ("8. Where each reported number comes from",
         find("where each reported number"), None),
        ("9. Design decisions, and the reason for each", find("design decisions"), None),
        ("10. Deviations from the proposal", find("deviations from the proposal"), None),
        ("11. Glossary", find("glossary"), None),
        ("12. Questions a supervisor is likely to ask",
         find("questions a supervisor"), None),
        ("13. What is deliberately not claimed", find("deliberately not claimed"), None),
    ]

    def strip_heading(block: str) -> str:
        """Remove a leading '## ...' line from a new-section block."""
        lines = block.splitlines()
        if lines and lines[0].startswith("## "):
            lines = lines[1:]
        return "\n".join(lines).strip()

    out = [preamble.rstrip() + "\n"]
    for title, existing, new_block in order:
        if new_block is not None:
            body = strip_heading(new_block)
            if existing is not None:
                body = body + "\n\n" + by_title[existing].strip()
        else:
            body = by_title[existing].strip()
        out.append(f"## {title}\n{body}\n")

    GUIDES_BACKUP = GUIDE.with_suffix(".md.bak")
    GUIDES_BACKUP.write_text(text)
    GUIDE.write_text("\n".join(out))
    print(f"restructured {GUIDE.name}: {len(order)} sections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

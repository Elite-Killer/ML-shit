"""Insert an explanatory markdown cell above every code cell.

Run once. Keys match on a distinctive substring of the cell source, so the
notebook can be regenerated or reordered without silently annotating the wrong
cell: an unmatched code cell aborts the run rather than being skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


NB01 = {
    "import json\nfrom pathlib import Path":
        """**What this does.** Resolves the repository, artifact and manifest directories, then
defines two helpers: `jload` for reading JSON artifacts and `show` for tabular output.

Nothing is downloaded and no GPU is touched — every later cell reads files that the pipeline
already produced. That is the point of this notebook: the numbers can be inspected without
re-running anything.""",
    "audit = jload(LOGS / \"data_audit.json\")":
        """**Cohort accounting.** Rebuilds the headline counts from `data_audit.json` rather than
quoting them: how many images collection 212 holds, how many carry a HAM10000 label, how many
were quarantined, and how many lesion groups the deduplication produced.""",
    "fs = audit.get(\"frozen_splits\"":
        """**Split sizes.** Prints the frozen train/validation/test counts per class. Every class
should hold roughly 15% of its images in each split — that is what "class-stratified" means
here, and it is why the rare classes still have a non-zero test presence.""",
    "# Independent leakage check":
        """**Leakage check, recomputed.** Grouping the split table by lesion and counting how many
distinct partitions each group appears in. The expected answer is exactly one; anything above
that would mean one patient's lesion sat in two splits and the test score was inflated.""",
    "if (MAN / \"hair_screening.csv\").exists()":
        """**Hair screening.** Reports how many images the frozen detector marked hair-present,
overall and per split. This number drives the cost of the whole study, because only hair-present
images are processed in Arms B and C.""",
    "arms = pd.DataFrame([":
        """**Arm definitions.** A short reference table of what each input condition actually is,
so a reader does not have to hold the mapping in their head while reading the results.""",
    "seed = pd.read_csv(LOGS / \"classification_seed_summary.csv\")":
        """**Headline classification results.** Macro-F1, accuracy, balanced accuracy and macro AUC
per arm, averaged over the three seeds. Macro-F1 is the primary endpoint because the classes are
imbalanced and plain accuracy is dominated by `nv`.""",
    "import matplotlib.pyplot as plt":
        """**Result figures.** The same numbers as a bar chart with standard-deviation error bars,
which makes the gap between arms immediately legible.""",
    "per_class = pd.read_csv(LOGS / \"classification_per_class.csv\")":
        """**Per-class detail.** Shows where the arms differ by diagnosis, not just in aggregate.
Small-support classes (`df`, `vasc`) are noisy and are not interpreted individually.""",
    "pq = pd.read_csv(LOGS / \"paired_psnr_ssim_test.csv\")":
        """**Image quality.** PSNR and SSIM against genuine hairless references on the held-out
synthetic pairs, with the unprocessed hairy input as a floor. A method that scores below its own
input has made that image worse than doing nothing.""",
    "import glob\n\nxai = pd.concat":
        """**Explanation metrics.** Aggregates Grad-CAM and attention-roll-out overlap and pointing
accuracy per arm, alongside the two chance baselines. The baselines are the interpretable part:
without them an overlap number means very little.""",
    "fig, ax = plt.subplots(figsize=(7, 3.2))":
        """**Explanation figure.** Both saliency methods against the uniform-chance line, so a
reader can see at a glance whether either method clears a trivial baseline.""",
    "stats = pd.read_csv(LOGS / \"statistical_comparisons.csv\")":
        """**The full comparison table.** Every endpoint, every pairwise comparison, with both the
bootstrap confidence interval and the Wilcoxon p-value, each Holm-corrected.""",
    "# Compact significance readout":
        """**Readable significance summary.** Collapses the table to the decision a reader cares
about: does the confidence interval exclude zero, and what is the corrected p-value.""",
    "def row(endpoint, comparison, subset=None)":
        """**Hypothesis verdicts.** Maps the three pre-stated hypotheses onto the specific measured
comparisons that bear on them, in plain language, and names the resolution control.""",
    "from IPython.display import Image, display":
        """**Visual panels.** The contact sheets the pipeline wrote during processing, so the
qualitative behaviour can be seen next to the numbers."""
        ,
    "acc = jload(LOGS / \"acceptance_report.json\")":
        """**Acceptance gates.** The project's own automated checks, including any item that is
blocked rather than passed, with its evidence path.""",
}

NB03 = {
    "print(subprocess.run([\"nvidia-smi\"":
        """**Environment check.** Confirms what hardware this notebook is actually running on before
anything assumes a GPU exists.""",
    "-m\", \"pip\", \"install\", \"-q\", \"-r\", \"requirements.txt\"":
        """**Dependencies.** Installs the pinned versions the reported results were produced with.
On Colab this replaces the preinstalled torch — intended, because a different torch version is a
different experiment.""",
    "REPO_URL = ":
        """**Get the code.** Clones the repository at the branch containing the pipeline, and enters
it. Skipped if you are already inside the repository.""",
    "IN_COLAB = ":
        """**Where the data lives.** Sets `DATA_ROOT` and `ARTIFACT_ROOT`. In Colab these are
pointed at Drive so a long Arm C run survives the session ending; locally they sit inside the
repository.""",
    "def run(script":
        """**Script runner.** A thin wrapper that invokes a pipeline script with those two
environment variables set. Every later cell goes through this, so no cell needs to repeat the
environment handling.""",
    "run(\"00_download_data.py\")":
        """**Phase 1 — download.** Fetches all 11,720 images of ISIC collection 212 plus the
HAM10000 metadata and segmentations. All public: no account or key required.""",
    "rep = json.loads((Path(os.environ[\"ARTIFACT_ROOT\"])":
        """**Phase 1 — verify.** Reads back the download report so the counts are seen, not
assumed.""",
    "run(\"01_audit_and_splits.py\")":
        """**Phase 2 — audit and split.** Verifies every image decodes, separates the labelled
cohort from the extras, merges confirmed near-duplicates, and freezes the lesion-grouped
70/15/15 split. CPU only, a couple of minutes.""",
    "audit = json.loads((log / \"data_audit.json\")":
        """**Phase 2 — verify.** Confirms the cohort, group count and leakage check directly from
the audit artifact. From a fresh download this reproduces the reference split hash exactly.""",
    "run(\"02_hair_pipeline.py\")":
        """**Phase 3 — hair screening, Arm B and synthetic pairs.** Detects hair, produces the
classical removal, the 128 px control round-trip and the synthetic hairy/hairless pairs. The
detector threshold is frozen here, before any test metric exists.""",
    "weights_dir = Path(os.environ[\"DATA_ROOT\"])":
        """**Phase 4 — released checkpoint.** Fetches the DM-AHR authors' published weights. They
are needed for the smoke test, which is the only external check on the adapter.""",
    "run(\"03_dmahr_smoke.py\")":
        """**Phase 4 — smoke test.** Verifies the vendored upstream code runs, that two documented
upstream defects are worked around correctly, and that a batch round-trips deterministically.""",
    "run(\"03b_dmahr_bench.py\"":
        """**Phase 4 — benchmark.** Measures sampling throughput on this GPU. Use it to decide
whether the full Arm C run fits your time and budget before committing to it.""",
    "run(\"04_dmahr_train.py\", \"--overfit\"":
        """**Phase 5 — sanity pass.** Overfits a handful of pairs to confirm the loss actually
descends before spending hours on a real run.""",
    "run(\"04_dmahr_train.py\")":
        """**Phase 5 — diffusion training.** Trains DM-AHR on train-split sources only, selecting on
validation. Checkpoints as it goes, so an interrupted run resumes.""",
    "t0 = time.time()\nrun(\"05_dmahr_sample.py\"":
        """**Phase 6 — Arm C, one shard.** Generates hair-removed images for one eighth of the
hair-present images and reports how long it took. Run this before the rest: it is the only honest
way to know whether the full run fits.""",
    "for shard in range(1, 8)":
        """**Phase 6 — Arm C, remaining shards.** Processes the rest. Safe to re-run: images already
written are skipped, so a killed session resumes rather than restarting.""",
    "run(\"05_dmahr_sample.py\", \"--pairs-split\"":
        """**Phase 6 — held-out pairs.** Generates Arm C output for the synthetic test pairs, which
is what makes PSNR and SSIM measurable.""",
    "run(\"05b_fill_c_passthrough.py\")":
        """**Phase 6 — passthrough.** Copies hair-absent images into Arm C. They are identical in
every arm, and without this the Arm C dataset is incomplete.""",
    "for arm in (\"A\", \"B\", \"C\", \"Astar\"):\n    d = processed / arm":
        """**Phase 7 — arm inventory.** Counts what exists on disk per arm, so a missing directory
is caught before training rather than during it.""",
    "run(\"06_train_vit.py\", \"--arm\", \"A\", \"--seed\", 42)":
        """**Phase 8 — one ViT run.** Trains a single arm/seed pair to confirm the recipe works on
this machine and data before launching twelve.""",
    "for arm in [\"A\", \"Astar\", \"B\", \"C\"]:\n    for seed in [42, 123, 2026]:\n        if (arm":
        """**Phase 8 — all ViT runs.** The remaining eleven arm/seed combinations, identical recipe.
Each writes its own checkpoint directory with config, seed, class weights and git commit.""",
    "run(\"07_eval_vit.py\"":
        """**Phase 9 — locked-test evaluation.** Scores each frozen checkpoint once on the test set,
emitting per-image predictions so every downstream statistic can be recomputed.""",
    "run(\"07b_image_quality.py\"":
        """**Phase 10 — image quality.** PSNR and SSIM for each arm against the hairless references,
with the hairy input as the floor.""",
    "run(\"08_xai.py\"":
        """**Phase 10 — explainability.** Grad-CAM and attention roll-out against the lesion masks,
with chance baselines computed per image.""",
    "run(\"09_stats.py\")":
        """**Phase 11 — statistics, report, gates.** Bootstrap plus Wilcoxon with Holm correction;
the report generated from artifacts; the hashed inventory; and the acceptance check.""",
    "stats = pd.read_csv(logs / \"statistical_comparisons.csv\")":
        """**Phase 11 — result readout.** Prints the comparison table from the run you just
produced, so the numbers on screen are the ones this machine generated.""",
    "root = Path(os.environ[\"ARTIFACT_ROOT\"])":
        """**Where the outputs are.** Counts files per artifact directory. Point
`01_results_walkthrough.ipynb` at the same tree to explore the results interactively.""",
}

NB02 = {
    "print(\"MODAL_API_TOKEN set:\"":
        """**Credential check.** Reports whether a Modal token is present in the environment,
without ever printing its value.""",
    "\"-m\", \"pip\", \"install\", \"-q\", \"modal\"":
        """**Install the Modal client.** Pins nothing, so check the printed version — the app uses
the v1 API names.""",
    "REPO_URL = ":
        """**Get the code.** Clones the pipeline branch and installs its pinned requirements.""",
    "!modal profile current":
        """**Confirm access.** Shows the active Modal profile and lists the volume, so a credential
or naming problem surfaces here rather than mid-run.""",
    "!modal run modal_app.py::gpu_probe":
        """**Verify the GPU from inside Modal.** The hardware is established by running code on
Modal, not by trusting the config string, and the result is written to
`artifacts/logs/environment_report.json`.""",
    "!modal run modal_app.py::download_data":
        """**Phase 1 — download.** Fetches collection 212 and the HAM10000 metadata and masks into
the volume. Raw data is immutable and never enters Git.""",
    "!modal run modal_app.py::build_manifests":
        """**Phase 2 — audit, manifests and frozen split.** Recomputes every count from the files,
merges confirmed duplicates, and proves zero group overlap before hashing the split.""",
    "!modal run modal_app.py::hair_pipeline":
        """**Phase 3 — screening, Arm B and synthetic pairs.** The detector threshold is frozen
here, before any test metric exists.""",
    "!modal run modal_app.py::dmahr_smoke":
        """**Phase 4 — audit and smoke test.** Confirms the pinned upstream commit, parses its
config, loads the released checkpoint and checks a batch is finite and reproducible.""",
    "!modal run --detach modal_app.py::dmahr_train":
        """**Phase 5 — diffusion training.** Leakage-controlled: train-split sources only, model
selection on validation. Detached so it survives a dropped connection.""",
    "!modal run --detach modal_app.py::dmahr_sample_driver_ablation":
        """**Phase 6 — Arm C generation.** Sharded across containers, each shard resuming from what
is already written. This is the expensive phase.""",
    "!modal run modal_app.py::fill_c_passthrough":
        """**Phase 6 — passthrough.** Copies hair-absent images into Arm C, where they are
identical to the originals.""",
    "print(f\"launching arm {arm} seed {seed}\")":
        """**Phase 7 — ViT training.** Four arms by three seeds, identical recipe, each detached and
resumable.""",
    "!modal run modal_app.py::vit_eval":
        """**Phase 8 — locked-test evaluation.** Per-image predictions and metrics, scored once per
frozen checkpoint.""",
    "!modal run modal_app.py::image_quality":
        """**Phase 9 — paired image quality.** PSNR and SSIM against the synthetic hairless
references.""",
    "!modal run --detach modal_app.py::xai_run":
        """**Phase 9 — explainability.** Grad-CAM and attention roll-out with lesion-overlap,
pointing game and chance baselines.""",
    "!modal run modal_app.py::stats_run":
        """**Phase 10 — statistics, report and gates.** Bootstrap and Wilcoxon with Holm
correction, then the report, the hashed inventory and the acceptance check.""",
    "!mkdir -p artifacts/logs":
        """**Retrieve results.** Pulls the small artifacts from the volume into the notebook
runtime; checkpoints and processed images stay on the volume by design.""",
    "!modal volume get dmahr-vit-vol /data/manifests":
        """**Retrieve the split manifests.** They are git-ignored, so they are fetched separately
when a split-level analysis is needed.""",
}


def annotate(path: Path, table: dict[str, str]) -> int:
    nb = nbformat.read(path, as_version=4)
    out: list = []
    used: set[str] = set()
    inserted = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            out.append(cell)
            continue
        match = None
        for key in table:
            if key in cell.source and key not in used:
                match = key
                break
        if match is None:
            print(f"  UNMATCHED code cell in {path.name}: {cell.source[:70]!r}",
                  file=sys.stderr)
            out.append(cell)
            continue
        used.add(match)
        out.append(md(table[match]))
        out.append(cell)
        inserted += 1
    nb.cells = out
    nbformat.write(nb, path)
    return inserted


def main() -> int:
    total = 0
    for name, table in (("01_results_walkthrough.ipynb", NB01),
                        ("02_pipeline_reproduction.ipynb", NB02),
                        ("03_local_gpu_pipeline.ipynb", NB03)):
        n = annotate(HERE / name, table)
        print(f"{name}: annotated {n} code cells")
        total += n
    print("total", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

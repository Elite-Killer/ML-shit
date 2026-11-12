"""Exploratory statistics for the released-checkpoint arm (Creleased).

The pre-registered comparison table in scripts/09_stats.py is frozen: this
script adds A-vs-Creleased and B-vs-Creleased rows in a SEPARATE table so the
committed protocol and its Holm family are not modified post hoc.

Everything about the Creleased arm is exploratory: it uses the authors'
released DM-AHR checkpoint (not our trained one) with a 25-step respaced
sampling schedule (the paper's schedule is 2,000 steps), and its outputs are
128x128. No hypothesis in the protocol covers it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir  # noqa: E402
from dm_ahr_vit.stats.bootstrap import holm  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()


def load_stats_module():
    spec = importlib.util.spec_from_file_location(
        "stats09", Path(__file__).resolve().parents[1] / "scripts" / "09_stats.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    protocol = load_config("configs/protocol.json")
    seeds = list(protocol["seeds"])
    reps = int(protocol["bootstrap_replicates"])
    bseed = int(protocol["bootstrap_seed"])

    stats09 = load_stats_module()
    # Runtime-only override: 09_stats.py itself and its frozen outputs are
    # untouched; this process sees only the arms the exploratory table needs.
    stats09.ARMS = ["A", "B", "Creleased"]
    comparisons = [("A", "Creleased"), ("B", "Creleased")]

    out: list = []
    with timed("stats_exploratory", log) as t:
        for subset in ("full", "hair_present"):
            out += stats09.classification_endpoint(
                seeds, subset, reps, bseed, comparisons
            )

        p = artifact_dir("logs") / "paired_psnr_ssim_test_Creleased.csv"
        if p.exists():
            df = pd.read_csv(p)
            for metric in ("psnr", "ssim"):
                out += stats09.paired_column_endpoint(
                    df, f"Creleased_{metric}", f"B_{metric}",
                    f"paired_{metric}", reps, bseed, [("Creleased", "B")]
                )
                out += stats09.paired_column_endpoint(
                    df, f"Creleased_{metric}", f"hairy_input_{metric}",
                    f"paired_{metric}", reps, bseed, [("Creleased", "hairy input")]
                )

    if not out:
        log.error("no exploratory comparisons could be computed")
        return 1

    table = pd.DataFrame(out)
    table["family"] = "exploratory_Creleased"
    table["p_holm"] = np.nan
    for _, grp in table.groupby(["endpoint", "subset"]):
        table.loc[grp.index, "p_holm"] = holm(grp["p_raw"].tolist())
    cols = ["endpoint", "subset", "family", "comparison", "arm_a", "arm_b", "arm_a_mean",
            "arm_b_mean", "difference", "ci_low", "ci_high", "p_raw", "p_holm",
            "effect_size_dz", "wilcoxon_p_raw", "wilcoxon_p_holm", "wilcoxon_statistic",
            "wilcoxon_n_groups", "n_images", "n_groups", "replicates", "test"]
    table = table[[c for c in cols if c in table.columns]]
    table = table.sort_values(["endpoint", "subset", "comparison"])
    table["negligible"] = (
        (table["p_holm"] < 0.05)
        & (table["effect_size_dz"].abs() < 0.2)
    )
    table.to_csv(
        artifact_dir("logs") / "statistical_comparisons_exploratory_Creleased.csv",
        index=False,
    )
    save_json(
        {"seconds": t["seconds"], "rows": int(len(table)),
         "endpoints": sorted(table["endpoint"].unique().tolist()),
         "arm": "Creleased",
         "arm_description": ("released DM-AHR checkpoint, 25-step respaced sampling, "
                             "exploratory (not pre-registered)")},
        artifact_dir("logs") / "stats_summary_exploratory_Creleased.json",
    )
    record_run("stats_exploratory", gpu=None, seconds=t["seconds"], rows=int(len(table)))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

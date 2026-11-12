"""Independently recompute every claim in the research report.

Reads the source artifacts and checks the report's numbers against them, so a
stale or mis-stated figure is caught rather than trusted.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = (ROOT / "reports" / "final_research_report.md").read_text()
LOGS = ROOT / "artifacts" / "logs"
MAN = ROOT / "data" / "manifests"

fails: list[str] = []


def chk(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def main() -> int:
    audit = json.loads((LOGS / "data_audit.json").read_text())
    split = pd.read_csv(MAN / "splits.csv")
    hair = pd.read_csv(MAN / "hair_screening.csv")
    pairs = pd.read_csv(MAN / "synthetic_pairs.csv")
    metr = pd.read_csv(LOGS / "classification_seed_summary.csv")
    stats = pd.read_csv(LOGS / "statistical_comparisons.csv")
    qual = pd.read_csv(LOGS / "paired_psnr_ssim_test.csv")
    per = pd.read_csv(LOGS / "classification_per_class.csv")
    neardup = pd.read_csv(MAN / "near_duplicates_confirmed.csv")

    # cohort
    chk("collection 11720", audit["collection_total"] == 11720 and "11720" in REP)
    chk("c0 10015", audit["c0_decoded"] == 10015 and "10015" in REP)
    chk("extras 1705", audit["c0_extra"] == 1705 and "1705" in REP)
    chk("groups 7470", audit["groups_total"] == 7470 and "7470" in REP)
    chk("split sizes", split["split"].value_counts().to_dict()
        == {"train": 7011, "test": 1502, "val": 1502})
    chk("split hash frozen", audit["frozen_splits"]["splits_sha256"].startswith("fda56cf3"))
    chk("no group spans splits", split.groupby("group_id")["split"].nunique().max() == 1)

    # duplicate merge: does it change grouping?
    conf = neardup[neardup["confirmed"]]
    lesion_of = dict(zip(split["image_id"], split["lesion_id"]))
    same = sum(1 for a, b in zip(conf["id_a"], conf["id_b"])
               if lesion_of.get(a) == lesion_of.get(b))
    chk("21 confirmed pairs", len(conf) == 21)
    chk("all pairs already same lesion", same == 21, f"{same}/21")
    chk("groups == unique lesion_id",
        split["group_id"].nunique() == split["lesion_id"].nunique() == 7470)

    # hair
    chk("8993 hair present", int(hair["hair_present"].sum()) == 8993 and "8993" in REP)
    chk("89.8% rate", abs(hair["hair_present"].mean() - 0.898) < 0.001)
    chk("1022 pairs", len(pairs) == 1022)
    chk("186 test pairs", int((pairs["split"] == "test").sum()) == 186 and len(qual) == 186)

    # classification
    full = metr[metr["subset"] == "full"].set_index("arm")
    for arm, want in [("A", 0.750), ("Astar", 0.732), ("B", 0.698), ("C", 0.426)]:
        got = float(full.loc[arm, "macro_f1_mean"])
        chk(f"macro-F1 {arm}={want}", abs(round(got, 3) - want) < 0.001, f"{got:.4f}")
    for arm in ["A", "Astar", "B", "C"]:
        sub = per[per["arm"] == arm]
        chk(f"support sums 1502 ({arm})", int(sub["support"].sum()) == 1502,
            str(int(sub["support"].sum())))

    # statistics
    ab = stats[(stats["endpoint"] == "test_macro_f1") & (stats["subset"] == "full")
               & (stats["comparison"] == "A-B")].iloc[0]
    chk("A-B diff 0.052", abs(float(ab["difference"]) - 0.0523) < 0.001)
    chk("A-B Holm p 0.0012", abs(float(ab["p_holm"]) - 0.0012) < 0.0001)
    chk("all rows have Wilcoxon", bool(stats["wilcoxon_p_raw"].notna().all()),
        f"{len(stats)} rows")
    chk("A-Astar is secondary family",
        bool(stats[stats["comparison"] == "A-Astar"]["family"].eq("secondary_control").all()))
    planned = stats[stats["family"] == "planned"].groupby(["endpoint", "subset"]).size()
    chk("planned family = 3 per endpoint/subset", bool((planned == 3).all()))

    # image quality
    for col, want in [("hairy_input_psnr", 22.127), ("B_psnr", 35.732), ("C_psnr", 16.312),
                      ("hairy_input_ssim", 0.520), ("B_ssim", 0.918), ("C_ssim", 0.130)]:
        got = float(qual[col].mean())
        chk(f"{col}~{want}", abs(round(got, 3) - want) < 0.002, f"{got:.3f}")
    chk("C ssim sd ~0.010", abs(float(qual["C_ssim"].std()) - 0.010) < 0.002,
        f"{qual['C_ssim'].std():.4f}")
    chk("185 of 186 below 0.15", int((qual["C_ssim"] < 0.15).sum()) == 185,
        str(int((qual["C_ssim"] < 0.15).sum())))

    # centre prior
    xai = pd.concat([pd.read_csv(f) for f in
                     sorted(glob.glob(str(LOGS / "xai_localization_*.csv")))])
    g_ov = float(xai["chance_gaussian_overlap20"].mean())
    g_pt = float(xai["chance_gaussian_pointing"].mean())
    by_arm = xai.groupby("arm")[["gradcam_overlap20", "rollout_overlap20",
                                 "gradcam_pointing", "rollout_pointing"]].mean()
    best_ov = float(by_arm[["gradcam_overlap20", "rollout_overlap20"]].max().max())
    best_pt = float(by_arm[["gradcam_pointing", "rollout_pointing"]].max().max())
    chk("gaussian overlap 0.644", abs(g_ov - 0.644) < 0.002, f"{g_ov:.3f}")
    chk("gaussian pointing 0.953", abs(g_pt - 0.953) < 0.002, f"{g_pt:.3f}")
    chk("no arm beats gaussian overlap", best_ov < g_ov, f"{best_ov:.3f} vs {g_ov:.3f}")
    chk("no arm beats gaussian pointing", best_pt < g_pt, f"{best_pt:.3f} vs {g_pt:.3f}")

    # stale claims must be gone
    for stale in ("does not state they were expert-traced",
                  "No independent hair annotations exist",
                  "part of that margin"):
        chk(f"stale claim removed: {stale[:34]}", stale not in REP)

    print()
    if fails:
        print(f"{len(fails)} FAILURES: {fails}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

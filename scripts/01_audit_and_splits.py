"""Audit the collection, build the manifest, confirm near-duplicates, freeze splits.

Stages are cached so re-runs do not re-read the image volume:
  data_manifest.csv -> near_duplicates.csv -> near_duplicates_confirmed.csv -> splits.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import (  # noqa: E402
    CLASSES,
    EXPECTED_C0_COUNTS,
    config_hash_file,
    load_config,
    save_json,
)
from dm_ahr_vit.data import ham10000, manifest as manifest_mod, masks, splits as splits_mod  # noqa: E402
from dm_ahr_vit.paths import (  # noqa: E402
    artifact_dir,
    ham_masks_dir,
    ham_root,
    isic_images_dir,
    manifest_dir,
    raw_dir,
)
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()


def load_or_build_manifest(records_path: Path, metadata: pd.DataFrame, force: bool) -> pd.DataFrame:
    out = manifest_dir() / "data_manifest.csv"
    n_records = len([ln for ln in records_path.read_text().splitlines() if ln.strip()])
    if out.exists() and not force:
        df = pd.read_csv(out)
        if len(df) == n_records and df["sha256"].notna().all() and df["dhash"].notna().all():
            log.info("reusing cached manifest (%d rows)", len(df))
            return df
    man, _, _ = manifest_mod.build_manifest(
        records_path, isic_images_dir(), metadata, manifest_dir()
    )
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-manifest", action="store_true")
    ap.add_argument("--force-confirm", action="store_true")
    args = ap.parse_args()

    protocol = load_config("configs/protocol.json")
    protocol_hash = config_hash_file("configs/protocol.json")

    with timed("audit_and_splits", log) as t:
        metadata = ham10000.read_metadata(ham_root() / "HAM10000_metadata.tab")
        comparison = ham10000.expected_comparison(metadata)
        records_path = raw_dir() / "isic_212" / "records.jsonl"

        man = load_or_build_manifest(records_path, metadata, args.force_manifest)
        excl = pd.read_csv(manifest_dir() / "exclusions.csv") if (
            manifest_dir() / "exclusions.csv"
        ).exists() else pd.DataFrame(columns=["image_id", "cohort", "reason", "path"])

        c0 = man[(man["cohort"] == "C0") & (man["decode_status"] == "ok")].copy()
        c0["dx"] = c0["dx"].astype(str)
        c0["lesion_id"] = c0["lesion_id"].astype(str)
        log.info("C0 decoded images: %d", len(c0))

        # dHash candidates come straight from the manifest, no extra file reads.
        cand_path = manifest_dir() / "near_duplicates.csv"
        if cand_path.exists():
            candidates = pd.read_csv(cand_path)
        else:
            hashes = c0["dhash"].astype(np.uint64).to_numpy()
            candidates = pd.DataFrame(
                manifest_mod.hamming_pairs(hashes, 6, c0["image_id"].tolist()),
                columns=["id_a", "id_b", "distance"],
            )
            candidates.to_csv(cand_path, index=False)
        log.info("dHash candidates: %d", len(candidates))

        conf_path = manifest_dir() / "near_duplicates_confirmed.csv"
        if conf_path.exists() and not args.force_confirm:
            confirmed = pd.read_csv(conf_path)
            log.info("reusing cached confirmation (%d rows)", len(confirmed))
        else:
            with timed("confirm_near_duplicates", log):
                confirmed = manifest_mod.confirm_near_duplicates(
                    candidates, man, max_mean_abs_diff=4.0, min_correlation=0.985
                )
            if len(confirmed):
                confirmed.to_csv(conf_path, index=False)

        kept = confirmed[confirmed["confirmed"]] if len(confirmed) else confirmed
        log.info("confirmed near-duplicate pairs: %d of %d", len(kept), len(candidates))

        grouped = splits_mod.build_groups(c0, kept if len(kept) else candidates.iloc[0:0])
        class_purity = grouped.groupby("group_id")["dx"].nunique()
        mixed_groups = int((class_purity > 1).sum())

        summary = splits_mod.freeze_splits(
            grouped,
            seed=int(protocol["split_seed"]),
            fractions=protocol["split_fractions"],
            out_dir=manifest_dir(),
            protocol_hash=protocol_hash,
        )
        split_df = pd.read_csv(manifest_dir() / "splits.csv")
        verification = splits_mod.verify_splits(split_df)
        x1 = masks.build_x1_manifest(
            ham_masks_dir(), split_df, man, manifest_dir() / "mask_manifest_x1.csv"
        )

    fractions = protocol["split_fractions"]
    balance = {
        c: {
            s: round(summary["counts"][s][c] / max(sum(summary["counts"][x][c] for x in fractions), 1), 4)
            for s in fractions
        }
        for c in CLASSES
    }
    audit = {
        "collection_total": int(len(man)),
        "collection_expected": protocol["expected_collection_total"],
        "c0_decoded": int(len(c0)),
        "c0_extra": int((man["cohort"] == "C0-extra").sum()),
        "exclusions": int(len(excl)),
        "exclusion_reasons": excl["reason"].value_counts().to_dict() if len(excl) else {},
        "metadata_comparison": comparison,
        "dhash_candidates": int(len(candidates)),
        "near_duplicates_confirmed": int(len(kept)),
        "mixed_class_groups": mixed_groups,
        "groups_total": int(grouped["group_id"].nunique()),
        "frozen_splits": summary,
        "split_verification": verification,
        "class_fraction_balance": balance,
        "target_fractions": fractions,
        "x1_masks": {
            "test_rows": int((x1["split"] == "test").sum()),
            "test_with_mask": int(((x1["split"] == "test") & (x1["has_mask"])).sum()),
            "test_ok": int(((x1["split"] == "test") & (x1["status"] == "ok")).sum()),
            "all_ok": int((x1["status"] == "ok").sum()),
            "all_present": int(x1["has_mask"].sum()),
            "size_mismatches": int((x1["size_matches"] == False).sum()),  # noqa: E712
        },
        "protocol_hash": protocol_hash,
        "seconds": t["seconds"],
    }
    save_json(audit, artifact_dir("logs") / "data_audit.json")
    record_run("audit_and_splits", gpu=None, seconds=t["seconds"],
               images=audit["collection_total"], c0=audit["c0_decoded"])

    lines = [
        "# Data audit (recomputed from files)",
        "",
        f"- Collection 212 records: **{audit['collection_total']}** (expected {audit['collection_expected']})",
        f"- C0 decoded images: **{audit['c0_decoded']}** (expected {protocol['expected_c0_total']})",
        f"- C0-extra quarantined: **{audit['c0_extra']}**",
        f"- Exclusions: {audit['exclusions']} ({audit['exclusion_reasons']})",
        f"- dHash candidates (distance <= 6): {audit['dhash_candidates']}; "
        f"pixel-confirmed near-duplicates: **{audit['near_duplicates_confirmed']}**",
        f"- Groups after merging lesion + confirmed near-duplicates: {audit['groups_total']} "
        f"({audit['mixed_class_groups']} groups span more than one class)",
        f"- Split sizes: {summary['sizes']} (target {fractions})",
        f"- Split hash: `{summary['splits_sha256']}`",
        f"- Split verification: {'PASS' if verification['ok'] else 'FAIL ' + str(verification['problems'])}",
        "",
        "## Recovered class counts vs expected",
        "",
        "| class | count | expected | delta |",
        "|---|---|---|---|",
    ]
    for c in CLASSES:
        lines.append(
            f"| {c} | {comparison['counts'][c]} | {EXPECTED_C0_COUNTS[c]} | {comparison['count_deltas'][c]} |"
        )
    lines += [
        "",
        "## Per-split class counts (fraction of class in each split)",
        "",
        "| class | " + " | ".join(f"{s} n (frac)" for s in fractions) + f" | total | target frac |",
        "|---|" + "---|" * (len(fractions) + 2),
    ]
    for c in CLASSES:
        total = sum(summary["counts"][s][c] for s in fractions)
        cells = " | ".join(
            f"{summary['counts'][s][c]} ({balance[c][s]:.3f})" for s in fractions
        )
        lines.append(f"| {c} | {cells} | {total} | " + "/".join(f"{fractions[s]:.2f}" for s in fractions) + " |")
    lines += [
        "",
        "## X1 lesion masks (released HAM10000 segmentations)",
        "",
        f"- Masks available: {audit['x1_masks']['all_present']}; QA-ok: {audit['x1_masks']['all_ok']}",
        f"- Test images: {audit['x1_masks']['test_rows']}, with mask: {audit['x1_masks']['test_with_mask']}, "
        f"QA-ok: {audit['x1_masks']['test_ok']}",
        f"- Mask/image size mismatches: {audit['x1_masks']['size_mismatches']} (masks are resized with "
        "nearest-neighbour when scoring)",
        "- Provenance: Harvard Dataverse doi:10.7910/DVN/DBW86T file "
        "`HAM10000_segmentations_lesion_tschandl.zip`. The source does not state that the masks were "
        "expert-drawn, so they are labelled X1 released masks, not \"expert-traced\".",
        "",
    ]
    (artifact_dir("reports") / "data_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, default=str))
    ok = verification["ok"] and comparison["all_match"]
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


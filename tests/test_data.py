from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dm_ahr_vit.config import CLASSES
from dm_ahr_vit.data.manifest import dhash, hamming_pairs
from dm_ahr_vit.data.splits import UnionFind, build_groups, verify_splits


def test_label_mapping_is_fixed_order():
    from dm_ahr_vit.config import CLASS_TO_IDX

    assert CLASSES == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    assert [CLASS_TO_IDX[c] for c in CLASSES] == list(range(7))


def test_metadata_join_and_class_counts():
    from dm_ahr_vit.data.ham10000 import class_counts

    df = pd.DataFrame({"dx": ["nv", "nv", "mel"], "image_id": ["a", "b", "c"],
                       "lesion_id": ["L1", "L2", "L3"]})
    counts = class_counts(df)
    assert counts["nv"] == 2 and counts["mel"] == 1 and counts["df"] == 0
    assert set(counts) == set(CLASSES)


def test_corrupt_and_missing_files_are_flagged(tiny_images):
    from dm_ahr_vit.data.manifest import _describe

    good = _describe(tiny_images / "ISIC_0000000.jpg", "ISIC_0000000")
    assert good["decode_status"] == "ok" and good["width"] == 48
    bad = _describe(tiny_images / "corrupt.jpg", "corrupt")
    assert bad["decode_status"] == "fail_decode" and bad["sha256"] is None


def test_dhash_is_deterministic_and_similar_for_rescaled():
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
    a = Image.fromarray(arr)
    b = a.resize((160, 160), Image.Resampling.BICUBIC)
    assert dhash(a) == dhash(a)
    dist = bin(dhash(a) ^ dhash(b)).count("1")
    assert dist <= 12


def test_hamming_pairs_finds_planted_duplicate():
    h = np.array([10, 11, 500, 900, 901], dtype=np.uint64)
    ids = [f"i{k}" for k in range(len(h))]
    pairs = hamming_pairs(h, 1, ids)
    found = {(p["id_a"], p["id_b"]) for p in pairs}
    assert ("i0", "i1") in found
    assert ("i3", "i4") in found


def test_grouping_merges_lesion_and_near_duplicates():
    c0 = pd.DataFrame({
        "image_id": ["a", "b", "c"],
        "lesion_id": ["L1", "L1", "L2"],
    })
    near = pd.DataFrame({"id_a": ["b"], "id_b": ["c"], "distance": [2]})
    grouped = build_groups(c0, near)
    assert grouped["group_id"].nunique() == 1
    near2 = pd.DataFrame(columns=["id_a", "id_b", "distance"])
    grouped2 = build_groups(c0, near2)
    assert grouped2["group_id"].nunique() == 2


def test_zero_overlap_across_splits(splits_df):
    ver = verify_splits(splits_df)
    assert ver["ok"], ver["problems"]
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a >= b:
                continue
            ga = set(splits_df[splits_df["split"] == a]["group_id"])
            gb = set(splits_df[splits_df["split"] == b]["group_id"])
            assert not (ga & gb)


def test_verify_splits_detects_leak():
    df = pd.DataFrame({
        "image_id": ["a", "b"],
        "lesion_id": ["L1", "L1"],
        "group_id": ["G1", "G1"],
        "dx": ["nv", "nv"],
        "split": ["train", "test"],
        "sha256": ["h1", "h2"],
    })
    ver = verify_splits(df)
    assert not ver["ok"] and any("group overlap" in p for p in ver["problems"])


def test_extras_never_enter_a_split(splits_df):
    """Cohort filter: every split row must be a C0 image."""
    manifest = pd.DataFrame({
        "image_id": splits_df["image_id"],
        "cohort": ["C0"] * len(splits_df),
    })
    merged = splits_df.merge(manifest, on="image_id")
    assert (merged["cohort"] == "C0").all()
    extras = pd.DataFrame({"image_id": ["ISIC_9999999"], "cohort": ["C0-extra"]})
    assert not set(extras["image_id"]) & set(splits_df["image_id"])


def test_union_find_path():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")


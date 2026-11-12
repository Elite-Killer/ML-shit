"""Lesion-grouped, class-stratified, frozen splits with leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CLASSES, save_json, sha256_file
from ..utils import utc_now


class UnionFind:
    """Dict-backed union-find with path compression."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller root wins
            if ra < rb:
                self._parent[rb] = ra
            else:
                self._parent[ra] = rb


def build_groups(c0: pd.DataFrame, near_duplicates: pd.DataFrame) -> pd.DataFrame:
    """Merge images sharing a lesion_id or a confirmed near-duplicate link."""
    uf = UnionFind()
    lesions = c0["lesion_id"].astype(str).unique().tolist()
    for lesion in lesions:
        members = c0.loc[c0["lesion_id"].astype(str) == lesion, "image_id"].astype(str).tolist()
        for other in members[1:]:
            uf.union(members[0], other)
    if len(near_duplicates):
        for r in near_duplicates.itertuples(index=False):
            uf.union(str(r.id_a), str(r.id_b))

    df = c0.copy()
    df["image_id"] = df["image_id"].astype(str)
    # Assign group ids in sorted order of each component's sorted member set.
    components: dict[str, list[str]] = {}
    for image_id in df["image_id"]:
        components.setdefault(uf.find(image_id), []).append(image_id)
    ordered = sorted(components.values(), key=lambda members: sorted(members)[0])
    group_of: dict[str, str] = {}
    for i, members in enumerate(ordered):
        for m in members:
            group_of[m] = f"G{i:05d}"
    df["group_id"] = df["image_id"].map(group_of)
    return df


def _assign_exact(images_per_group: list[int], target: int, rng: np.random.Generator) -> list[int]:
    """Pick indices of groups whose image counts sum exactly to `target`.

    Greedy over a shuffled order, then exchange with remaining groups until the
    exact image target is reached. Falls back to the closest achievable sum.
    """
    order = rng.permutation(len(images_per_group))
    chosen: list[int] = []
    rest = list(order)
    total = 0
    while rest and total < target:
        idx = rest.pop(0)
        if total + images_per_group[idx] <= target or not chosen:
            chosen.append(idx)
            total += images_per_group[idx]
        else:
            rest.append(idx)
            if all(total + images_per_group[j] > target for j in rest):
                break
    improved = True
    while total != target and improved:
        improved = False
        gap = target - total
        if gap > 0:
            for c_pos, c in enumerate(chosen):
                for r_pos, r in enumerate(rest):
                    delta = images_per_group[r] - images_per_group[c]
                    if 0 < delta <= gap:
                        chosen[c_pos], rest[r_pos] = r, c
                        total += delta
                        improved = True
                        break
                if improved:
                    break
        elif gap < 0:
            for c_pos, c in enumerate(chosen):
                for r_pos, r in enumerate(rest):
                    delta = images_per_group[c] - images_per_group[r]
                    if 0 < delta <= -gap:
                        chosen[c_pos], rest[r_pos] = r, c
                        total -= delta
                        improved = True
                        break
                if improved:
                    break
    return chosen


def freeze_splits(
    grouped: pd.DataFrame,
    seed: int,
    fractions: dict[str, float],
    out_dir,
    protocol_hash: str = "",
) -> dict:
    """Class-stratified group assignment targeting exact per-class image counts.

    Targets per class and split are round(fraction * class_images); val and test
    therefore receive the same image counts by construction. Groups never split.
    """
    rng = np.random.default_rng(seed)
    split_of: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in fractions}
    for dx in CLASSES:
        cls = grouped[grouped["dx"] == dx]
        groups = sorted(cls["group_id"].unique())
        sizes = {g: int((cls["group_id"] == g).sum()) for g in groups}
        n_images = sum(sizes.values())
        test_target = int(round(fractions.get("test", 0.15) * n_images))
        val_target = int(round(fractions.get("val", 0.15) * n_images))
        test_groups = {groups[i] for i in _assign_exact([sizes[g] for g in groups], test_target, rng)}
        remaining = [g for g in groups if g not in test_groups]
        val_groups = {remaining[i] for i in _assign_exact([sizes[g] for g in remaining], val_target, rng)}
        for g in groups:
            split = "test" if g in test_groups else "val" if g in val_groups else "train"
            split_of[g] = split
            counts[split][dx] += sizes[g]

    df = grouped.copy()
    df["split"] = df["group_id"].map(split_of)
    order = {"test": 0, "train": 1, "val": 2}
    df["_o"] = df["split"].map(order)
    df = df.sort_values(["_o", "group_id", "image_id"]).drop(columns="_o")
    out_path = out_dir / "splits.csv"
    df[["image_id", "lesion_id", "group_id", "dx", "split", "sha256"]].to_csv(out_path, index=False)

    sizes_summary = {s: int((df["split"] == s).sum()) for s in fractions}
    groups_summary = {s: int(df.loc[df["split"] == s, "group_id"].nunique()) for s in fractions}
    summary = {
        "counts": counts,
        "sizes": sizes_summary,
        "groups": groups_summary,
        "fractions": {s: float(v) for s, v in fractions.items()},
        "seed": int(seed),
        "protocol_hash": protocol_hash,
        "splits_sha256": sha256_file(out_path),
        "total_images": int(len(df)),
        "total_groups": int(df["group_id"].nunique()),
        "total_lesions": int(df["lesion_id"].nunique()),
        "created_at": utc_now(),
    }
    save_json(summary, out_dir / "split_summary.json")
    return summary


def verify_splits(df: pd.DataFrame) -> dict:
    """Automated leakage check: no group may span two splits."""
    problems: list[str] = []
    splits = [s for s in df["split"].unique() if isinstance(s, str)]
    for a_idx in range(len(splits)):
        for b_idx in range(a_idx + 1, len(splits)):
            a, b = splits[a_idx], splits[b_idx]
            ga = set(df.loc[df["split"] == a, "group_id"])
            gb = set(df.loc[df["split"] == b, "group_id"])
            for g in sorted(ga & gb):
                problems.append(f"group overlap: {g} appears in {a} and {b}")
    if not len(df):
        problems.append("empty splits table")
    return {"ok": not problems, "problems": problems}

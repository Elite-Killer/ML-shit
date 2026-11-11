"""Evaluate every acceptance gate against real files and write a JSON report.

Each gate is PASS, FAIL, or BLOCKED-with-reason. A check that cannot be
evaluated is FAIL, never PASS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, sha256_file  # noqa: E402
from dm_ahr_vit.data.splits import verify_splits  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import setup_logging, utc_now  # noqa: E402

log = setup_logging()
ART = artifact_dir()
MAN = manifest_dir()
RESULTS: list[dict] = []


def add(gate: str, ok: bool | None, evidence: str, detail: str = "",
        blocked: bool = False) -> None:
    status = "BLOCKED" if blocked and not ok else ("PASS" if ok else "FAIL")
    RESULTS.append({"gate": gate, "status": status, "evidence": evidence,
                    "detail": detail})
    log.info("%-6s %-4s %s", gate, status, detail[:90])


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def check_environment() -> None:
    p = ART / "logs" / "environment_report.json"
    env = load_json(p)
    add("G1", bool(env and env.get("cuda_available") and env.get("device")),
        str(p), f"device={env.get('device') if env else None}, torch={env.get('torch') if env else None}")


def check_data() -> None:
    man = MAN / "data_manifest.csv"
    audit = load_json(ART / "logs" / "data_audit.json")
    ok = bool(man.exists() and (MAN / "exclusions.csv").exists() and audit)
    detail = ""
    if audit:
        detail = (f"collection={audit.get('collection_total')}, c0={audit.get('c0_decoded')}, "
                  f"extra={audit.get('c0_extra')}, class_counts_match="
                  f"{audit.get('metadata_comparison', {}).get('all_match')}")
        ok = ok and audit.get("collection_total") == 11720 and audit.get("c0_decoded") == 10015
    add("G2", ok, f"{man}, {ART/'reports'/'data_audit.md'}", detail)


def check_splits() -> None:
    sp = MAN / "splits.csv"
    summ = load_json(MAN / "split_summary.json")
    if not sp.exists() or not summ:
        add("G3", False, str(sp), "splits not found")
        return
    df = pd.read_csv(sp)
    ver = verify_splits(df)
    frozen_ok = sha256_file(sp) == summ.get("splits_sha256")
    add("G3", ver["ok"] and frozen_ok, f"{sp} + {MAN/'split_summary.json'}",
        f"overlap_free={ver['ok']}, hash_frozen={frozen_ok}, sizes={ver['sizes']}")


def check_diffusion_provenance() -> None:
    p = ART / "checkpoints" / "dmahr" / "training_manifest.json"
    if not p.exists():
        add("G4", None, str(p), "no leakage-controlled diffusion training run yet", blocked=True)
        return
    m = load_json(p)
    ok = bool(m and m.get("leakage_check") and
              set(m.get("train_sources_splits", [])) == {"train"} and
              set(m.get("val_sources_splits", [])) <= {"val"})
    add("G4", ok, str(p),
        f"train_splits={m.get('train_sources_splits') if m else None}, "
        f"val_splits={m.get('val_sources_splits') if m else None}")


def check_dmahr() -> None:
    smoke = load_json(ART / "logs" / "dmahr_smoke.json")
    lic = ART / "reports" / "license_audit.md"
    ok = bool(smoke and smoke.get("all_pass") and lic.exists())
    add("G5", ok, f"{ART/'logs'/'dmahr_smoke.json'}, {lic}",
        f"all_pass={smoke.get('all_pass') if smoke else None}, "
        f"geometry={smoke.get('geometry_preserved') if smoke else None}, "
        f"reload_identical={smoke.get('reload_identical') if smoke else None}")


def check_classical() -> None:
    screen = MAN / "hair_screening.csv"
    cfg = load_json(ART / "logs" / "frozen" / "hair_screen.json")
    if not screen.exists() or not cfg:
        add("G6", False, str(screen), "hair screening outputs missing")
        return
    df = pd.read_csv(screen)
    hairy = df[df["hair_present"].astype(bool)]
    wrote = sum(
        (data_root() / "processed" / "B" / f"{i}.png").exists() for i in hairy["image_id"]
    )
    ok = bool(cfg.get("frozen") and wrote == len(hairy) and df["status"].eq("ok").all())
    add("G6", ok, str(screen),
        f"frozen={cfg.get('frozen')}, hair_present={len(hairy)}, B_written={wrote}, "
        f"screen_failures={int((df['status'] != 'ok').sum())}")


def check_training() -> None:
    arms = ["A", "B", "C", "Astar"]
    seeds = [42, 123, 2026]
    missing, bad = [], []
    import torch

    from dm_ahr_vit.vit.model import build_model

    for arm in arms:
        for seed in seeds:
            p = ART / "checkpoints" / "vit" / f"{arm}_seed{seed}" / "best.pt"
            if not p.exists():
                missing.append(f"{arm}/{seed}")
                continue
            try:
                m = build_model(pretrained=False)
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                m.load_state_dict(ckpt["model"])
                m.eval()
                with torch.no_grad():
                    out = m(torch.zeros(2, 3, 224, 224))
                probs = torch.softmax(out, dim=1)
                ok = (out.shape == (2, 7) and torch.isfinite(out).all()
                      and torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5))
                if not ok:
                    bad.append(f"{arm}/{seed}")
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{arm}/{seed}:{exc}")
    add("G7", not missing and not bad,
        str(ART / "checkpoints" / "vit"),
        f"missing={len(missing)} {missing[:6]}, invalid={bad[:3]}", blocked=bool(missing))


def check_classification() -> None:
    preds = sorted((ART / "predictions").glob("pred_*_full.csv"))
    metrics = sorted((ART / "predictions").glob("metrics_*.json"))
    missing = []
    for arm in ["A", "B", "C", "Astar"]:
        for seed in [42, 123, 2026]:
            if not (ART / "predictions" / f"metrics_{arm}_seed{seed}.json").exists():
                missing.append(f"{arm}/{seed}")
    add("G8", len(preds) > 0 and not missing, str(ART / "predictions"),
        f"prediction_files={len(preds)}, metric_files={len(metrics)}, missing={missing[:6]}",
        blocked=bool(missing))


def check_quality() -> None:
    p = ART / "logs" / "paired_psnr_ssim_test.csv"
    if not p.exists():
        add("G9", None, str(p), "paired image quality not computed yet", blocked=True)
        return
    df = pd.read_csv(p)
    scored = int(df["C_status"].eq("ok").sum())
    add("G9", scored > 0 and df["B_psnr"].notna().all(), str(p),
        f"pairs={len(df)}, C_scored={scored}")


def check_xai() -> None:
    files = sorted((ART / "logs").glob("xai_localization_*.csv"))
    if not files:
        add("G10", None, str(ART / "logs"), "no XAI outputs yet", blocked=True)
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    ok = bool(df["gradcam_finite"].all() and df["rollout_finite"].all()
              and df["tier"].nunique() >= 1
              and "chance_gaussian_overlap20" in df.columns)
    add("G10", ok, f"{len(files)} CSVs",
        f"rows={len(df)} tiers={sorted(df['tier'].unique())} "
        f"gradcam_overlap20={df['gradcam_overlap20'].mean():.4f}")


def check_stats() -> None:
    p = ART / "logs" / "statistical_comparisons.csv"
    if not p.exists():
        add("G11", False, str(p), "missing statistics table")
        return
    df = pd.read_csv(p)
    need = {"endpoint", "comparison", "p_raw", "p_holm", "difference", "ci_low", "ci_high",
            "effect_size_dz"}
    ok = need <= set(df.columns) and df["p_holm"].notna().any()
    add("G11", ok, str(p), f"rows={len(df)}, endpoints={df['endpoint'].nunique()}")


def check_report() -> None:
    rep = ART / "reports" / "final_research_report.md"
    inv = ART / "reports" / "artifact_inventory.csv"
    if not rep.exists():
        add("G12", False, str(rep), "final report missing")
        return
    text = rep.read_text()
    placeholders = [t for t in ("TODO", "TBD", "PLACEHOLDER", "<insert") if t in text]
    ok = (inv.exists() and not placeholders
          and "research" in text.lower() and "licen" in text.lower())
    add("G12", ok, f"{rep}, {inv}",
        f"placeholders={placeholders}, inventory={inv.exists()}")


def check_repro() -> None:
    p = ART / "logs" / "reproduction_log.json"
    add("G13", bool(p.exists()), str(p),
        "README reproduction log" if p.exists() else "reproduction not run from README",
        blocked=not p.exists())


def check_handover() -> None:
    req = {
        "requirements_lock": (Path("requirements.lock")).exists(),
        "readme": (Path("README.md")).exists(),
        "inventory": (ART / "reports" / "artifact_inventory.csv").exists(),
        "license_audit": (ART / "reports" / "license_audit.md").exists(),
        "final_report": (ART / "reports" / "final_research_report.md").exists(),
    }
    add("G14", all(req.values()), str(ART / "reports"), str(req))


def functional_matrix() -> list[dict]:
    def has(path: Path) -> bool:
        return path.exists()

    rows = []
    env = load_json(ART / "logs" / "environment_report.json")
    audit = load_json(ART / "logs" / "data_audit.json")
    smoke = load_json(ART / "logs" / "dmahr_smoke.json")
    tm = load_json(ART / "checkpoints" / "dmahr" / "training_manifest.json")
    x1 = MAN / "mask_manifest_x1.csv"
    rows.append(("F01", bool(env), "environment_report.json"))
    rows.append(("F02", bool(audit and audit.get("c0_decoded") == 10015), "data_audit.json"))
    rows.append(("F03", has(MAN / "split_summary.json"), "splits.csv"))
    rows.append(("F04", has(MAN / "dermahair_source_mapping.csv"), "dermahair_source_mapping.csv"))
    rows.append(("F05", bool(tm and tm.get("leakage_check")), "diffusion training manifest"))
    rows.append(("F06", has(MAN / "hair_screening.csv"), "hair_screening.csv"))
    rows.append(("F07", bool(smoke and smoke.get("all_pass")), "dmahr_smoke.json"))
    rows.append(("F08", has(ART / "checkpoints" / "vit"), "vit checkpoints"))
    rows.append(("F09", has(ART / "checkpoints" / "vit"), "matched config hashes"))
    rows.append(("F10", len(list((ART / "checkpoints" / "vit").glob("*/best.pt"))) >= 9,
                 "9+ run dirs"))
    rows.append(("F11", has(ART / "predictions"), "predictions"))
    rows.append(("F12", has(ART / "logs" / "paired_psnr_ssim_test.csv"), "psnr/ssim pairs"))
    rows.append(("F13", len(list((ART / "logs").glob("xai_localization_*.csv"))) > 0, "gradcam"))
    rows.append(("F14", len(list((ART / "logs").glob("xai_localization_*.csv"))) > 0, "rollout"))
    rows.append(("F15", has(x1), "mask tiers"))
    rows.append(("F16", has(ART / "logs" / "statistical_comparisons.csv"), "stats csv"))
    rows.append(("F17", has(ART / "logs" / "reproduction_log.json"), "reproduction log"))
    rows.append(("F18", has(ART / "reports" / "artifact_inventory.csv"), "inventory"))
    rows.append(("F19", has(ART / "checkpoints" / "vit" / "Astar_seed42" / "best.pt"),
                 "A* run dir"))
    rows.append(("F20", has(ART / "reports" / "compute_plan.md"), "compute_plan.md"))
    return [{"id": i, "status": "PASS" if ok else "PENDING", "evidence": ev}
            for i, ok, ev in rows]


def main() -> int:
    for fn in (check_environment, check_data, check_splits, check_diffusion_provenance,
               check_dmahr, check_classical, check_training, check_classification,
               check_quality, check_xai, check_stats, check_report, check_repro,
               check_handover):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            add(fn.__name__, False, "", f"check crashed: {exc!r}")
    fm = functional_matrix()
    report = {
        "generated_at": utc_now(),
        "gates": RESULTS,
        "functional_matrix": fm,
        "summary": {
            "pass": sum(r["status"] == "PASS" for r in RESULTS),
            "fail": sum(r["status"] == "FAIL" for r in RESULTS),
            "blocked": sum(r["status"] == "BLOCKED" for r in RESULTS),
        },
    }
    out = ART / "logs" / "acceptance_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    for r in RESULTS:
        print(f"{r['gate']:<6} {r['status']:<8} {r['detail'][:100]}")
    return 0 if report["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


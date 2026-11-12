"""G13: reproduce one recorded ViT run from a clean checkpoint directory.

Re-runs arm A, seed 42 through the same README entrypoint, writes into a fresh
directory (`A_seed42_repro`) so the recorded run is untouched, and compares the
validation macro-F1 and metric table against the recorded values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed, utc_now  # noqa: E402

log = setup_logging()

ARM, SEED = "A", 42
TOLERANCE = 0.02  # absolute macro-F1 difference treated as reproduced


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    from dm_ahr_vit.vit.train import train_arm  # noqa: PLC0415
    import torch  # noqa: PLC0415

    cfg = load_config("configs/vit.json")
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    baseline = read_json(
        artifact_dir("checkpoints", "vit", f"{ARM}_seed{SEED}") / "train_result.json"
    )
    if baseline is None:
        raise FileNotFoundError("recorded baseline result not found")

    run_dir = artifact_dir("checkpoints", "vit", f"{ARM}_seed{SEED}_repro")
    with timed("reproduction_train", log) as t:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = train_arm(
            arm=ARM,
            seed=SEED,
            cfg=cfg,
            splits_df=splits,
            raw_images_dir=isic_images_dir(),
            processed_root=data_root() / "processed",
            run_dir=run_dir,
            device=device,
        )

    recorded_f1 = float(baseline["best_val_macro_f1"])
    repro_f1 = float(result["best_val_macro_f1"])
    delta = abs(repro_f1 - recorded_f1)
    reproduced = delta <= TOLERANCE
    log_payload = {
        "generated_at": utc_now(),
        "gate": "G13",
        "arm": ARM,
        "seed": SEED,
        "device": device,
        "recorded": {
            "best_val_macro_f1": recorded_f1,
            "epochs_run": baseline["epochs_run"],
            "seconds": baseline["seconds"],
        },
        "reproduction": {
            "best_val_macro_f1": repro_f1,
            "epochs_run": result["epochs_run"],
            "seconds": result["seconds"],
        },
        "absolute_difference": round(delta, 6),
        "tolerance": TOLERANCE,
        "reproduced": reproduced,
        "command": "modal run --detach modal_app.py::reproduction_check",
        "entrypoint": "scripts/06_train_vit.py --arm A --seed 42 (fresh output dir)",
        "note": (
            "Both runs use the frozen split, the same config and the same torch "
            "seed. cuDNN kernel selection is not forced deterministic, so a "
            "small difference is expected; the tolerance is stated explicitly."
        ),
    }
    save_json(log_payload, artifact_dir("logs") / "reproduction_log.json")
    record_run("reproduction_check", gpu=device, seconds=t["seconds"],
               reproduced=reproduced, delta=round(delta, 6))
    print(json.dumps(log_payload, indent=2))
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())


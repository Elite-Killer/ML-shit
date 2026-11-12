"""Upload artifacts straight from the Modal volume to Google Drive.

Runs inside a Modal container: reads the volume, streams to Drive with rclone,
so nothing transits the user's machine. Requires a Google service-account JSON
whose client email has been granted access to the destination Drive folder.

Tiers keep the transfer small by default; `--tier all` moves everything.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.paths import artifact_root, data_root  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()

SA_ENV = "GDRIVE_SA_JSON"
TOKEN_ENV = "GDRIVE_TOKEN"
SA_FILE = Path("/tmp/gdrive-sa.json")
TOKEN_FILE = Path("/tmp/gdrive-token.json")
REMOTE = "gdrive"

# Each tier is (local_root, [relative paths to include]).
TIERS: dict[str, list[tuple[str, list[str]]]] = {
    "results": [
        ("artifacts", ["reports", "logs", "predictions", "panels"]),
        ("data", ["manifests"]),
    ],
    "checkpoints": [
        ("artifacts", ["checkpoints/dmahr"]),
        ("artifacts", ["checkpoints/vit"]),
    ],
    "processed": [
        ("data", ["processed"]),
    ],
    "raw": [
        ("data", ["raw"]),
    ],
}


def ensure_service_account() -> Path:
    """Materialise the service-account JSON from the environment or a secret file."""
    if SA_FILE.exists():
        return SA_FILE
    for candidate in ("/secrets/gdrive-sa.json", "/secrets/service_account.json"):
        if Path(candidate).exists():
            SA_FILE.write_text(Path(candidate).read_text())
            return SA_FILE
    raw = os.environ.get(SA_ENV)
    if not raw:
        raise SystemExit(
            f"No service-account credentials. Set the {SA_ENV} env var or mount a "
            "secret file at /secrets/gdrive-sa.json."
        )
    SA_FILE.write_text(raw)
    parsed = json.loads(raw)
    log.info("service account: %s", parsed.get("client_email", "?"))
    return SA_FILE


def write_rclone_config(sa_path: Path) -> None:
    """Configure an rclone remote backed by the service account."""
    config = (
        f"[{REMOTE}]\n"
        "type = drive\n"
        "scope = drive\n"
        f"service_account_file = {sa_path}\n"
    )
    path = Path("/tmp/rclone.conf")
    path.write_text(config)
    os.environ["RCLONE_CONFIG"] = str(path)
    log.info("wrote rclone config for remote '%s'", REMOTE)


def oauth_token() -> str | None:
    """Return an rclone OAuth token JSON, if the user supplied one.

    A service account can upload to a folder shared with it, but Google often
    rejects it with "Service Accounts do not have storage quota" on personal
    Gmail. OAuth as the real user avoids that entirely, so it is preferred when
    available.
    """
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    for candidate in ("/secrets/gdrive-token.json", "/secrets/rclone-token.json"):
        if Path(candidate).exists():
            return Path(candidate).read_text().strip()
    return os.environ.get(TOKEN_ENV)


def write_rclone_config_oauth(token: str) -> None:
    config = (
        f"[{REMOTE}]\n"
        "type = drive\n"
        "scope = drive\n"
        f"token = {token}\n"
    )
    path = Path("/tmp/rclone.conf")
    path.write_text(config)
    os.environ["RCLONE_CONFIG"] = str(path)
    log.info("wrote rclone config using OAuth user token")


def rclone(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["rclone"] + args
    log.info("rclone %s", " ".join(args))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def normalise_drive_folder(dest: str) -> str:
    dest = dest.strip().strip("/")
    return f"{REMOTE}:{dest}" if dest else f"{REMOTE}:"


def upload_tier(tier: str, dest: str, dry_run: bool) -> list[dict]:
    """Upload one tier and return a per-path record."""
    records: list[dict] = []
    for root_name, subpaths in TIERS[tier]:
        root = artifact_root() if root_name == "artifacts" else data_root()
        for sub in subpaths:
            src = root / sub
            if not src.exists():
                log.warning("skip missing %s", src)
                records.append({"source": str(src), "status": "missing"})
                continue
            target = f"{normalise_drive_folder(dest)}/{sub}"
            args = [
                "copy", str(src), target,
                "--transfers", "8",
                "--checkers", "16",
                "--drive-chunk-size", "128M",
                "--retries", "5",
                "--low-level-retries", "20",
                "--stats", "30s",
                "--stats-one-line",
                "--verbose" if dry_run else "--quiet",
            ]
            if dry_run:
                args.append("--dry-run")
            started = timed = None
            try:
                with timed(f"upload_{sub}", log) as el:
                    proc = rclone(args)
                records.append({
                    "source": str(src),
                    "destination": target,
                    "status": "ok",
                    "seconds": el["seconds"],
                })
            except subprocess.CalledProcessError as exc:
                tail = (exc.stderr or "")[-800:]
                log.error("upload failed for %s: %s", src, tail)
                records.append({
                    "source": str(src),
                    "destination": target,
                    "status": "failed",
                    "error": tail,
                })
    return records


def require_rclone() -> str:
    """Fail early with an actionable message instead of a bare FileNotFoundError."""
    path = shutil.which("rclone")
    if path is None:
        raise SystemExit(
            "rclone not found on PATH. This script is meant to run inside the "
            "Modal image (modal_app.py::upload_to_drive), which installs it."
        )
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tier",
        default="results",
        choices=["results", "checkpoints", "processed", "raw", "all"],
    )
    ap.add_argument("--dest", default="dmahr-vit",
                    help="folder path inside the shared Drive")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    require_rclone()
    token = oauth_token()
    if token:
        write_rclone_config_oauth(token)
    else:
        write_rclone_config(ensure_service_account())

    version = rclone(["--version"], check=False)
    log.info("rclone: %s", (version.stdout or "").splitlines()[:1])

    tiers = list(TIERS) if args.tier == "all" else [args.tier]
    all_records: list[dict] = []
    total = {"seconds": 0.0}
    with timed(f"upload_to_drive_{args.tier}", log) as total:
        for tier in tiers:
            log.info("=== tier: %s ===", tier)
            all_records.extend(upload_tier(tier, args.dest, args.dry_run))

    failed = [r for r in all_records if r["status"] == "failed"]
    summary = {
        "tier": args.tier,
        "dest": args.dest,
        "dry_run": args.dry_run,
        "paths": len(all_records),
        "failed": len(failed),
        "records": all_records,
        "seconds": total["seconds"],
    }
    out = artifact_root() / "logs" / f"drive_upload_{args.tier}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    record_run("upload_to_drive", gpu=None, seconds=total["seconds"],
               tier=args.tier, failed=len(failed))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    for r in all_records:
        print(f"  {r['status']:8} {r.get('destination', r['source'])}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())


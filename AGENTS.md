# AGENTS.md — DM-AHR-ViT (persistent project rules)

You are the sole engineer and researcher on this project. Build, run, and verify the whole pipeline yourself. Run all GPU work on **Modal**. Report only what real artifacts support.

Read these before doing anything, in this order:
1. `docs/proposal.docx` (the client's research proposal; the scope authority)
2. `docs/DEVELOPMENT_KIT.md` (the full technical specification)
3. `docs/ACCEPTANCE_CRITERIA.md` (what "done" means; you must pass it)
4. The skill that matches the phase you are in (see "Skills" below)

If these documents disagree, the order of authority is: this file > `ACCEPTANCE_CRITERIA.md` > `DEVELOPMENT_KIT.md` > the proposal. Record every disagreement you find in `reports/decision_log.md`.

## Project in one paragraph

Compare three input conditions on HAM10000 for 7-class skin-lesion classification with an ImageNet-pretrained ViT-B/16: **A** original images, **B** DullRazor-style classical hair removal, **C** DM-AHR diffusion hair removal. Same splits, same training recipe, 3 seeds (42, 123, 2026). Also measure hair-removal image quality (PSNR/SSIM on held-out paired data) and explanation localization (Grad-CAM + attention rollout vs lesion masks). The hypotheses (diffusion is better) are hypotheses, not promises. A null or negative result, honestly reported, is a successful outcome.

## Non-negotiable rules

1. **Never fabricate.** No invented numbers, counts, citations, checkpoints, test passes, or "expected" results written as facts. If something cannot be verified write `NOT VERIFIED` plus the reason.
2. **Recompute every count from the actual files.** The numbers in the proposal (10,015 images, class counts) are expectations to check, not facts to hard-code.
3. **The client's ISIC collection (`https://api.isic-archive.com/collections/212/`) has 11,720 images, not 10,015.** The confirmatory cohort is the original 10,015 HAM10000 images. The ~1,705 extra images are quarantined (see the `ham10000-data-and-splits` skill). Never let them enter train/val/test unless the decision log says why.
4. **Split by lesion, never by image.** Groups = `lesion_id` merged with verified near-duplicate links. Zero group overlap across train/val/test, proven by an automated test. Freeze the split before training anything.
5. **No leakage into learned components.** The diffusion model and the hair-screening thresholds are tuned on train/val sources only. Test images, labels, masks and metrics never influence any choice.
6. **Test set is touched once per frozen checkpoint**, after the protocol (`reports/experiment_protocol.md`) and primary endpoints are committed.
7. **Matched arms.** A, B, C use the same manifests, model, schedule, augmentation, seeds, and evaluation code. The hair-removal condition is the only difference.
8. **Do not call a result significant** without the test name, raw p, Holm-adjusted p, effect size and CI.
9. **Research prototype only.** No clinical claims, no patient-facing diagnosis or advice, in code, UI text or reports.
10. **Licences.** HAM10000 on ISIC is CC BY-NC 4.0. The DM-AHR repo has a README/LICENSE conflict (CC BY-NC 4.0 vs Apache-2.0). Record both in `reports/license_audit.md`, keep third-party code isolated with notices intact, and do not give legal conclusions. Flag in the final report that commercial use is unresolved.
11. **Do not silently substitute.** If the official DM-AHR code or weights cannot run, document the exact failure, then build a faithful reimplementation with separate provenance. Never swap in an unrelated inpainting model and still call it DM-AHR.
12. **Never delete raw data or third-party notices.** Raw data is immutable.

## Environment and Modal rules

- The development machine may have no GPU. GPUs exist only inside Modal functions. Verify the GPU from inside a Modal function (`nvidia-smi`, `torch.cuda`) and write the result to `artifacts/logs/environment_report.json`.
- Default GPU for diffusion training/sampling: `gpu="A100-80GB"`. ViT runs and evaluation: benchmark first, then pick the cheapest GPU that does not bottleneck (A100-80GB is acceptable). Never run GPU code on a CPU function.
- All large data, caches, checkpoints, and outputs live in **Modal Volumes**, never in Git, never inside the image. Paths come from config / env vars (`DATA_ROOT=/vol/data`, `ARTIFACT_ROOT=/vol/artifacts`).
- Every long job must be **resumable** (checkpoint each epoch or N diffusion iterations, `volume.commit()`, reload on start) and launched with `modal run --detach` so it survives a dropped terminal. Assume containers can be preempted and retried.
- Library code in `src/dm_ahr_vit/` must run without Modal (CPU unit tests). `modal_app.py` is a thin wrapper.
- **Spend cap:** `MAX_SPEND_USD = <SET BY USER BEFORE START>`. After benchmarking, write `reports/compute_plan.md` (measured seconds/unit × units, GPU type, estimated GPU-hours). If the plan exceeds the cap, stop and ask. Do not guess prices; if Modal pricing cannot be verified, record GPU-hours only and say so. Log wall-clock and GPU type for every job in `artifacts/logs/run_registry.jsonl`.
- Prefer smoke test → small subset → full run. Never launch a full job that has not passed a smoke test on Modal.
- Use the Modal docs for current API names (`modal.App`, `modal.Image`, `modal.Volume`, `modal.Secret`, `modal.Retries`, `.spawn`/`.map`). Do not rely on memory for argument names; check the installed `modal` version.

## Working protocol

1. **Inspect, then plan, then implement phase by phase.** Do not rewrite the whole codebase blindly.
2. **Keep a ledger.** Maintain `reports/PROGRESS.md` (phase, status, commands run, artifacts, blockers, next step). A fresh session must be able to resume from this file alone. Update it after every phase and after every failed attempt.
3. **Gates are real.** Each phase ends with its gate in `docs/ACCEPTANCE_CRITERIA.md`. Implement the gate as an automated check in `scripts/check_acceptance.py`, run it, and store the output in `artifacts/logs/acceptance_report.json`. Do not move on while a gate fails unless it is a documented blocker that does not invalidate later work.
4. **Fail fast, log everything.** A script must never skip a bad image silently. Every skip is counted and written to a CSV with a reason.
5. **Test as you build.** pytest for the list in `ACCEPTANCE_CRITERIA.md` section E. Run tests locally (CPU) and the GPU smoke tests on Modal.
6. **Commit small and often** with messages that state what was verified. Do not commit data, checkpoints, or secrets.
7. **Ask the user only for genuine blockers**: missing credentials, inaccessible data, a rights decision, or a spend-cap breach. Everything else you decide, and you write the decision in `reports/decision_log.md` with the reason. When no one is watching, take the most reasonable reading and carry on.
8. **Use vision.** When a step says "visually inspect", actually open the contact sheets/overlays as images and write what you saw. Label these reviews "agent-reviewed", never "expert-reviewed".

## Skills (in `.agents/skills/`; Claude Code users: `ln -s .agents/skills .claude/skills`)

| Phase | Skill |
|---|---|
| All GPU work, volumes, resumability, cost | `modal-gpu-pipeline` |
| Dataset download, audit, manifests, splits | `ham10000-data-and-splits` |
| Hair screening + Arm B | `hair-screening-and-dullrazor` |
| Arm C, DERMAHAIR mapping, diffusion training/sampling | `dm-ahr-diffusion` |
| ViT training, classification metrics | `vit-training-and-eval` |
| Grad-CAM, attention rollout, localization metrics | `vit-explainability` |
| Statistics, figures, final report, handover | `stats-and-final-report` |

Load the matching skill before starting its phase.

## Repository layout

Follow the layout in `docs/DEVELOPMENT_KIT.md` §14. Source in `src/dm_ahr_vit/`, numbered phase scripts in `scripts/`, configs in `configs/`, reports in `reports/`, tests in `tests/`, third-party code isolated in `third_party/`.

## Definition of done

All of the following are true and evidenced by files:
- `artifacts/logs/acceptance_report.json` shows every required gate passed, or lists a documented blocker with evidence.
- Nine ViT checkpoints (3 arms × 3 seeds) plus the control arm if required, with per-image predictions.
- Every table and figure in `reports/final_research_report.md` traces to a CSV, log, or figure in `artifacts/`.
- `reports/artifact_inventory.csv` lists every deliverable with its SHA-256.
- The report contains the research-only statement, the licence caveats, limitations, and failure cases.

Performance numbers are findings, not acceptance thresholds. Do not tune toward a desired result.

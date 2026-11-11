# Short kickoff prompt (paste this if the agent already has the repo files loaded)

You are the sole engineer on the DM-AHR-ViT project. Read `AGENTS.md`, then `docs/proposal.docx`, `docs/DEVELOPMENT_KIT.md`, and `docs/ACCEPTANCE_CRITERIA.md`. Build, run on Modal GPUs, and verify the whole pipeline end to end, phase by phase, loading the matching skill from `.agents/skills/` before each phase.

Start now:
1. Summarize the repo state and write `reports/PROGRESS.md`.
2. Run the environment/GPU check from inside a Modal GPU function; write `artifacts/logs/environment_report.json`.
3. Create the repository skeleton from kit §14, `modal_app.py`, and `scripts/check_acceptance.py`.
4. Audit the DM-AHR repo, weights, DERMAHAIR, and licences (`reports/resource_audit.md`, `reports/license_audit.md`).
5. Download the client's ISIC collection 212, define the 10,015-image cohort versus the extras, and produce the data audit before anything is trained.
6. Continue through the phases; stop only at genuine blockers (credentials, inaccessible data, a rights decision, spend cap) and then ask with specifics.

Never fabricate results. Report real outcomes, including null or negative ones. Finish when `acceptance_report.json` passes or every open item is a documented blocker.

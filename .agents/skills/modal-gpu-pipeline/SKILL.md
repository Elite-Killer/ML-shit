---
name: modal-gpu-pipeline
description: Run, resume, and budget every GPU/CPU job of the DM-AHR-ViT project on Modal. Use before writing modal_app.py, launching any training/sampling job, handling volumes or checkpoints, or estimating compute cost.
---

# Modal GPU pipeline

Check the installed Modal version and the current docs (https://modal.com/docs) for exact argument names before coding. The patterns below are the intended design, not a copy-paste API reference. Modal 1.x renamed some arguments (for example `container_idle_timeout` became `scaledown_window`).

## Facts checked when this skill was written
- GPU strings include `T4, L4, A10, L40S, A100-40GB, A100-80GB, H100, H200, B200`. Use `gpu="A100-80GB"` for the 80 GB card. A plain `"A100"` may be upgraded to 80 GB; `"H100"` may be upgraded to H200 (`"H100!"` prevents that). Multi-GPU: `"H100:4"`. Fallback list: `gpu=["A100-80GB", "H100"]`.
- Volumes: `modal.Volume.from_name("name", create_if_missing=True)`, mounted with `volumes={"/vol/data": vol}`; call `vol.commit()` after writes and `vol.reload()` to see other containers' writes.
- Secrets: `modal secret create kaggle KAGGLE_USERNAME=... KAGGLE_KEY=...`, used with `secrets=[modal.Secret.from_name("kaggle")]`.
- Run: `modal run modal_app.py::fn`, add `--detach` for jobs that must outlive the terminal; `modal app logs`, `modal volume ls/get/put`.
- VERIFY yourself: the maximum function `timeout`, retry semantics, `.spawn()` / `.map()` / `.starmap()` signatures, and current GPU prices. If prices are unverifiable, record GPU-hours only.

## Architecture
- One `modal.App("dm-ahr-vit")`. One pinned image: CUDA-capable PyTorch (pin torch/torchvision versions), `opencv-python-headless`, `scikit-learn`, `pandas`, `scipy`, `scikit-image`, `pytest`, `imagehash`, `pyyaml`, `kaggle`, `gdown`, `isic-cli` (needs Python ≥ 3.12; if the DM-AHR repo needs an older Python, give it its own image/function). Add `src/` and `configs/` with `add_local_python_source` / `add_local_dir`. Freeze the result into `requirements.lock`.
- Volumes: `dmahr-data` (`/vol/data`: raw, interim, processed caches) and `dmahr-artifacts` (`/vol/artifacts`: checkpoints, outputs, metrics, logs). Config reads `DATA_ROOT` and `ARTIFACT_ROOT`.
- Functions: CPU functions for data audit, hashing, DullRazor, metrics, bootstrap (set `cpu=` high, e.g. 16, and use multiprocessing). GPU functions for DM-AHR training and sampling, ViT training, explanation generation. Never run GPU code in a CPU function.
- Library code in `src/dm_ahr_vit/` imports nothing from `modal`. `modal_app.py` only wires configs, volumes and entry points. Unit tests run locally on CPU with tiny fixtures.

## Resumable job pattern
```
start:   vol.reload(); load latest checkpoint if present (model, optimizer, scheduler, scaler, epoch/iter, RNG state)
loop:    every epoch (ViT) or N iterations (diffusion): save ckpt to a temp name, rename atomically, vol.commit()
finish:  write final.json (config hash, git commit, metrics, wall-clock, GPU name, peak memory); vol.commit()
retries: modal.Retries(max_retries=3, ...) so a preempted container resumes from the last checkpoint
```
Idempotence: every output file or shard is skipped when it exists and its checksum entry is in the manifest. A rerun of a finished job does nothing.

## Data-loading tips for few-CPU containers
- Pre-resize each arm once into a single uint8 array or a few shards on the volume; copy it to container-local disk (or memory) at job start. Many small files on a Volume are slow.
- Do augmentation on the GPU (flip, rotate, affine, brightness via torch ops or `torchvision.transforms.v2` on tensors) so the CPU does not bottleneck an A100.
- Use bf16 autocast and `torch.backends.cuda.matmul.allow_tf32 = True`; record these flags.

## Parallelism
- 9 (or 12) ViT runs are independent: launch with `.spawn()`/`.starmap()` in separate containers, each with its own output dir `artifacts/checkpoints/vit/{arm}/{seed}/`.
- DM-AHR sampling: split the hair-present image list into shards, `.map()` over shards, batch inside each shard, write each shard's PNGs plus a shard manifest.
- Keep the number of simultaneous GPUs modest; Modal queueing grows with large requests.

## Compute plan and spend cap (mandatory)
1. Smoke test each GPU function on 1–8 inputs.
2. Benchmark: seconds per ViT epoch; seconds per DM-AHR sampling batch at full steps; seconds per 100 diffusion training iterations; peak memory.
3. Write `reports/compute_plan.md`: unit costs measured, units needed, GPU type, GPU-hours total, retries margin, and the cost if prices were verified.
4. Compare with `MAX_SPEND_USD` in `AGENTS.md`. If the plan exceeds it, stop and ask the user with the numbers. Never launch a full run first and "see how it goes".
5. Append every job to `artifacts/logs/run_registry.jsonl`: job, GPU, start/end, status, GPU-hours, checkpoint path.

## Pitfalls
- Container filesystems are ephemeral: anything not under a mounted Volume is lost.
- Forgetting `vol.commit()` loses work on preemption.
- Writing from several containers into one file: use one output path per container/shard.
- Don't download datasets into the image build; download once into the volume.
- Outbound internet from containers is normally available; if a source is blocked, record it and ask the user to upload the file with `modal volume put`.
- Do not print secrets in logs.

## Handover
`scripts/export_handover.py` collects small artifacts; list the exact `modal volume get dmahr-artifacts <path> <local>` commands for checkpoints in the README. State the volume names so the user can keep or delete them.

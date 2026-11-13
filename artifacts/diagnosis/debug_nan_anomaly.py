"""Throwaway: count NaN gradients on the upstream p_losses path vs our training_loss path."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/tmp/hoplite/workspace")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_trainer_sampler_contract as T  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import _ensure_path, build_opt  # noqa: E402
from dm_ahr_vit.diffusion.train import training_loss  # noqa: E402

_ensure_path()
import model.networks as networks  # noqa: E402

threads = int(sys.argv[1])
torch.set_num_threads(threads)
cfg = T._small_cfg()
torch.manual_seed(123)
n = networks.define_G(build_opt(cfg))
n.set_new_noise_schedule({**cfg["beta_schedule"], "n_timestep": 2000}, device=torch.device("cpu"))
n.set_loss("cpu")
n.train()

torch.autograd.set_detect_anomaly(True)
bad = {"upstream": 0, "ours": 0}
total = 400
for i in range(total):
    hr = torch.rand(4, 3, 16, 16) * 2 - 1
    cond = torch.rand(4, 3, 16, 16) * 2 - 1
    for name, fn in (("upstream", lambda: n.p_losses({"HR": hr, "SR": cond}) / hr.numel()),
                     ("ours", lambda: training_loss(n, hr, cond))):
        np.random.seed(i)
        torch.manual_seed(i)
        n.zero_grad(set_to_none=True)
        loss = fn()
        try:
            loss.backward()
        except RuntimeError as e:
            print("ANOMALY", name, i, str(e)[:160]); import traceback; print("".join(traceback.format_exc().splitlines(True)[-3:])[:400]); raise SystemExit
        g = torch.cat([p.grad.flatten() for p in n.parameters() if p.grad is not None])
        if torch.isnan(g).any():
            bad[name] += 1
print(f"threads={threads} runs={total} nan_grad_runs={bad}")

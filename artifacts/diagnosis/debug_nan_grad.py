"""Throwaway: find why the gradient differential test intermittently yields NaN."""
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

for build in range(12):
    cfg = T._small_cfg()
    n = networks.define_G(build_opt(cfg))
    n.set_new_noise_schedule({**cfg["beta_schedule"], "n_timestep": 2000}, device=torch.device("cpu"))
    n.set_loss("cpu")
    n.train()
    row = []
    for seed in (1, 2):
        hr = torch.rand(4, 3, 16, 16) * 2 - 1
        cond = torch.rand(4, 3, 16, 16) * 2 - 1

        def run(fn):
            np.random.seed(seed)
            torch.manual_seed(seed)
            n.zero_grad(set_to_none=True)
            loss = fn()
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in n.parameters() if p.grad is not None])
            return float(loss), g

        lu, gu = run(lambda: n.p_losses({"HR": hr, "SR": cond}) / hr.numel())
        ls, gs = run(lambda: training_loss(n, hr, cond))
        row.append((seed, round(lu, 5), round(ls, 5),
                    int(torch.isnan(gu).sum()), int(torch.isnan(gs).sum()),
                    float(gu.norm()), float(gs.norm())))
    print(build, row)

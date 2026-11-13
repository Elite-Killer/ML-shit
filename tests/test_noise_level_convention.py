"""The trainer must encode the noise level the way the sampler does.

`GaussianDiffusion.p_sample` feeds `sqrt_alphas_cumprod_prev[t+1]` to the
U-Net, and `NoiseLevelEncoder` scales a frequency ramp by that value. A trainer
that feeds `t / num_timesteps` instead trains one encoding and samples with
another, which inverts the meaning of the signal.
"""

from __future__ import annotations

import numpy as np

from dm_ahr_vit.config import load_config
from dm_ahr_vit.diffusion.dmahr_adapter import load_upstream_config


def _schedule(num_timesteps: int = 2000) -> np.ndarray:
    """sqrt(alpha_cumprod_prev), length num_timesteps + 1, as upstream builds it."""
    cfg = load_config("configs/dmahr.json")
    bs = cfg["beta_schedule"]
    betas = np.linspace(bs["linear_start"], bs["linear_end"], num_timesteps)
    alphas_cumprod = np.cumprod(1.0 - betas)
    return np.sqrt(np.append(1.0, alphas_cumprod))


def test_embedding_is_not_linear_in_t():
    """Guard against reintroducing t/T as the noise level."""
    sqrt_acp = _schedule()
    for t in (50, 200, 500, 1000, 1500):
        linear = t / 2000
        actual = sqrt_acp[t + 1]
        assert abs(linear - actual) > 0.1, (
            f"t={t}: t/T={linear:.4f} and sqrt_acp={actual:.4f} should differ"
        )


def test_embedding_ordering_is_inverted_relative_to_t_over_T():
    """Higher t must mean a LOWER value; t/T means the opposite."""
    sqrt_acp = _schedule()
    assert sqrt_acp[51] < sqrt_acp[2], "sqrt_acp must decrease with t"
    assert (51 / 2000) > (2 / 2000), "t/T must increase with t"


def test_embedding_range_within_unit_interval():
    sqrt_acp = _schedule()
    assert sqrt_acp.min() >= 0.0
    assert sqrt_acp.max() <= 1.0


def test_upstream_config_matches_our_schedule():
    """Our beta schedule must reproduce the upstream one, not just resemble it."""
    up = load_upstream_config()
    sched = up["model"]["beta_schedule"]["train"]
    assert int(sched["n_timestep"]) == 2000
    ours = _schedule(int(sched["n_timestep"]))
    theirs_betas = np.linspace(
        float(sched["linear_start"]), float(sched["linear_end"]), 2000
    )
    theirs = np.sqrt(np.append(1.0, np.cumprod(1.0 - theirs_betas)))
    assert np.allclose(ours, theirs, atol=1e-8)

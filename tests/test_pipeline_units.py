from __future__ import annotations

import numpy as np
import pytest
import torch

from dm_ahr_vit.config import CLASSES


def test_rgb_resize_normalization():
    from dm_ahr_vit.data.dataset import build_transforms
    from dm_ahr_vit.config import load_config

    cfg = load_config("configs/vit.json")
    tfm = build_transforms(cfg, train=False)
    from PIL import Image

    img = Image.fromarray(np.zeros((50, 70, 3), dtype=np.uint8))
    out = tfm(img)
    assert out.shape == (3, 224, 224)
    # ImageNet-normalized zero image is (0-mean)/std, not zero.
    assert torch.allclose(out[:, 0, 0], torch.tensor([-2.1179, -2.0357, -1.8044]),
                          atol=1e-3)


def test_dullrazor_deterministic_and_same_size():
    from dm_ahr_vit.config import load_config
    from dm_ahr_vit.hair.dullrazor import apply_dullrazor
    from dm_ahr_vit.hair.screening import detect_hair

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(120, 100, 3), dtype=np.uint8)
    cfg = load_config("configs/hair_screen.json")
    dull = load_config("configs/dullrazor.json")
    m1, s1 = detect_hair(img, cfg)
    m2, s2 = detect_hair(img, cfg)
    assert m1.shape == img.shape[:2]
    assert np.array_equal(m1, m2) and s1["coverage"] == s2["coverage"]
    out1 = apply_dullrazor(img, m1, dull)
    out2 = apply_dullrazor(img, m2, dull)
    assert out1.shape == img.shape
    assert np.array_equal(out1, out2)


def test_synthetic_hair_is_seeded_and_masked():
    from PIL import Image

    from dm_ahr_vit.data.synthetic_hair import render_hair

    img = Image.fromarray(np.full((96, 80, 3), 200, dtype=np.uint8))
    a, m1, p1 = render_hair(img, "ISIC_0000001", seed=7)
    b, m2, p2 = render_hair(img, "ISIC_0000001", seed=7)
    c, _, _ = render_hair(img, "ISIC_0000001", seed=8)
    assert np.array_equal(np.array(a), np.array(b))
    assert not np.array_equal(np.array(a), np.array(c))
    assert p1["n_hairs"] == p2["n_hairs"]
    assert m1.size == img.size and set(np.unique(np.array(m1))) <= {0, 255}


def test_preprocess_postprocess_dmahr_range():
    from PIL import Image

    from dm_ahr_vit.diffusion.dmahr_adapter import postprocess, preprocess

    imgs = [Image.fromarray(np.full((40, 40, 3), v, dtype=np.uint8)) for v in (0, 128, 255)]
    x = preprocess(imgs, size=128)
    assert x.shape == (3, 3, 128, 128)
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    back = postprocess(x)
    assert back[0].size == (128, 128)
    assert np.array(back[2]).max() == 255


def test_vit_output_shape_and_probabilities():
    from dm_ahr_vit.vit.model import build_model

    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, len(CLASSES))
    probs = torch.softmax(out, dim=1)
    assert torch.isfinite(out).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_checkpoint_roundtrip_equivalence(tmp_path):
    from dm_ahr_vit.vit.model import build_model, load_checkpoint, save_checkpoint
    from dm_ahr_vit.vit.train import cosine_with_warmup

    model = build_model(pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = cosine_with_warmup(opt, 1, 3, 4)
    x = torch.zeros(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        before = model(x)
    p = tmp_path / "ckpt.pt"
    save_checkpoint(p, model, opt, sched, 0, 0.5, {"arm": "A", "seed": 42})
    fresh = build_model(pretrained=False)
    meta = load_checkpoint(p, fresh)
    fresh.eval()
    with torch.no_grad():
        after = fresh(x)
    assert torch.allclose(before, after, atol=1e-6)
    assert meta["meta"]["seed"] == 42


def test_gradcam_shape_finite_and_target_dependent():
    from dm_ahr_vit.xai.vit_xai import GradCAM

    from dm_ahr_vit.vit.model import build_model

    model = build_model(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    with GradCAM(model) as cam:
        m0 = cam(x, class_idx=0)
        m5 = cam(x, class_idx=5)
    assert m0.shape == (14, 14)
    assert np.isfinite(m0).all() and m0.min() >= 0 and m0.max() <= 1.0
    assert not np.allclose(m0, m5)


def test_attention_rollout_dims_and_validity():
    from dm_ahr_vit.xai.vit_xai import attention_rollout, capture_attention

    from dm_ahr_vit.vit.model import build_model

    model = build_model(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    with capture_attention(model) as captured:
        model(x)
    assert len(captured) == 12
    assert captured[0].shape == (1, 197, 197)
    roll = attention_rollout(model, x)
    assert roll.shape == (14, 14)
    assert np.isfinite(roll).all()
    assert roll.min() >= 0 and roll.max() <= 1.0


def test_mask_matching_and_binary(tmp_path):
    from dm_ahr_vit.data.masks import load_mask_224

    from PIL import Image

    arr = np.zeros((60, 60), dtype=np.uint8)
    arr[20:40, 20:40] = 255
    p = tmp_path / "ISIC_0000001_segmentation.png"
    Image.fromarray(arr).save(p)
    m = load_mask_224(p)
    assert m.shape == (224, 224)
    assert set(np.unique(m)) <= {0, 1}
    assert 0 < m.mean() < 1


def test_xai_metrics_on_handmade_example():
    from dm_ahr_vit.xai.vit_xai import lesion_overlap, pointing_game

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    sal = np.zeros((10, 10))
    sal[0:2, 0:2] = 1.0
    assert lesion_overlap(sal, mask, 0.04) == pytest.approx(1.0)
    assert pointing_game(sal, mask) == 1.0
    far = np.zeros((10, 10))
    far[9, 9] = 1.0
    assert pointing_game(far, mask) == 0.0
    # Top 1% of a 10x10 map is exactly the single peak, which is off the lesion.
    assert lesion_overlap(far, mask, 0.01) == pytest.approx(0.0)


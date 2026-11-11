---
name: vit-explainability
description: Implement ViT Grad-CAM (patch-embedding grid) and attention rollout for torchvision ViT-B/16, generate overlays, and compute lesion-overlap and pointing-game localization against lesion masks with chance baselines. Use for Phase 10.
---

# ViT explainability and localization

## Grad-CAM (primary, per proposal: patch-embedding feature grid)
- Hook `model.conv_proj`. Its output is `(B, 768, 14, 14)`: no reshape needed. Capture activations in a forward hook and gradients with `register_full_backward_hook` or `tensor.register_hook`.
- Backpropagate the logit of the predicted class (`target_class` argument for tests). `weights = grad.mean(dim=(2,3))`; `cam = relu((weights[:, :, None, None] * act).sum(1))`; bilinear upsample to 224×224; per-image min–max to [0,1] (constant map → zeros, log it).
- Use `model.eval()`, `torch.enable_grad()` around the explanation pass, fp32 for explanations (no autocast) for stability.
- Sensitivity variant (log as a separate method name `gradcam_lastblock`): output of `model.encoder.layers[-1]` `(B,197,768)`, drop token 0, reshape `(B,768,14,14)`. Never replace the primary method after seeing results.
- Tests: shape, finite, determinism, different target classes → different maps, hook removal leaves the model unchanged.

## Attention rollout
torchvision's encoder block calls `self.self_attention(x, x, x, need_weights=False)`, so a normal forward never returns attention. Options:
- Wrap each `EncoderBlock.self_attention` so it calls the underlying `nn.MultiheadAttention` with `need_weights=True, average_attn_weights=False` and stores the `(B, 12, 197, 197)` tensor; or run `ln_1` + `self_attention` manually per block while collecting attention.
- Prove it: rows sum to 1; the model's output with the wrapper equals the original output within tolerance.
Rollout (Abnar & Zuidema 2020): for each layer `A = mean over heads`; `A = 0.5·A + 0.5·I`; row-normalize; `R = A_12 @ … @ A_1`; take `R[:, 0, 1:]` (class token → 196 patches); reshape 14×14; bilinear to 224; per-image min–max. Same procedure for every arm.
Tests: attention dims, class-token index 0, finite values, rollout rows sum ≈ 1.

## Inputs and alignment
- Use the same checkpoint and the same preprocessed input that the arm was trained and tested on. Compute maps for all 3 seeds of each arm.
- Masks: resize with the same geometric transform as the image (nearest-neighbour), binarize, check non-empty. For Arm C confirm no geometric shift against the original (the check from the diffusion skill) before reusing a mask.
- Tiers (never pooled): X1 released HAM10000 masks on the C0 test split (primary), X2 ISIC 2017 expert masks (external; apply each arm's own preprocessing, which for Arms B/C means running the hair screen plus classical/diffusion cleaning on those images), X3 U-Net pseudo-masks only as fallback. Label the tier in every table and filename.

## Metrics (pre-registered)
- **Lesion-overlap ratio**: fraction of the top-k% most salient pixels (k = 20% primary; optional 10% and 30% fixed in advance) that fall inside the lesion mask. Deterministic tie-break (stable sort by value then index).
- **Pointing game**: 1 if argmax pixel is inside the mask.
- **Chance baselines on the same masks**: uniform-random map (fixed seed) and a centred Gaussian map; report the mean lesion area fraction.
- Report on all test images (explanation of the predicted class) and on the correctly classified subset. Do not select images by whether the proposed arm looks better.
- Rows: `mask_tier, image_id, lesion_id, group_id, arm, seed, method, k, overlap, pointing, correct, hair_present`.

## Figures
Per method and arm: original | mask | heatmap | overlay for random, best, and worst cases (success and failure), including cases where the map sits on hair, rulers, gel bubbles, vignette, borders, or background. Save raw maps as `.npy`. Open the figures and note what you actually see.

## Guardrail
Overlap with a lesion mask is a spatial-alignment measure only. It is not evidence of clinically correct reasoning. Say this in the report.

# Compute plan and measured usage

All GPU work runs on Modal A100-80GB PCIe. Modal's current price list was not
queried, so this document reports measured GPU-hours only and does not claim a
dollar figure for GPU time.

Provisional cap recorded in `configs/protocol.json`: `max_spend_usd = 250`.
The user did not supply `MAX_SPEND_USD`, so this is a documented placeholder
(see decision log D10).

## Measured unit costs

| Unit | Measured | Source |
|---|---|---|
| DM-AHR sampling, batch 1 | 35-38 s/image | `dmahr_smoke.json` |
| DM-AHR sampling, batch 8 | 6.04 s/image | `dmahr_smoke.json` |
| DM-AHR sampling, batch 16 | 4.69 s/image | `dmahr_smoke.json` |
| DM-AHR sampling, batch 32 | 4.35 s/image | `dmahr_smoke.json` |
| DM-AHR sampling, batch 64 | 4.10 s/image | `dmahr_smoke.json` |
| DM-AHR training | ~0.055 s/iteration steady state | `dmahr_train.json` |
| ViT-B/16 epoch (7,011 train images, batch 32) | ~17 s | `train_result.json` |
| ViT-B/16 full run (early stop, 25 epochs observed) | ~420 s | `A_seed42/train_result.json` |
| Peak GPU memory, DM-AHR sampling | 1.52 GB | `dmahr_smoke.json` |

## Projected units

| Job | Units | GPU-hours |
|---|---|---|
| DM-AHR training, 20,000 iterations + validation | 1 run | ~0.35 |
| Arm C, hair-present images, batch 32 | 8,993 x 4.35 s | ~10.9 |
| Arm C, held-out synthetic test pairs | 186 x 4.35 s | ~0.22 |
| ViT training, 4 arms x 3 seeds | 12 runs x ~7 min | ~1.4 |
| ViT locked-test evaluation | 12 runs | ~0.2 |
| XAI, 4 arms x 3 seeds on 1,502 test images | 12 runs | ~1.0 |
| **Projected total** | | **~14.1 GPU-hours** |

Batch 64 is only 6% faster per image than batch 32 while using more memory, so
batch 32 is used for Arm C.

## Recorded usage

Every job appends wall-clock seconds and GPU type to
`artifacts/logs/run_registry.jsonl`. The authoritative spend figure is the sum
over that file, not this projection.


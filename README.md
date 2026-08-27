<div align="center">

# ABot-Recon Evaluation

### Reproducible Camera-Pose and Dense-Reconstruction Benchmarks

[English](README.md) | [中文](README_ZH.md)

[![Hugging Face](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=Hugging%20Face&color=7867A8)](https://huggingface.co/acvlab/ABot-Recon) [![ModelScope](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=5578B8)](https://modelscope.cn/models/amap_cvlab/ABot-Recon) [![License](https://img.shields.io/static/v1?label=License&message=Apache-2.0&color=438A68)](LICENSE)

</div>

This branch reproduces the camera-pose and dense-reconstruction evaluations used in the ABot-Recon technical report. The `main` branch contains the minimal inference runtime; this branch adds fixed protocols, third-party adapters, dataset loaders, metrics, and release checks.

Start from the environment described on the `main` branch, switch to `eval`, and install the evaluation extras with `pip install -e ".[eval,loop]"`.

## Evaluation Data

Dataset download and preprocessing follow the public protocols used by the corresponding prior work:

- **KITTI and VBR:** follow [LoGeR](https://loger-project.github.io/) for the long-sequence pose benchmarks;
- **Oxford Spires:** follow [LingBot-Map](https://github.com/Robbyant/lingbot-map), including its Oxford preprocessing pipeline; and
- **7Scenes and TUM-Dynamic:** follow the evaluation-data preparation in [CUT3R](https://github.com/CUT3R/CUT3R).

The datasets are not redistributed in this repository. Expected directory layouts and configurable relative paths are documented in [data/README.md](data/README.md).

## Supported Protocols

### Camera pose

KITTI odometry 00--10, Oxford Spires, and VBR are processed once at stride 1. Each sequence is aligned independently with Umeyama Sim(3), and the evaluator reports ATE RMSE, gap-1 RPE-R, and gap-1 RPE-T.

### Dense reconstruction

| Dataset | Forward frames | Metric frames |
|---|---|---|
| 7Scenes | `seq-01` from all seven scenes, stride 1 | every forwarded frame |
| TUM-Dynamics-Full | all associated RGB frames, stride 1 | every forwarded frame |
| Oxford Spires | all ten rectified sequences, stride 1 | source IDs `0,10,20,...` |

Oxford therefore uses stride-1 model inference and interval-10 point-cloud evaluation. The formal launchers expose `--oxford-metric-interval`, default it to 10, and reject any other value so that a formal run cannot silently deviate from the report protocol.

## Repository Layout

```text
abot_recon/             released ABot-Recon runtime
configs/                Hydra model, dataset, and protocol configuration
datasets/               formal evaluation dataset loaders
interfaces/             model-specific adapters
mv_recon/               reconstruction metrics and protocol validation
relpose/                camera-pose evaluation
scripts/                checked release launchers
third_party_patches/    pinned patches for external official repositories
tests/                  release and integration tests
```

## Third-Party Methods

Third-party checkpoints and source trees are external inputs. Place clean, pinned official checkouts under `third_party/`, or override `OFFICIAL_ROOT` and `HS_ROOT`. Clean official checkouts are supported directly. The pinned patches under `third_party_patches/` reproduce the memory-efficient code paths used in our formal long-sequence evaluation. See [third_party/README.md](third_party/README.md) and [third_party_patches/README.md](third_party_patches/README.md).

Supported streaming baselines include [CUT3R](https://github.com/CUT3R/CUT3R), [TTT3R](https://github.com/Inception3D/TTT3R), [LingBot-Map](https://github.com/Robbyant/lingbot-map), [LongStream](https://github.com/3DAgentWorld/LongStream), [InfiniteVGGT](https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT), [OVGGT](https://github.com/VAISR/OVGGT), [STream3R-window5](https://github.com/NIRVANALAN/STream3R), and [HorizonStream](https://github.com/3DAgentWorld/HorizonStream). Their source and weights retain their original licenses.

## Camera-Pose Evaluation

Run one dataset:

```bash
bash scripts/run_long_pose_protocol.sh \
  --method abot_recon \
  --dataset kitti \
  --stride 1 \
  --gpu 0 \
  --ckpt checkpoints/abot_recon.safetensors \
  --attention-backend auto \
  --out-dir outputs/pose/kitti
```

Complete commands for every method, reset policy, loop mode, and custom image sequence are in [relpose/README.md](relpose/README.md).

## Dense-Reconstruction Evaluation

Run the report protocol on all three datasets:

```bash
bash scripts/run_mv_recon_stride1_suite.sh \
  --method abot_recon \
  --ckpt checkpoints/abot_recon.safetensors \
  --gpu 0 \
  --align sim3 \
  --attention-backend auto \
  --oxford-metric-interval 10 \
  --out-root outputs/mv_recon/abot_recon
```

Dense reconstruction always disables loop closure. For Oxford, every image is still forwarded to the model; `--oxford-metric-interval 10` only selects the TLS metric frames. Use `scripts/run_mv_recon_protocol.sh` for one dataset and `scripts/run_all_models_mv_recon.sh` for the complete model matrix. See [mv_recon/README.md](mv_recon/README.md) for all model commands.

## Reproducibility Contract

Formal launchers record the command, checkpoint SHA256, source revision, resolved input shape and dtype, processed frame count, and runtime manifest. Strict protocol validation rejects wrong strides, Oxford metric IDs, alignment, resize mode, precision, voxel size, or F1 thresholds. The full protocol is documented in [README_ONLINE_EVAL_PROTOCOL.md](README_ONLINE_EVAL_PROTOCOL.md).

Before a long run, validate command construction and Hydra composition:

```bash
bash scripts/run_mv_recon_stride1_suite.sh \
  --method abot_recon \
  --ckpt checkpoints/abot_recon.safetensors \
  --out-root outputs/config_check \
  --config-check
```

## Tests

```bash
pytest -q
ABOT_RECON_REQUIRE_CUROPE=1 pytest -q tests/test_curope_parity.py
pytest -q mv_recon/tests relpose/tests
bash -n scripts/*.sh
sha256sum -c third_party_patches/SHA256SUMS
```

Real-checkpoint tests for both paged and SDPA attention are opt-in:

```bash
ABOT_RECON_CHECKPOINT=checkpoints/abot_recon.safetensors \
ABOT_RECON_IMAGE_DIR=examples/images \
ABOT_RECON_DEVICE=cuda \
pytest -q tests/integration/test_real_checkpoint.py \
  tests/integration/test_eval_adapter_real_checkpoint.py
```

## License

Repository code is released under the [Apache License 2.0](LICENSE), released model weights under [MODEL_LICENSE.md](MODEL_LICENSE.md), and component-level origins and terms by [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Review all upstream model and dataset licenses before redistributing third-party assets.

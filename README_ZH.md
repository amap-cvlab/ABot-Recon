<div align="center">

# ABot-Recon 评测

### 可复现的相机位姿与稠密重建基准

[English](README.md) | [中文](README_ZH.md)

[![Hugging Face](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=Hugging%20Face&color=7867A8)](https://huggingface.co/acvlab/ABot-Recon) [![ModelScope](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=5578B8)](https://modelscope.cn/models/amap_cvlab/ABot-Recon) [![License](https://img.shields.io/static/v1?label=License&message=Apache-2.0&color=438A68)](LICENSE)

</div>

本分支用于复现 ABot-Recon 技术报告中的相机位姿和稠密三维重建实验。`main` 分支提供最小推理代码；本分支额外包含固定评测协议、第三方方法适配器、数据集 loader、指标实现和发布检查。

请先按照 `main` 分支配置环境，再切换到 `eval` 分支并运行 `pip install -e ".[eval,loop]"` 安装评测依赖。

## 评测数据

数据下载与预处理遵循对应工作的公开协议：

- **KITTI 和 VBR：** 按照 [LoGeR](https://loger-project.github.io/) 提供的长序列位姿基准准备；
- **Oxford Spires：** 按照 [LingBot-Map](https://github.com/Robbyant/lingbot-map) 及其 Oxford 预处理流程准备；
- **7Scenes 和 TUM-Dynamic：** 按照 [CUT3R](https://github.com/CUT3R/CUT3R) 的评测数据流程准备。

本仓库不重新分发数据。预期目录结构和可配置相对路径见 [data/README.md](data/README.md)。

## 评测协议

### 相机位姿

KITTI Odometry 00--10、Oxford Spires 和 VBR 均按照时间顺序以 stride 1 单次处理。每条序列独立进行 Umeyama Sim(3) 对齐，并报告 ATE RMSE、gap-1 RPE-R 和 gap-1 RPE-T。

### 稠密三维重建

| 数据集 | 模型前向帧 | 点云指标帧 |
|---|---|---|
| 7Scenes | 7 个场景各自的 `seq-01`，stride 1 | 所有前向帧 |
| TUM-Dynamics-Full | 8 条动态序列的全部关联 RGB 帧，stride 1 | 所有前向帧 |
| Oxford Spires | 10 条 rectified 序列的全部帧，stride 1 | 源帧 ID `0,10,20,...` |

因此 Oxford 是 stride-1 模型推理、interval-10 点云评测。正式启动器暴露 `--oxford-metric-interval`，默认值为 10，并拒绝其他值，避免正式结果静默偏离技术报告协议。

## 目录结构

```text
abot_recon/             ABot-Recon 发布推理代码
configs/                Hydra 模型、数据和协议配置
datasets/               正式评测数据集 loader
interfaces/             各模型评测适配器
mv_recon/               重建指标与协议验证
relpose/                相机位姿评测
scripts/                发布启动脚本
third_party_patches/    针对固定官方 commit 的补丁
tests/                  发布与集成测试
```

## 第三方方法

第三方源码与权重不包含在本仓库中。建议将干净且固定 commit 的官方仓库放在 `third_party/`，也可以通过 `OFFICIAL_ROOT` 和 `HS_ROOT` 覆盖。评测代码支持直接使用干净的官方仓库；`third_party_patches/` 中固定 commit 的补丁用于复现正式长序列评测采用的低内存执行路径。详见 [third_party/README.md](third_party/README.md) 和 [third_party_patches/README.md](third_party_patches/README.md)。

当前支持 [CUT3R](https://github.com/CUT3R/CUT3R)、[TTT3R](https://github.com/Inception3D/TTT3R)、[LingBot-Map](https://github.com/Robbyant/lingbot-map)、[LongStream](https://github.com/3DAgentWorld/LongStream)、[InfiniteVGGT](https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT)、[OVGGT](https://github.com/VAISR/OVGGT)、[STream3R-window5](https://github.com/NIRVANALAN/STream3R) 和 [HorizonStream](https://github.com/3DAgentWorld/HorizonStream)。第三方源码和权重继续遵循其原始许可证。

## 相机位姿评测

单个数据集运行示例：

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

每种方法的完整命令、reset 策略、回环模式和无 GT 自定义序列运行方式见 [relpose/README.md](relpose/README.md)。

## 稠密重建评测

依次运行技术报告中的三个重建数据集：

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

稠密重建默认关闭回环。Oxford 的每一帧仍会参与模型前向；`--oxford-metric-interval 10` 只控制 TLS 点云指标使用的帧。单数据集使用 `scripts/run_mv_recon_protocol.sh`，完整方法矩阵使用 `scripts/run_all_models_mv_recon.sh`。所有模型命令见 [mv_recon/README.md](mv_recon/README.md)。

## 可复现性约束

正式启动器会记录完整命令、checkpoint SHA256、源码 revision、实际输入尺寸与 dtype、处理帧数和 runtime manifest。严格协议验证会拒绝错误的 stride、Oxford metric frame ID、对齐方式、resize 模式、精度、voxel size 和 F1 threshold。完整协议见 [README_ONLINE_EVAL_PROTOCOL.md](README_ONLINE_EVAL_PROTOCOL.md)。

长时间评测前可先检查命令和 Hydra 配置：

```bash
bash scripts/run_mv_recon_stride1_suite.sh \
  --method abot_recon \
  --ckpt checkpoints/abot_recon.safetensors \
  --out-root outputs/config_check \
  --config-check
```

## 测试

```bash
pytest -q
ABOT_RECON_REQUIRE_CUROPE=1 pytest -q tests/test_curope_parity.py
pytest -q mv_recon/tests relpose/tests
bash -n scripts/*.sh
sha256sum -c third_party_patches/SHA256SUMS
```

使用真实权重测试 paged 与 SDPA 后端及 eval adapter：

```bash
ABOT_RECON_CHECKPOINT=checkpoints/abot_recon.safetensors \
ABOT_RECON_IMAGE_DIR=examples/images \
ABOT_RECON_DEVICE=cuda \
pytest -q tests/integration/test_real_checkpoint.py \
  tests/integration/test_eval_adapter_real_checkpoint.py
```

## 许可证

仓库代码采用 [Apache License 2.0](LICENSE)，模型权重许可证见 [MODEL_LICENSE.md](MODEL_LICENSE.md)，第三方组件来源与条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。重新分发第三方源码、权重或数据前，请单独检查对应项目和数据集的许可证。

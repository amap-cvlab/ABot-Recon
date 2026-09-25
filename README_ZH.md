# ABot-Recon 微调

[English](README.md) | [中文](README_ZH.md)

本仓库用于微调已发布的 ABot-Recon 模型，支持 18 个训练数据源，提供加权数据混合、因果训练、EMA、验证和断点续训。示例配置用于微调，不是论文完整多阶段训练流程的复现配置。方法和训练策略请参阅[论文](https://arxiv.org/abs/2608.27529) §3.4 和附录 A。

## 安装

参考环境为 Linux、Python 3.11、PyTorch 2.5.1、CUDA 12.1。请在源码仓库中执行；wheel 不包含示例配置、预处理脚本和测试。

```bash
conda create -n abot-recon python=3.11 -y
conda activate abot-recon
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[train]"
```

建议编译 cuRoPE 以提升训练速度：

```bash
cd abot_recon/modeling/pi3/models/curope
pip install ninja
python setup.py build_ext --inplace
cd -
```

## 预训练权重

从 [Hugging Face](https://huggingface.co/acvlab/ABot-Recon) 或 [ModelScope](https://modelscope.cn/models/amap_cvlab/ABot-Recon) 下载已发布权重：

```bash
hf download acvlab/ABot-Recon abot_recon.safetensors --local-dir checkpoints
```

训练需要本地权重；存放在其他位置时修改配置的 `pretrained`。推理和可选回环依赖请参考[推理仓库](https://github.com/amap-cvlab/ABot-Recon)。微调无需回环资源；未安装这些资源时，调用推理 API 请设置 `loop_closure=False`。

## 数据准备

### 选择预处理方式

| 数据 | 准备方式 |
|---|---|
| 16 个 CUT3R 兼容数据源，包括 DL3DV、VKITTI2 | 按[数据集指南](docs/training_datasets.md#layout-and-sampling)完成全部处理阶段和辅助文件下载。DL3DV、VKITTI2 使用 CUT3R 发布的预处理数据。 |
| ScanNet++ Seq | 使用本仓库的[序列预处理脚本](preprocess/README.md)，而非 CUT3R 的 pair 导出格式。 |
| TartanGround | 使用下方四目录格式，也建议以此作为自有 RGB-D 数据的适配模板。 |

各适配器保留原生磁盘格式；兼容的是**最终预处理产物**，不代表可以直接读取原始下载文件，也不代表清洗规则完全相同。请遵守各数据集原有条款。数据根目录必须是本地或挂载存储的文件系统路径，不支持直接填写 `oss://` URI。

### 加载后的统一接口与几何约定

统一的是适配器**加载后的样本接口**，不是所有数据集的磁盘格式。返回内容包括图像、世界坐标点、有效性掩码、内参、C2W 位姿及数据集／序列标签。深度是构造点的中间数据；返回的 `valid_mask=True` 表示有效。

相机采用 OpenCV 坐标约定（右、下、前），深度为 camera-z，针孔内参以像素为单位。深度和位姿平移必须同尺度，但不是所有数据集都使用米。RGB、深度、掩码需配准，内参需匹配图像分辨率。Camera-only 需要适配器明确支持已标定图像和位姿，不等于任意 RGB 文件夹都可训练。

### 数据处理示例

#### DL3DV

按 [CUT3R 的 DL3DV 说明](https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/docs/preprocess.md#dl3dv)下载并合并 RGB／相机和深度／掩码两部分：

```text
DL3DV_ROOT/<bucket>/<scene>/dense/
  rgb/00000.png
  depth/00000.npy          # float32 (H, W)
  cam/00000.npz            # intrinsic (3, 3), pose (4, 4)，C2W
  sky_mask/00000.png       # >=127 表示无效
  outlier_mask/00000.png   # >=127 表示无效
```

设置 `data.dl3dv.root: DL3DV_ROOT`。这是 CUT3R 兼容格式，不是独立的 ABot 自定义格式。使用其他预处理版本时请检查[黑名单设置](docs/training_datasets.md#bundled-training-blacklists)。

#### TartanGround

```text
TARTANGROUND_ROOT/<sequence>/
  images/00000.jpg
  depths/00000.npy         # float32 (H, W)，camera-z 深度，单位米
  cameras/00000.npz        # camera_intrinsics (3, 3)，camera_pose (4, 4)，C2W
  masks/00000.npy          # bool (H, W)，True 表示无效
```

文件使用相同的补零帧名，相机矩阵为 float32。设置 `data.tartanground.root: TARTANGROUND_ROOT`。

#### 在自有数据上微调

建议以 TartanGround 格式为模板，从 [configs/finetune_custom.yaml](configs/finetune_custom.yaml) 开始，修改 `pretrained`、`output_dir`、`data.tartanground.root`；该示例已禁用其他训练数据源。训练前先检查深度单位、位姿方向，并查看多帧反投影点云是否对齐。

复用该适配器也会采用其米制监督、normal loss、静态场景折返采样，以及由序列名前缀触发的环境采样间隔覆盖。若监督或时序假设不同，应编写独立适配器，详见[自有数据策略](docs/training_datasets.md#fine-tune-on-your-own-rgb-d-sequences)。

使用留出验证时，将同一批序列名**同时**填写到 `data.tartanground.eval_scenes` 和 `data.tartanground.exclude_scenes`；选择验证序列不会自动将其从训练中排除。否则设置 `validation_enabled: false`；只有明确不使用该留出集、希望训练全部序列时，才设置 `exclude_scenes: []`。

### 公共预处理与增强

RGB、深度、掩码和相机几何同步变换，光度增强不作用于 padding。序列内一致增强概率为 0.2，保留水平 FOV 的处理概率为 0.9。`principal_align_skip_prob` 默认 **0**，即示例始终对齐主点，并未启用论文对部分数据集保留偏心主点的配方。参数见配置，方法见论文附录 A.3。

## 生成示例数据

生成 DL3DV 和 TartanGround 格式的小样例，使用下载好的权重执行四步 GPU 冒烟测试：

```bash
python scripts/make_training_sample.py --output examples/training_data --frames 32
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_machines 1 --num_processes 1 \
  --mixed_precision bf16 --dynamo_backend no \
  -m abot_recon.training.cli --config configs/finetune_smoke.yaml
```

该测试只训练生成的 TartanGround 数据，关闭验证和自动续训，输出至 `outputs/finetune_smoke`。每次使用新输出目录。合成样例用于验证流程，不用于评估真实数据质量或学习效果。

## 训练配置

| 配置 | 用途 |
|---|---|
| [finetune.yaml](configs/finetune.yaml) | DL3DV + TartanGround，使用 TartanGround 验证。 |
| [finetune_cut3r.yaml](configs/finetune_cut3r.yaml) | 支持的数据源示例，初始仅启用 TartanAir，关闭验证。 |
| [finetune_custom.yaml](configs/finetune_custom.yaml) | 使用 TartanGround 格式及策略的自有 RGB-D 数据。 |
| [finetune_smoke.yaml](configs/finetune_smoke.yaml) | 四步生成数据冒烟测试。 |

各 YAML 独立，不互相继承；省略字段使用 `TrainConfig` 默认值。默认示例使用 **32 帧、504×280**，每卡 batch size 为 1，采用 AdamW、OneCycleLR、梯度裁剪和 EMA（`0.999`），峰值学习率为 `1e-6`。

数据权重归一化为 batch 抽取概率，不使用的数据源设 `weight: 0`。`data.num_frames` 控制训练片段长度，`max_frames` 是时间位置索引／KV cache 容量，不是采样帧数。

## 启动训练

在所选配置中设置权重、输出目录及启用的数据根目录：

```yaml
pretrained: checkpoints/abot_recon.safetensors
output_dir: outputs/my_finetune
data:
  dl3dv:
    root: /path/to/processed_dl3dv_ours
  tartanground:
    root: /path/to/processed_TartanGround
```

在仓库根目录运行。单卡：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_machines 1 --num_processes 1 \
  --mixed_precision bf16 --dynamo_backend no \
  -m abot_recon.training.cli --config configs/finetune.yaml
```

四卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu \
  --num_machines 1 --num_processes 4 --mixed_precision bf16 --dynamo_backend no \
  -m abot_recon.training.cli --config configs/finetune.yaml
```

新实验请使用**新输出目录**，默认会自动续训。`steps_per_epoch` 表示 micro-batch 数，必须能被 `gradient_accumulation_steps` 整除。任一 rank 出现非有限 loss／梯度或 loss 超过 `max_loss` 时，所有 rank 都丢弃该累积窗口。

## 训练范围与损失

编码器始终冻结。`trainable_scope` 支持 `all`（默认）、`heads`、`rot_correction_only`。Confidence 由 `enable_confidence` 单独控制，默认关闭；开启时权重必须包含 confidence 分支。

训练包含 point、normal、relative-camera loss、旋转修正正则及可选 confidence 监督。公式见论文 §3.4，实现与权重见 [loss.py](abot_recon/training/loss.py) 和配置的 `loss` 部分。Camera-only 样本不计算 point、normal、confidence 监督。

## 验证、权重保存与续训

验证只使用 TartanGround 兼容的留出数据，固定 **128 帧、stride 1 和 6** 两组，与训练帧数无关。采用确定性预处理和逐帧流式推理，每段重置状态；`eval_with_ema: true` 时使用 EMA。增加训练数据源不会自动增加对应验证。`validation_steps: -1` 完整运行两组验证，正值限制每 rank、每组的 batch 数。

`output_dir/metrics.jsonl` 保存训练与验证指标，包括 `val/s1/*`、`val/s6/*` 及平均值 `val/*`。完整检查点保存在 `output_dir/checkpoint-XXXX/`：

- `abot_recon.safetensors`：在线模型；
- `abot_recon_ema.safetensors`：完整 EMA 模型，推荐用于推理；
- `ema.pt`、`trainer_state/`：续训所需 EMA 和训练状态。

EMA 文件仅在启用 EMA 时生成。完整 EMA 模型会自动导出，也可用 [scripts/merge_ema_checkpoint.py](scripts/merge_ema_checkpoint.py) 手动重建。最后一个 epoch 总会保存，已有检查点目录不会覆盖。

自动续训选择最新的完整检查点。也可显式指定：

```yaml
resume: outputs/my_finetune/checkpoint-0004
auto_resume: false
resume_schedule: strict
```

`strict` 拒绝训练计划变更。`restart` 保留模型、优化器和 EMA 状态，但对剩余 epoch 启动新的 OneCycle。从较早检查点分支训练时使用新输出目录。续训恢复训练状态，但**不保证**数据顺序和增强与不中断训练完全一致。

## 测试与代码入口

训练循环见 [abot_recon/training/trainer.py](abot_recon/training/trainer.py)，数据混合见 [data.py](abot_recon/training/data.py)。

```bash
pip install -e ".[train,loop,dev]"
pytest -q
```

GPU／权重相关集成测试需显式启用。验证已发布模型的 forward、loss、backward：

```bash
ABOT_RECON_CHECKPOINT=checkpoints/abot_recon.safetensors \
ABOT_RECON_DEVICE=cuda \
pytest -q tests/integration/test_real_training.py
```

## 引用与许可

使用本训练代码请引用 ABot-Recon 论文。项目原创代码采用 Apache-2.0，另有标注的除外。第三方衍生组件保留上游条款，包括采用 CC BY-NC-SA 4.0 的 CUT3R 衍生数据适配器；详见 [LICENSE](LICENSE) 和[第三方声明](THIRD_PARTY_NOTICES.md)。

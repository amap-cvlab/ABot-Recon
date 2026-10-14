# 离线可视化

[English](README.md) | 简体中文

将 ABot-Recon 结果渲染为地面跟随视角点云视频及鸟瞰图片：
**推理 → 生成天空 mask → 渲染**。这些脚本不会重新推理或修改重建位姿。
以下命令均在仓库根目录执行。

## 环境准备

先按[主 README](../README_ZH.md) 配置推理环境与权重，再安装：

```bash
pip install 'opencv-python>=4.8' 'scipy>=1.10'
pip install onnxruntime-gpu  # 用于天空分割；仅用 CPU 时替换为 onnxruntime。
```

视频输出需要 `ffmpeg`。推荐 CUDA；CPU 运行使用 `--device cpu`。无需显示服务器或 OpenGL。

## 快速开始

### 1. 保存推理结果

如果已有包含稠密点图、RGB 和 confidence 的结果，可跳过此步。

```bash
python -m abot_recon.cli \
  --image-dir examples/images --output-dir outputs/inference \
  --stride 5 --dense-stride 1 --confidence-threshold 0 \
  --save-local-points --save-confidence --no-loop-closure
```

此示例关闭回环，开启方法见主 README。建议推理时将 confidence 阈值设为 0，方便渲染时再调整过滤强度。

### 2. 生成天空 mask

```bash
python visualization/generate_sky_masks.py \
  --result-dir outputs/inference --download-model --device cuda:0
```

命令会将 [SkySeg](https://huggingface.co/JianyuanWang/skyseg/tree/main) 下载到 `checkpoints/skyseg.onnx`，并将 mask 保存到 `outputs/inference/sky_masks`。请遵守模型许可。已有权重时，将 `--download-model` 替换为 `--model-path PATH`；需要镜像时，在命令前加 `HF_ENDPOINT=https://hf-mirror.com`。

默认使用 `--non-sky-threshold 0.3 --sky-dilate 3`，膨胀半径为推理分辨率下的 3 像素。分数按每张图归一化，并非校准后的概率。更改参数后需重新生成 mask，并保留与 PNG 配套的 `manifest.json`。

### 3. 渲染

```bash
python visualization/render.py \
  --result-dir outputs/inference --output-dir outputs/visualization \
  --device cuda:0
```

**默认开启天空过滤**。mask 缺失或不匹配时会报错。可用 `--sky-mask-dir PATH` 指定其他位置，或用 `--no-sky` 关闭。渲染脚本不会调用或下载天空分割模型。

默认在推理分辨率下生成点云。如果需要上采样到原图分辨率，并同时输出对应的 RGB 视频：

```bash
python visualization/render.py \
  --result-dir outputs/inference --output-dir outputs/dense_visualization \
  --image-dir examples/images --image-stride 5 \
  --resolution source --upsample bilinear --pixel-step 1 \
  --outputs video trajectory pointcloud rgb --device cuda:0
```

原图顺序和 `--image-start/--image-end/--image-stride` 必须与推理完全一致。RGB 保持原图宽高比。`--width/--height` 控制渲染视频分辨率，不是点云采样分辨率。

## 时序与采样

采样以**推理帧**而非原视频帧为单位。例如 `--view-step 2 --cloud-step 1` 表示首帧之后，每个视频帧推进两个位姿，并加入中间两帧的稠密点云；只能使用推理时已保存的稠密帧。播放帧率不自动匹配原视频，不使用图像插帧。

输入足够长时，默认输出 **600 帧地面 + 120 帧转场，共 24 秒**。转场时冻结重建时间，仅移动渲染相机并逐渐淡出点云，RGB 保持最后一帧。用 `--start/--end` 选择片段；再加 `--global-scope segment` 可将鸟瞰图片与轨迹历史限制到该片段，默认使用整个序列。

稠密渲染时可增大 `--cloud-step` 或 `--pixel-step`，或限制 `--history` 来控制内存；仅增大 `--view-step` 不会减少加入的点云。超出 `--max-resident-points` 预算会报错，不会静默降低密度。

## 天空分割完整参数

以下为 `generate_sky_masks.py` 的全部参数，N 表示推理帧总数。可选的原图输入必须对应**完整推理序列**，不能只提供待渲染片段。输出 mask 始终采用推理分辨率。

| 参数 | 默认值 | 设置说明 |
|---|---|---|
| `--result-dir` | 必填 | 发布版推理结果目录。 |
| `--output-dir` | `RESULT_DIR/sky_masks` | mask PNG 及其 manifest 的输出目录。 |
| `--image-dir` | 未设置 | 原图目录；不提供时使用已保存的推理 RGB。与 `--frame-list` 互斥。 |
| `--image-start` | `0` | 按 stride 采样前的原图起始索引，需与推理一致。 |
| `--image-end` | 输入末尾 | 原图结束索引，不包含该索引，需与推理一致。 |
| `--image-stride` | `1` | 原图采样间隔，需与推理一致；例如推理使用 `--stride 5`，这里也设为 5。 |
| `--frame-list` | 未设置 | 按顺序列出恰好 N 张原图路径的 JSON。相对路径以 JSON 所在目录为基准；不会再应用 image-start/end/stride。 |
| `--model-path` | `<repo>/checkpoints/skyseg.onnx` | ONNX 权重路径；默认相对于脚本所在仓库，而非当前工作目录。 |
| `--download-model` | `false` | 权重缺失时从 `JianyuanWang/skyseg` 下载，支持 `HF_ENDPOINT`；已有权重直接复用。 |
| `--device` | `cpu` | `cpu` 或 `cuda:N`。指定 CUDA 时必须可用，不会静默回退到 CPU。 |
| `--non-sky-threshold` | `0.3` | 剔除归一化非天空分数 ≤ 此值的像素，取值 [0,1]；越大通常剔除越多，设为 0 不等于关闭天空过滤。 |
| `--sky-dilate` | `3` | 剔除 mask 的膨胀半径，以推理分辨率像素计；3 对应 7×7 椭圆核，0 关闭膨胀。 |
| `--overwrite` | `false` | 替换本脚本已生成的 mask 和 manifest，保留无关文件；修改参数后重新生成时可使用。 |
| `-h`、`--help` | — | 显示命令行帮助并退出。 |

## 渲染完整参数

以下为 `render.py` 的全部参数。N 表示推理帧总数。距离和深度采用重建尺度，不一定是米；除非另有说明，像素大小均指输出画面像素。

### 输入与帧选择

| 参数 | 默认值 | 设置说明 |
|---|---|---|
| `--result-dir` | 必填 | 发布版推理结果目录。 |
| `--output-dir` | 必填 | 渲染输出目录，必须与推理结果目录不同。 |
| `--poses` | `final` | 选择已保存的 `camera_poses.npy`（`final`）、`camera_poses_noloop.npy`（`noloop`）或 `camera_poses_loop.npy`（`loop`）。对应文件必须存在，不会重新执行回环优化。 |
| `--start` | `0` | 地面片段的起始推理帧索引。 |
| `--end` | `N` | 结束推理帧索引，不包含该索引；要求 `0 ≤ start < end ≤ N`。 |
| `--ground-frames` | `600` | 地面视频最大帧数；输入不足时不会重复补帧。仅不输出 `video`/`rgb` 时允许设为 0，不表示不限帧数。 |
| `--view-step` | `1` | 每个输出帧推进的推理帧数：第 k 个视角对应 `start + k × view-step`。 |
| `--cloud-step` | `1` | 选择满足 `(index − start) % cloud-step == 0` 的已保存稠密帧；截至当前视角的所有命中帧都会加入点云。 |
| `--global-scope` | `all` | 鸟瞰范围：`all` 为全序列，`segment` 为 `[start,end)`，不受 `--ground-frames` 截短。视频轨迹历史分别从 0/start 开始，始终只画到当前视角。 |
| `--image-dir` | 未设置 | 原图目录，原分辨率点云或 RGB 输出需要提供；native 点云可不提供。选择结果需对应完整推理序列，而非仅渲染片段。与 `--frame-list` 互斥。 |
| `--image-start` | `0` | 按 stride 采样前的原图起始索引，需与推理一致。 |
| `--image-end` | 输入末尾 | 原图结束索引，不包含该索引，需与推理一致。 |
| `--image-stride` | `1` | 原图采样间隔，需与推理一致，不是 `--view-step`。 |
| `--frame-list` | 未设置 | 按顺序列出恰好 N 张原图路径的 JSON。相对路径以 JSON 所在目录为基准；不再应用 image-start/end/stride。 |

### 点云与过滤

地面与鸟瞰点云共用置信度、深度、边缘和天空过滤，这些过滤发生在原图分辨率上采样之前。历史限制与视锥剔除仅作用于地面累计点云，发生在上采样之后。采样间隔必须为正整数。

| 参数 | 默认值 | 设置说明 |
|---|---|---|
| `--resolution` | `native` | `native` 使用保存的推理分辨率；`source` 使用原图分辨率，需要原图输入。对地面和鸟瞰点云均生效。 |
| `--upsample` | `bilinear` | `bilinear` 双线性或 `nearest` 最近邻，仅在 source 模式生效。插值深度和射线，无效区域仍被剔除。 |
| `--pixel-step` | `1` | 地面点云在所选分辨率下沿两个方向采样；1 使用所有有效像素，2 约保留 1/4。 |
| `--confidence` | `0.15` | 保留 confidence ≥ 阈值的点，取值 [0,1]。0 关闭该过滤；正阈值要求存在 `confidence.pt`。 |
| `--min-depth` | `0` | 保留相机局部 Z 大于此下限的点；非正或无效深度始终剔除。 |
| `--max-depth` | `6` | 相机局部 Z 上限；0 仅关闭上限，否则必须大于 `--min-depth`。 |
| `--depth-edge` | `0.15` | 局部 `(最大深度 − 最小深度) / Z` 超过阈值时剔除；0 关闭边缘过滤。 |
| `--edge-kernel` | `3` | 深度边缘检测邻域宽度，须为正奇数，以推理分辨率像素计。 |
| `--sky`、`--no-sky` | `true` | 开启/关闭预生成天空 mask 过滤；开启时必须有完整、匹配的 mask。 |
| `--sky-mask-dir` | `RESULT_DIR/sky_masks` | 上述脚本生成的 mask PNG 与 manifest 所在目录。 |
| `--history` | `0` | 保留最近 N 个推理帧的点云；0 不限制历史长度，但视锥剔除仍生效。不限制鸟瞰图或相机历史。 |
| `--prune-every` | `10` | 每 N 个输出帧永久剔除地面视角视锥外点云，首帧即执行；0 关闭。转场期间暂停剔除。 |
| `--bird-cloud-step` | `5` | 鸟瞰点云的推理帧采样间隔，以所选全局范围的起始帧为采样原点。 |
| `--bird-pixel-step` | `8` | 鸟瞰点云在所选分辨率下沿两个方向的像素采样间隔，独立于 `--pixel-step`。 |

### 渲染相机与外观

只平滑虚拟渲染相机，不修改重建位姿。渲染光心为 `C − dF + d × height-ratio × U`，注视点为 `C + d × look-ahead × F`，其中 C/F/U 为平滑后的相机光心/前向/上方向。前向取预测相机方向，不一定等于轨迹切线。

| 参数 | 默认值 | 设置说明 |
|---|---|---|
| `--width` | `1920` | 渲染视频/PNG 宽度，须为正整数，输出视频时须为偶数。不改变 RGB 对照视频尺寸。 |
| `--height` | `1080` | 渲染视频/PNG 高度，须为正整数，输出视频时须为偶数。不改变 RGB 对照视频尺寸。 |
| `--background` | `32 32 32` | 视频与不透明末帧的 RGB 背景，三个值均在 [0,255]；鸟瞰 PNG 仍透明。 |
| `--point-size` | `2` | 点的方形绘制直径，以输出像素计，须为正整数。 |
| `--ground-fov` | `60` | 地面渲染相机的垂直视场角，单位为度，严格大于 1 且小于 175。 |
| `--bird-fov` | `60` | 鸟瞰相机的垂直视场角，单位为度，严格大于 1 且小于 175；转场逐渐过渡到此值。 |
| `--bird-margin` | `1.12` | 共用鸟瞰相机拟合轨迹时的留白系数，须 ≥ 1，越大留白越多。 |
| `--follow-distance` | `0` | 后退距离 d；正值固定距离并忽略自动距离计算参数，0 使用自动距离。 |
| `--distance-ratio` | `0.12` | 自动 d = 此正比例 × 全轨迹半径，再按下方上下限截断。半径为到中位光心距离的第 90 百分位，最小取 0.5。 |
| `--distance-min` | `0.6` | 自动跟随距离的下限，须大于 0。 |
| `--distance-max` | `2` | 自动跟随距离的上限，须不小于下限。 |
| `--height-ratio` | `0.2` | 相对平滑光心沿局部上方向抬高的距离与 d 的比例，须非负。 |
| `--look-ahead` | `0.35` | 注视点沿局部前向的偏移与 d 的比例；增大后看得更靠前。 |
| `--smooth-radius` | `6` | 光心与前向平滑的推理帧半径，通常使用 `2r+1` 帧；0 仅使用当前帧。 |
| `--smooth-frames` | `6` | 渲染相机上方向平滑窗口的帧数，须为正整数；1 仅使用当前帧。 |
| `--frustum-step` | `1` | 每 N 个推理帧索引画一个视锥，从全序列索引 0 起计；0 隐藏视锥，不隐藏轨迹。 |
| `--frustum-size` | `0.05` | 显示视锥沿重建相机局部 Z 轴的长度，须为正值。 |
| `--trajectory-width` | `2` | 轨迹连线宽度，以输出像素计，须为正整数。 |
| `--frustum-width` | `1` | 视锥线宽，以输出像素计，须为正整数。 |
| `--center-size` | `2` | 相机光心方形标记大小，以输出像素计；0 隐藏光心标记。 |

### 输出与运行

| 参数 | 默认值 | 设置说明 |
|---|---|---|
| `--fps` | `30` | 两个输出视频的播放帧率，须为正值，独立于原视频帧率。 |
| `--transition-frames` | `120` | 额外的鸟瞰转场帧数；0 关闭。转场冻结重建时间，只移动渲染相机并淡出点云。 |
| `--outputs` | `video trajectory pointcloud` | 可选一个或多个：`video`、`trajectory`、`pointcloud`、`rgb`。`rgb` 需要原图；仅选 `trajectory pointcloud` 可只输出 PNG、不生成视频。 |
| `--device` | `cpu` | 点云光栅化设备：`cpu` 或 `cuda:N`。 |
| `--chunk-points` | `200000` | 每批光栅化的最大点数，用于控制临时工作内存，不降低点云密度。 |
| `--max-resident-points` | `20000000` | 地面点云驻留点数预算，不是字节数。加入点云时就会检查，包括历史/视锥清理之前；超出则报错。 |
| `--crf` | `20` | H.264 质量参数，范围 0–51；通常越低质量越高、文件越大。 |
| `--preset` | `medium` | 编码预设：`ultrafast`、`fast`、`medium` 或 `slow`；越慢通常压缩效果越好。 |
| `--overwrite` | `false` | 替换输出目录内本脚本指定的渲染文件，不修改推理结果或无关文件。 |
| `-h`、`--help` | — | 显示命令行帮助并退出。 |

## 输出文件

| 文件 | 内容 |
|---|---|
| `ground_pointcloud.mp4` | 地面跟随视角点云视频，可包含鸟瞰转场 |
| `ground_rgb.mp4` | 对应的原图 RGB 视频，仅在 `--outputs ... rgb` 时生成 |
| `last_frame.png` | 渲染的最后一帧；启用转场时还生成 `last_frame_transparent.png` |
| `birdseye_trajectory.png` | 指定范围内的完整轨迹与相机，透明背景 |
| `birdseye_pointcloud.png` | 指定范围内的完整点云，透明背景 |
| `render_manifest.json` | 配置、帧映射与完成状态 |
| `filter_stats.jsonl` | 各渲染阶段的逐帧过滤统计 |
| `frustum_intrinsics.json` | 用于显示视锥的拟合内参与拟合误差 |

鸟瞰图片与转场终点共用同一相机，便于对齐叠加。请使用新的输出目录，或显式传入 `--overwrite`；推理结果不会被修改。

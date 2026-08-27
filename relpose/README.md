# Camera Pose Evaluation

This directory provides the formal camera trajectory evaluation for online and streaming reconstruction methods.

## Evaluation Protocol

| `--dataset` | Sequences | Evaluated frames |
|---|---|---|
| `kitti` | KITTI Odometry 00--10 | all available frames, stride 1 |
| `vbr` | seven long VBR sequences | all available frames, stride 1 |
| `oxford` | ten processed and rectified Oxford Spires sequences | all available frames, stride 1 |

Each sequence is aligned independently to the ground-truth trajectory using Umeyama Sim(3) alignment. We report ATE RMSE, gap-1 translational RPE, and gap-1 rotational RPE in degrees. Formal runs must complete every sequence and record the checkpoint, source revision, resolved input shape and dtype, processed frame count, and runtime configuration in the run manifest.

In the primary comparison, TTT3R follows its official reset-100 protocol, while `CUT3R w/ reset` applies the same protocol as a controlled ablation; no-reset results obtained with `--reset-interval 0` are reported separately. HorizonStream follows its official minimal inference setting with FP16 autocast.

## Method Commands

Run all commands from the repository root. Supported methods are `cut3r`, `ttt3r`, `longstream`, `infinitevggt`, `ovggt`, `stream3r_window5`, `horizon`, `lingbot`, and `abot_recon`. The following examples evaluate KITTI; replace `--dataset kitti` with `vbr` or `oxford` for the other formal benchmarks.

```bash
OUT=/path/to/pose_results
mkdir -p "$OUT/logs"
```

### CUT3R

CUT3R uses reset-100 below. Set `--reset-interval 0` for the no-reset ablation.

```bash
nohup bash scripts/run_long_pose_protocol.sh --method cut3r --dataset kitti \
  --stride 1 --gpu 0 --reset-interval 100 --out-dir "$OUT" \
  --ckpt checkpoints/cut3r/cut3r_512_dpt_4_64.pth \
  > "$OUT/logs/cut3r_kitti.log" 2>&1 &
```

### TTT3R

```bash
nohup bash scripts/run_long_pose_protocol.sh --method ttt3r --dataset kitti \
  --stride 1 --gpu 0 --reset-interval 100 --out-dir "$OUT" \
  --ckpt checkpoints/cut3r/cut3r_512_dpt_4_64.pth \
  > "$OUT/logs/ttt3r_kitti.log" 2>&1 &
```

### LongStream

```bash
nohup bash scripts/run_long_pose_protocol.sh --method longstream --dataset kitti \
  --stride 1 --gpu 0 --out-dir "$OUT" \
  --ckpt checkpoints/longstream/50_longstream.pt \
  > "$OUT/logs/longstream_kitti.log" 2>&1 &
```

### InfiniteVGGT

```bash
nohup bash scripts/run_long_pose_protocol.sh --method infinitevggt --dataset kitti \
  --stride 1 --gpu 0 --out-dir "$OUT" \
  --ckpt checkpoints/infinitevggt/checkpoints.pth \
  > "$OUT/logs/infinitevggt_kitti.log" 2>&1 &
```

### OVGGT

```bash
nohup bash scripts/run_long_pose_protocol.sh --method ovggt --dataset kitti \
  --stride 1 --gpu 0 --out-dir "$OUT" \
  --ckpt checkpoints/infinitevggt/checkpoints.pth \
  > "$OUT/logs/ovggt_kitti.log" 2>&1 &
```

### STream3R-window5

```bash
nohup bash scripts/run_long_pose_protocol.sh --method stream3r_window5 --dataset kitti \
  --stride 1 --gpu 0 --out-dir "$OUT" \
  --ckpt checkpoints/stream3r/model.safetensors \
  > "$OUT/logs/stream3r_window5_kitti.log" 2>&1 &
```

### HorizonStream

```bash
nohup bash scripts/run_long_pose_protocol.sh --method horizon --dataset kitti \
  --stride 1 --gpu 0 --horizon-amp-dtype fp16 --out-dir "$OUT" \
  --ckpt checkpoints/horizonstream/HorizonStream.pt \
  > "$OUT/logs/horizon_kitti.log" 2>&1 &
```

To evaluate HorizonStream with loop closure, use `--horizon-loop-mode both`. This preserves the no-loop result and produces a separate loop-refined trajectory. The `auto` preset selects the repository preset for KITTI and VBR and the generic configuration for Oxford.

```bash
nohup bash scripts/run_long_pose_protocol.sh --method horizon --dataset kitti \
  --stride 1 --gpu 0 --horizon-amp-dtype fp16 \
  --horizon-loop-mode both --horizon-loop-preset auto --out-dir "$OUT" \
  --ckpt checkpoints/horizonstream/HorizonStream.pt \
  --horizon-salad-ckpt checkpoints/loop/dino_salad.ckpt \
  --horizon-salad-dino-weights checkpoints/loop/dinov2_vitb14_pretrain.pth \
  > "$OUT/logs/horizon_kitti_both.log" 2>&1 &
```

### LingBot-Map

```bash
nohup bash scripts/run_long_pose_protocol.sh --method lingbot --dataset kitti \
  --stride 1 --gpu 0 --out-dir "$OUT" \
  --ckpt checkpoints/lingbot-map/lingbot-map.pt \
  > "$OUT/logs/lingbot_kitti.log" 2>&1 &
```

### ABot-Recon

ABot-Recon forwards the streaming model once, saves the no-loop trajectory, and produces the loop-refined trajectory as a separate post-processing result. Add `--abot-recon-loop-mode off` to evaluate no-loop inference only.

```bash
nohup bash scripts/run_long_pose_protocol.sh --method abot_recon --dataset kitti \
  --stride 1 --gpu 0 --out-dir "$OUT" \
  --ckpt checkpoints/abot_recon.safetensors \
  > "$OUT/logs/abot_recon_kitti.log" 2>&1 &
```

The no-loop and loop outputs are written to:

```text
OUT/abot_recon_kitti_s1_sim3/abot_recon_kitti_s1_sim3/kitti-long/
OUT/abot_recon_kitti_s1_sim3/abot_recon_kitti_s1_sim3_loop/kitti-long/
```
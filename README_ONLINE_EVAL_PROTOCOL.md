# Formal Evaluation Protocol for Online Models

This document defines the formal evaluation protocol supported by the release. Debug outputs must not be mixed with formal CSV files, and model-native input preprocessing must not be modified to improve individual metrics.

## 1. Tasks and Datasets

| Task | Dataset | Forward frames | Geometry metric frames |
|---|---|---:|---:|
| Point cloud | 7Scenes | `seq-01` from each scene, stride 1 | all valid-depth frames |
| Point cloud | TUM-Dynamics-Full | all associated RGB frames from eight dynamic sequences, stride 1 | all valid-depth frames |
| Point cloud | Oxford Spires | ten rectified sequences, stride 1 | `frame_ids=0,10,20,...` |
| Pose | KITTI | all sequences, stride 1 | all forwarded frames |
| Pose | Oxford Spires | all sequences, stride 1 | all forwarded frames |
| Pose | VBR | all sequences, stride 1 | all forwarded frames |

The 7Scenes and TUM poses are used only to transform predicted local geometry into the world frame and are not included in the formal pose summary. Oxford interval-10 applies only to dense geometry; pose metrics use the complete stride-1 trajectory.

## 2. Model Forward Pass

| Method | Formal mode | Forward precision |
|---|---|---|
| CUT3R | recurrent; reset and no-reset pose variants | FP32 |
| TTT3R | gated recurrent; reset and no-reset pose variants | FP32 |
| LingBot-Map | official streaming | BF16 on Ampere or newer, otherwise FP16 |
| LongStream | official batch-refresh/causal | FP32 outer context with official BF16 attention |
| InfiniteVGGT | official online | BF16 autocast |
| OVGGT | official online | BF16 autocast |
| STream3R-window5 | official window mode with native world points | official mixed FP32/BF16 |
| HorizonStream | official online | FP16 autocast |
| ABot-Recon | causal streaming with paged KV cache | BF16 autocast |

Each method retains its official RGB loader, resize/crop/pad policy, and normalization. The evaluator does not force a common network input shape. Runtime manifests record the resolved input shape, dtype, frame count, and source revision.

## 3. Formal Input Shapes

| Method | 7Scenes/TUM | Oxford | KITTI/VBR pose |
|---|---:|---:|---:|
| ABot-Recon | `280x504` | `280x504` | `280x504` |
| LingBot-Map | `434x574` | `378x518` | `280x504` |
| HorizonStream | `378x518` | `378x518` | official model loader |
| CUT3R/TTT3R | official 512 resize/crop | official 512 resize/crop | official 512 resize/crop |
| InfiniteVGGT/OVGGT | official crop path | official crop path | official crop path |
| LongStream | official demo loader | official demo loader | official demo loader |
| STream3R-window5 | official crop path | official crop path | official crop path |

## 4. Dense-Output Mapping

- XYZ point maps from ABot-Recon, CUT3R, TTT3R, InfiniteVGGT, OVGGT, and STream3R are mapped to the ground-truth grid with nearest-neighbor resampling.
- LingBot-Map and HorizonStream use predicted depth and intrinsics: depth is resized with nearest-neighbor interpolation, intrinsics are scaled to the target resolution, and points are back-projected to the camera frame before transformation to the world frame with the predicted pose.
- Metrics use only the observed image region. Areas outside crop/pad support are removed from both prediction and ground-truth masks.
- A prediction is valid when it is finite and has `z/depth > 1e-4`.

## 5. Alignment and Reconstruction Metrics

Formal point-cloud comparison uses Sim(3) for every method:

1. Estimate Umeyama Sim(3) from pixels valid in both prediction and ground truth.
2. Apply the protocol-specific voxel downsampling.
3. Run point-to-point rigid ICP for at most 20 iterations.
4. Compute Accuracy, Completeness, Chamfer Distance, and F1.

| Dataset | Voxel size | ICP threshold | F1 thresholds |
|---|---:|---:|---:|
| 7Scenes | `4/512 m` | `0.1 m` | `0.25 m` |
| TUM-Dynamics-Full | `4/512 m` | `0.1 m` | `0.25 m` |
| Oxford Spires | `0.05 m` | `0.5 m` | `4 m` |

Alignment-only masks do not alter the final metric point sets:

- Oxford uses `GT depth <= 80 m` to estimate Umeyama alignment.
- HorizonStream additionally uses `prediction depth <= 40 m` for alignment.
- Final Accuracy, Completeness, Chamfer Distance, and F1 use all valid prediction and ground-truth points within the observed field of view.

## 6. Pose Metrics

Pose evaluation uses every forwarded frame, aligns each sequence independently with Sim(3), and reports:

- ATE RMSE;
- delta-1 RPE translation; and
- delta-1 RPE rotation in degrees.

CUT3R/TTT3R reset and no-reset variants must be named and averaged separately. TTT3R follows its official overlap-reset protocol: a boundary frame triggers reset and is repeated as the first frame of the next segment; duplicate output is removed before the segment trajectories are composed. CUT3R has no official periodic-reset entry, so `CUT3R w/ reset` is a controlled variant using the same TTT3R protocol rather than a CUT3R default.

The formal HorizonStream baseline uses no loop closure. The optional `--horizon-loop-mode both` preserves the no-loop result and separately runs SALAD retrieval, candidate-window re-inference, and SE(3) PGO. KITTI and VBR use the corresponding repository presets; Oxford uses the official generic configuration. Loop caches are reused only when retrieval and PGO settings are unchanged.

## 7. Formal Entry Points

- Single-dataset point cloud: `scripts/run_mv_recon_protocol.sh`
- Three-dataset point cloud for one model: `scripts/run_mv_recon_stride1_suite.sh`
- Full point-cloud model matrix: `scripts/run_all_models_mv_recon.sh`
- Pose for one model: `scripts/run_long_pose_protocol.sh`
- Full pose model matrix: `scripts/run_all_models_pose.sh`

### Single-Model Point-Cloud Command

All models use the same entry point. `METHOD` is one of `abot_recon`, `lingbot`, `horizon`, `cut3r`, `ttt3r`, `longstream`, `infinitevggt`, `ovggt`, or `stream3r_window5`; `DATASET` is `7scenes`, `tum`, or `oxford`.

```bash
nohup bash scripts/run_mv_recon_protocol.sh \
  --method METHOD --dataset DATASET --gpu 0 \
  --ckpt CHECKPOINT --out-dir OUTPUT \
  > OUTPUT.log 2>&1 &
```

This launcher fixes `pi3_pc_world_source=points` and `abot_recon_loop_enabled=false`, so ABot-Recon reconstruction metrics use direct world points and are unaffected by the default pose-loop policy. Default third-party checkpoints and complete three-dataset commands are defined in `scripts/run_all_models_mv_recon.sh`; model-specific pose commands are documented in `relpose/README.md`. Formal runs record the checkpoint SHA256, source revision, resolved parameters, and runtime manifest. Third-party checkouts must use the matching patches under `third_party_patches/` and pass `SHA256SUMS` verification.

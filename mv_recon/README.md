# Multi-view reconstruction evaluation

This directory evaluates online multi-view reconstruction under three fixed protocols. All methods use Sim(3) alignment for the reported point-cloud metrics.

## Datasets

| Dataset | Forward frames | Metric frames | Reported F1 |
|---|---|---|---|
| 7Scenes | `seq-01` from each of the seven scenes, stride 1 | all forwarded frames |  0.25 m |
| TUM-Dynamics-Full | all associated RGB frames from eight sequences, stride 1 | all forwarded frames | 0.25 m |
| Oxford Spires | all rectified frames from ten sequences, stride 1 | frame IDs `0,10,20,...` | 4 m |

Dataset locations are configured in `configs/data/mv_recon.yaml`. The detailed preprocessing, alignment, observed-FOV, voxelization, and metric protocol is documented in `../README_ONLINE_EVAL_PROTOCOL.md`.

## Methods

The public evaluation surface contains:

- LingBot-Map
- HorizonStream
- CUT3R
- TTT3R
- STream3R-window5 (native world-point output)
- LongStream
- InfiniteVGGT
- OVGGT
- ABot-Recon

Model registrations and preprocessing options are defined in `configs/model/default.yaml`. Official third-party source checkouts are selected with `OFFICIAL_ROOT`; HorizonStream uses `HS_ROOT`. ABot-Recon is imported from this repository and does not require a separate source checkout.

## Run the three-dataset suite

Run commands from the repository root. Set these paths for the local machine:

```bash
cd /path/to/abot_recon
export PY=/path/to/python
export OFFICIAL_ROOT=/path/to/official-model-checkouts
export HS_ROOT=/path/to/HorizonStream
export RESULT_ROOT=/path/to/results
mkdir -p "$RESULT_ROOT/logs"
```

Each command below runs 7Scenes, TUM-Dynamics-Full, and Oxford Spires sequentially on one GPU. Replace the checkpoint paths before launching.

### LingBot-Map

```bash
nohup env PY="$PY" OFFICIAL_ROOT="$OFFICIAL_ROOT" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method lingbot --ckpt /path/to/lingbot-map.pt --gpu 0 \
  --align sim3 --out-root "$RESULT_ROOT/lingbot" \
  >"$RESULT_ROOT/logs/lingbot.log" 2>&1 &
```

### HorizonStream

```bash
nohup env PY="$PY" OFFICIAL_ROOT="$OFFICIAL_ROOT" HS_ROOT="$HS_ROOT" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method horizon --ckpt /path/to/HorizonStream.pt --gpu 0 \
  --align sim3 --out-root "$RESULT_ROOT/horizon" \
  >"$RESULT_ROOT/logs/horizon.log" 2>&1 &
```

### CUT3R

```bash
nohup env PY="$PY" OFFICIAL_ROOT="$OFFICIAL_ROOT" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method cut3r --ckpt /path/to/cut3r_512_dpt_4_64.pth --gpu 0 \
  --align sim3 --out-root "$RESULT_ROOT/cut3r" \
  >"$RESULT_ROOT/logs/cut3r.log" 2>&1 &
```

### TTT3R

```bash
nohup env PY="$PY" OFFICIAL_ROOT="$OFFICIAL_ROOT" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method ttt3r --ckpt /path/to/cut3r_512_dpt_4_64.pth --gpu 0 \
  --align sim3 --out-root "$RESULT_ROOT/ttt3r" \
  >"$RESULT_ROOT/logs/ttt3r.log" 2>&1 &
```

### STream3R-window5

```bash
nohup env PY="$PY" OFFICIAL_ROOT="$OFFICIAL_ROOT" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method stream3r_window5 --ckpt /path/to/stream3r/model.safetensors --gpu 0 \
  --align sim3 --stream3r-chunk-size 1 \
  --out-root "$RESULT_ROOT/stream3r_window5" \
  >"$RESULT_ROOT/logs/stream3r_window5.log" 2>&1 &
```

### LongStream

```bash
nohup env PY="$PY" OFFICIAL_ROOT="$OFFICIAL_ROOT" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method longstream --ckpt /path/to/50_longstream.pt --gpu 0 \
  --align sim3 --out-root "$RESULT_ROOT/longstream" \
  >"$RESULT_ROOT/logs/longstream.log" 2>&1 &
```

### InfiniteVGGT

```bash
nohup env PY="$PY" OFFICIAL_ROOT="$OFFICIAL_ROOT" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method infinitevggt --ckpt /path/to/checkpoints.pth --gpu 0 \
  --align sim3 --out-root "$RESULT_ROOT/infinitevggt" \
  >"$RESULT_ROOT/logs/infinitevggt.log" 2>&1 &
```

### OVGGT

```bash
nohup env PY="$PY" OFFICIAL_ROOT="$OFFICIAL_ROOT" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method ovggt --ckpt /path/to/checkpoints.pth --gpu 0 \
  --align sim3 --out-root "$RESULT_ROOT/ovggt" \
  >"$RESULT_ROOT/logs/ovggt.log" 2>&1 &
```

### ABot-Recon

```bash
nohup env PY="$PY" \
  bash scripts/run_mv_recon_stride1_suite.sh \
  --method abot_recon --ckpt checkpoints/abot_recon.safetensors --gpu 0 \
  --attention-backend auto \
  --align sim3 --out-root "$RESULT_ROOT/abot_recon" \
  >"$RESULT_ROOT/logs/abot_recon.log" 2>&1 &
```

By default the formal suite saves metrics and runtime manifests but not PLYs or trajectory figures. Add `--save-ply true` or `--save-traj true` when qualitative artifacts are required. For a single dataset, use `scripts/run_mv_recon_protocol.sh --help`.

# ABot-Recon Fine-tuning

[English](README.md) | [中文](README_ZH.md)

Fine-tune the released ABot-Recon model with 18 supported training sources.
The framework includes weighted data mixing, causal training, EMA, validation
and checkpoint resume. Example configurations are for fine-tuning, not
reproduction of the paper's full multi-stage training recipe. For the method
and training strategy, see the [paper](https://arxiv.org/abs/2608.27529),
especially §3.4 and Appendix A.

## Installation

The reference environment is Linux, Python 3.11, PyTorch 2.5.1 and CUDA 12.1.
Run from a source checkout; the wheel does not include example configs,
preprocessing scripts or tests.

```bash
conda create -n abot-recon python=3.11 -y
conda activate abot-recon
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[train]"
```

Compiling cuRoPE is recommended for training speed:

```bash
cd abot_recon/modeling/pi3/models/curope
pip install ninja
python setup.py build_ext --inplace
cd -
```

## Pretrained checkpoint

Download the released weights from
[Hugging Face](https://huggingface.co/acvlab/ABot-Recon) or
[ModelScope](https://modelscope.cn/models/amap_cvlab/ABot-Recon):

```bash
hf download acvlab/ABot-Recon abot_recon.safetensors --local-dir checkpoints
```

Training requires a local checkpoint. Set `pretrained` if saved elsewhere.
For inference and optional loop-closure dependencies, see the
[public inference repository](https://github.com/amap-cvlab/ABot-Recon).
Fine-tuning does not require loop-closure assets; pass `loop_closure=False`
when using the inference API without them.

## Data preparation

### Choose a preparation route

| Data | Preparation |
|---|---|
| 16 CUT3R-compatible sources, including DL3DV and VKITTI2 | Follow the [dataset guide](docs/training_datasets.md#layout-and-sampling), including auxiliary stages/downloads. DL3DV and VKITTI2 use CUT3R's published processed data. |
| ScanNet++ Seq | Use the bundled [sequence preprocessor](preprocess/README.md), not CUT3R's pair export. |
| TartanGround | Use the four-directory layout below, also recommended as a template for your own RGB-D data. |

Each adapter keeps its native disk format. Compatibility refers to **final
processed artifacts**, not raw downloads or identical data cleaning. Obtain
datasets under their original terms. Roots must be filesystem paths, including
mounted storage, rather than `oss://` URIs.

### Shared loaded-data interface and geometry conventions

The shared interface is the adapter's **loaded sample**, not a universal disk
format. Adapters return images, world-space points, valid masks, intrinsics,
C2W poses and dataset/sequence labels. Depth is an intermediate used to construct
points; the returned `valid_mask` is `True` for valid pixels.

Use OpenCV camera axes (right, down, forward), camera-z depth and pinhole
intrinsics in pixels. Depth and pose translations must share a scale; not every
dataset is metric. RGB, depth and masks must be registered, with intrinsics
matching their resolution. Camera-only supervision requires an adapter that
explicitly supports calibrated images and poses, not arbitrary RGB-only folders.

### Data preparation examples

#### DL3DV

Merge the RGB/camera and depth/mask downloads following
[CUT3R's DL3DV instructions](https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/docs/preprocess.md#dl3dv):

```text
DL3DV_ROOT/<bucket>/<scene>/dense/
  rgb/00000.png
  depth/00000.npy          # float32 (H, W)
  cam/00000.npz            # intrinsic (3, 3), pose (4, 4), C2W
  sky_mask/00000.png       # values >=127 are invalid
  outlier_mask/00000.png   # values >=127 are invalid
```

Set `data.dl3dv.root` to `DL3DV_ROOT`. This is CUT3R-compatible data, not a
separate ABot-specific format. See the [blacklist settings](docs/training_datasets.md#bundled-training-blacklists)
before using a different processed export.

#### TartanGround

```text
TARTANGROUND_ROOT/<sequence>/
  images/00000.jpg
  depths/00000.npy         # float32 (H, W), camera-z depth in metres
  cameras/00000.npz        # camera_intrinsics (3, 3), camera_pose (4, 4), C2W
  masks/00000.npy          # bool (H, W), True means invalid
```

Use matching zero-padded basenames and float32 camera matrices. Set
`data.tartanground.root` to `TARTANGROUND_ROOT`.

#### Fine-tune on your own data

Use the TartanGround layout as a template and start from
[configs/finetune_custom.yaml](configs/finetune_custom.yaml). Set `pretrained`,
`output_dir` and `data.tartanground.root`; other training sources are disabled
in this example. Check depth units, pose direction and a multi-frame point-cloud
overlay before training.

Reusing this adapter also adopts its metric supervision, normal loss and
static-scene foldback sampling. Sequence-name prefixes can select environment
interval overrides. For other supervision or sampling needs, implement a
dedicated adapter. See [custom-data policies](docs/training_datasets.md#fine-tune-on-your-own-rgb-d-sequences).

For held-out validation, put the same sequence names in **both**
`data.tartanground.eval_scenes` and `data.tartanground.exclude_scenes`:
selecting validation scenes does not exclude them from training. Otherwise set
`validation_enabled: false`; use `exclude_scenes: []` only when deliberately
training on all sequences without that holdout.

### Shared preprocessing and augmentation

RGB, depth, masks and camera geometry are transformed together. Photometric
augmentation affects image content, not padding. Sequence-consistent augmentation
has probability 0.2; preserve-horizontal-FOV routing has probability 0.9.
`principal_align_skip_prob` defaults to **0**, so the example always aligns
principal points; it does not enable the paper's selected-dataset off-center
recipe. See the configs for controls and paper Appendix A.3 for the method.

## Generated data example

Generate DL3DV and TartanGround layout fixtures and run a four-step GPU smoke
test with the downloaded checkpoint:

```bash
python scripts/make_training_sample.py --output examples/training_data --frames 32
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_machines 1 --num_processes 1 \
  --mixed_precision bf16 --dynamo_backend no \
  -m abot_recon.training.cli --config configs/finetune_smoke.yaml
```

This trains on generated TartanGround data, disables validation and automatic
resume, and writes to `outputs/finetune_smoke`. Use a fresh output directory.
Synthetic fixtures check wiring, not real-data quality or learning.

## Training configuration

| Configuration | Purpose |
|---|---|
| [finetune.yaml](configs/finetune.yaml) | DL3DV + TartanGround, with TartanGround validation. |
| [finetune_cut3r.yaml](configs/finetune_cut3r.yaml) | Supported sources; only TartanAir enabled initially, no validation. |
| [finetune_custom.yaml](configs/finetune_custom.yaml) | One custom RGB-D source using TartanGround's layout and policies. |
| [finetune_smoke.yaml](configs/finetune_smoke.yaml) | Four-step generated-data smoke test. |

Each YAML is standalone; omitted fields use `TrainConfig` defaults. The default
example trains **32 frames at 504×280**, batch size 1 per GPU, with AdamW,
OneCycleLR, gradient clipping and EMA (`0.999`). Peak learning rates are `1e-6`.
Weights are normalized batch-selection probabilities; set unused sources to
`weight: 0`. `data.num_frames` controls clip length; `max_frames` is the temporal
index/KV-cache capacity, not the sampled clip length.

## Start training

Edit the checkpoint, output directory and enabled roots in your chosen config:

```yaml
pretrained: checkpoints/abot_recon.safetensors
output_dir: outputs/my_finetune
data:
  dl3dv:
    root: /path/to/processed_dl3dv_ours
  tartanground:
    root: /path/to/processed_TartanGround
```

Run from the repository root. Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_machines 1 --num_processes 1 \
  --mixed_precision bf16 --dynamo_backend no \
  -m abot_recon.training.cli --config configs/finetune.yaml
```

Four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu \
  --num_machines 1 --num_processes 4 --mixed_precision bf16 --dynamo_backend no \
  -m abot_recon.training.cli --config configs/finetune.yaml
```

Use a **fresh output directory for a new experiment**: automatic resume is on
by default. `steps_per_epoch` counts micro-batches and must be divisible by
`gradient_accumulation_steps`. Non-finite loss/gradients or loss above `max_loss`
discard the accumulation window on all ranks.

## Trainable scopes and losses

The encoder is always frozen. `trainable_scope` selects `all` (default), `heads`
or `rot_correction_only`. Confidence is controlled separately by
`enable_confidence`, defaults to off, and requires a checkpoint containing the
confidence branch when enabled.

Training combines point, normal and relative-camera losses, rotation-refinement
regularization, and optional confidence supervision. See paper §3.4 for the
formulation and [loss.py](abot_recon/training/loss.py) / the config's `loss`
section for implementation and weights. Camera-only samples skip point, normal
and confidence supervision.

## Validation, checkpoints and resume

Validation uses TartanGround-compatible held-out data, with **128-frame** streams
at strides **1 and 6**, independent of training clip length. It is deterministic,
uses per-frame streaming inference with fresh state for each clip, and uses EMA
when `eval_with_ema: true`. Adding another training source does not add its
validation benchmark. `validation_steps: -1` evaluates both streams fully;
positive values cap batches per rank and stream.

`output_dir/metrics.jsonl` records training and validation metrics, including
`val/s1/*`, `val/s6/*` and their average `val/*`. Completed checkpoints are saved
under `output_dir/checkpoint-XXXX/`:

- `abot_recon.safetensors`: online model;
- `abot_recon_ema.safetensors`: full EMA model, recommended for inference;
- `ema.pt` and `trainer_state/`: EMA and training state for resume.

EMA files are present when EMA is enabled. The full EMA model is exported
automatically; [scripts/merge_ema_checkpoint.py](scripts/merge_ema_checkpoint.py)
can rebuild it manually. The final epoch is always saved; existing checkpoint
directories are not overwritten.

Automatic resume selects the latest complete checkpoint. To select one explicitly:

```yaml
resume: outputs/my_finetune/checkpoint-0004
auto_resume: false
resume_schedule: strict
```

`strict` rejects training-plan changes. `restart` retains model/optimizer/EMA
state but starts a new OneCycle over the remaining epochs. Use a new output
directory when branching from an earlier checkpoint. Resume restores training
state but does **not** guarantee identical data order or augmentation replay.

## Tests and code

The training loop is in [abot_recon/training/trainer.py](abot_recon/training/trainer.py);
data mixing is in [data.py](abot_recon/training/data.py).

```bash
pip install -e ".[train,loop,dev]"
pytest -q
```

GPU/checkpoint integration tests are opt-in. To test the released model's
forward/loss/backward path:

```bash
ABOT_RECON_CHECKPOINT=checkpoints/abot_recon.safetensors \
ABOT_RECON_DEVICE=cuda \
pytest -q tests/integration/test_real_training.py
```

## Citation and license

Please cite the ABot-Recon paper when using this training code. Project-authored
code uses Apache-2.0 except where otherwise noted. Third-party-derived components,
including CUT3R-derived dataset adapters (CC BY-NC-SA 4.0), retain their upstream
terms; see [LICENSE](LICENSE) and [Third-Party Notices](THIRD_PARTY_NOTICES.md).

# Offline visualization

English | [简体中文](README_ZH.md)

Render ABot-Recon results as a ground-follow point-cloud video and bird's-eye images:
**inference → sky masks → rendering**. These scripts do not rerun reconstruction or modify poses.
Run commands from the repository root.

## Setup

Follow the [main README](../README.md) for the inference environment and checkpoint, then install:

```bash
pip install 'opencv-python>=4.8' 'scipy>=1.10'
pip install onnxruntime-gpu  # For sky masks; use onnxruntime instead for CPU.
```

Video output requires `ffmpeg`. CUDA is recommended; use `--device cpu` for CPU execution. No display server or OpenGL is needed.

## Quick start

### 1. Save inference results

Skip this step if your results already include dense point maps, RGB, and confidence.

```bash
python -m abot_recon.cli \
  --image-dir examples/images --output-dir outputs/inference \
  --stride 5 --dense-stride 1 --confidence-threshold 0 \
  --save-local-points --save-confidence --no-loop-closure
```

This example disables loop closure; see the main README to enable it. Keep the inference confidence threshold at 0 so filtering can be adjusted later.

### 2. Generate sky masks

```bash
python visualization/generate_sky_masks.py \
  --result-dir outputs/inference --download-model --device cuda:0
```

This downloads [SkySeg](https://huggingface.co/JianyuanWang/skyseg/tree/main) to `checkpoints/skyseg.onnx` and saves masks under `outputs/inference/sky_masks`. Follow the model's license. For existing weights, replace `--download-model` with `--model-path PATH`; to use a mirror, prefix the command with `HF_ENDPOINT=https://hf-mirror.com`.

Defaults: `--non-sky-threshold 0.3 --sky-dilate 3` (3-native-pixel dilation radius). Scores are normalized per image, not calibrated probabilities. Regenerate masks after changing these settings, and keep `manifest.json` with the PNGs.

### 3. Render

```bash
python visualization/render.py \
  --result-dir outputs/inference --output-dir outputs/visualization \
  --device cuda:0
```

**Sky filtering is on by default.** Missing or incompatible masks cause an error. Use `--sky-mask-dir PATH` for another location or `--no-sky` to disable it. Rendering never runs or downloads the sky model.

By default, point clouds use the inference resolution. To upsample to the original image size and also export the matching RGB video:

```bash
python visualization/render.py \
  --result-dir outputs/inference --output-dir outputs/dense_visualization \
  --image-dir examples/images --image-stride 5 \
  --resolution source --upsample bilinear --pixel-step 1 \
  --outputs video trajectory pointcloud rgb --device cuda:0
```

Source image order and `--image-start/--image-end/--image-stride` must match inference exactly. RGB keeps its aspect ratio. `--width/--height` set the rendered video size, not point-cloud sampling resolution.

## Timing and sampling

Sampling uses **inference frames**, not source video frames. For example, after the first view, `--view-step 2 --cloud-step 1` advances two poses per video frame while adding both intervening dense maps. Only saved dense frames are available. Playback FPS is independent of the input; no image interpolation is used.

With enough input, defaults produce **600 ground + 120 transition frames = 24 seconds**. The transition freezes reconstruction time, moves the observer, and fades the cloud out; RGB holds its last frame. Use `--start/--end` for a segment and `--global-scope segment` to restrict bird images and trajectory history to it (default: full sequence).

For dense runs, increase `--cloud-step` or `--pixel-step`, or limit `--history`, to control memory. Increasing only `--view-step` does not reduce the added clouds. Exceeding `--max-resident-points` raises an error rather than silently reducing density.

## Sky-mask parameters

All options for `generate_sky_masks.py`. N denotes the number of inference frames. Optional source images must correspond to the **entire inference sequence**, not just the segment to render. Output masks always use the inference resolution.

| Parameter | Default | Setting / effect |
|---|---|---|
| `--result-dir` | Required | Published inference result directory. |
| `--output-dir` | `RESULT_DIR/sky_masks` | Directory for mask PNGs and their manifest. |
| `--image-dir` | Unset | Original image directory; otherwise use the saved inference RGB. Mutually exclusive with `--frame-list`. |
| `--image-start` | `0` | First source-image index, before stride selection; match inference. |
| `--image-end` | End of input | Exclusive source-image end index; match inference. |
| `--image-stride` | `1` | Source-image selection stride; match inference, e.g. 5 for inference `--stride 5`. |
| `--frame-list` | Unset | JSON list of exactly N ordered source paths. Relative paths resolve against the JSON file; image start/end/stride are not applied to this list. |
| `--model-path` | `<repo>/checkpoints/skyseg.onnx` | ONNX weights; the default is relative to the script's repository, not the working directory. |
| `--download-model` | `false` | Download missing weights from `JianyuanWang/skyseg`; respects `HF_ENDPOINT`. Existing weights are reused. |
| `--device` | `cpu` | `cpu` or `cuda:N`. Requested CUDA must be available; no silent CPU fallback. |
| `--non-sky-threshold` | `0.3` | Reject normalized non-sky scores ≤ this value, in [0,1]. Higher values reject more pixels; 0 does not disable sky filtering. |
| `--sky-dilate` | `3` | Rejection-mask dilation radius in inference pixels; 3 uses a 7×7 ellipse, 0 disables dilation. |
| `--overwrite` | `false` | Replace this script's existing masks and manifest, preserving unrelated files. Use when regenerating masks with changed settings. |
| `-h`, `--help` | — | Show command-line help and exit. |

## Rendering parameters

All options for `render.py`. N denotes the number of inference frames. Distances and depths use reconstruction units, not necessarily meters; pixel sizes use the output image unless noted otherwise.

### Input and frame selection

| Parameter | Default | Setting / effect |
|---|---|---|
| `--result-dir` | Required | Published inference result directory. |
| `--output-dir` | Required | Separate directory for rendered outputs; must differ from the inference directory. |
| `--poses` | `final` | Select saved `camera_poses.npy` (`final`), `camera_poses_noloop.npy` (`noloop`), or `camera_poses_loop.npy` (`loop`). The selected file must exist; no new loop optimization runs. |
| `--start` | `0` | First inference-frame index of the ground segment. |
| `--end` | `N` | Exclusive inference-frame end index. Require `0 ≤ start < end ≤ N`. |
| `--ground-frames` | `600` | Maximum number of ground video frames; shorter input is not repeated. 0 is allowed only without `video`/`rgb` outputs, not as an unlimited setting. |
| `--view-step` | `1` | Inference-frame advance per output frame: view k follows `start + k × view-step`. |
| `--cloud-step` | `1` | Add saved dense frames with `(index − start) % cloud-step == 0`; all selected frames up to the current view are added. |
| `--global-scope` | `all` | Bird-image scope: `all` = full sequence; `segment` = `[start,end)`, independent of `--ground-frames`. Video trajectory history starts at 0/start respectively and only grows through the current view. |
| `--image-dir` | Unset | Original images for source-resolution clouds or RGB output; optional for native clouds. Selection must match the full inference sequence, not just the rendered segment. Mutually exclusive with `--frame-list`. |
| `--image-start` | `0` | First source-image index before stride selection; match inference. |
| `--image-end` | End of input | Exclusive source-image end index; match inference. |
| `--image-stride` | `1` | Source-image selection stride; match inference, not `--view-step`. |
| `--frame-list` | Unset | JSON list of exactly N ordered image paths. Relative paths resolve against the JSON file; image start/end/stride do not apply. |

### Point clouds and filtering

Confidence, depth, edge, and sky filtering are shared by ground and bird clouds and occur before source-resolution upsampling. History limits and frustum pruning apply only to accumulated ground clouds, after upsampling. Sampling steps must be positive integers.

| Parameter | Default | Setting / effect |
|---|---|---|
| `--resolution` | `native` | `native`: saved inference resolution; `source`: original-image resolution, requiring source images. Applies to ground and bird clouds. |
| `--upsample` | `bilinear` | `bilinear` or `nearest`, used only in source mode. Interpolate depth and rays; invalid regions remain excluded. |
| `--pixel-step` | `1` | Ground-cloud sampling on both axes at the chosen resolution; 1 uses all valid pixels, 2 keeps about 1/4. |
| `--confidence` | `0.15` | Keep confidence ≥ threshold, in [0,1]. 0 disables this filter; a positive threshold requires `confidence.pt`. |
| `--min-depth` | `0` | Keep camera-local Z greater than this lower bound. Nonpositive/invalid depth is always rejected. |
| `--max-depth` | `6` | Upper camera-local Z bound; 0 removes only the upper bound. Otherwise must exceed `--min-depth`. |
| `--depth-edge` | `0.15` | Reject local relative discontinuities where `(max depth − min depth) / Z` exceeds this threshold. 0 disables the edge filter. |
| `--edge-kernel` | `3` | Positive odd neighborhood width for depth-edge detection, in inference pixels. |
| `--sky`, `--no-sky` | `true` | Enable/disable precomputed sky rejection. Enabled filtering requires complete, matching masks. |
| `--sky-mask-dir` | `RESULT_DIR/sky_masks` | Directory containing the mask PNGs and manifest generated above. |
| `--history` | `0` | Retain the last N inference frames' clouds; 0 removes this age limit, but frustum pruning still applies. Does not limit bird images or camera history. |
| `--prune-every` | `10` | Permanently remove off-frustum ground points every N output frames, starting at the first frame; 0 disables. Pruning pauses during the transition. |
| `--bird-cloud-step` | `5` | Bird-cloud temporal step in inference indices, measured from the selected global scope's start. |
| `--bird-pixel-step` | `8` | Bird-cloud sampling on both image axes at the chosen resolution; independent of `--pixel-step`. |

### Render cameras and appearance

Only the virtual observer is smoothed; reconstructed poses are unchanged. Its eye is `C − dF + d × height-ratio × U`, looking toward `C + d × look-ahead × F`, where C/F/U are the smoothed camera center/forward/up directions. Forward is the predicted camera direction, not necessarily the trajectory tangent.

| Parameter | Default | Setting / effect |
|---|---|---|
| `--width` | `1920` | Rendered video/PNG width; positive, and even for video. Does not resize the RGB reference video. |
| `--height` | `1080` | Rendered video/PNG height; positive, and even for video. Does not resize the RGB reference video. |
| `--background` | `32 32 32` | Three RGB values in [0,255] for video and the opaque final frame. Bird PNGs remain transparent. |
| `--point-size` | `2` | Positive square point-splat diameter in output pixels. |
| `--ground-fov` | `60` | Ground observer's vertical FoV in degrees, strictly between 1 and 175. |
| `--bird-fov` | `60` | Bird observer's vertical FoV in degrees, strictly between 1 and 175; reached through the transition. |
| `--bird-margin` | `1.12` | Trajectory-fitting margin for the common bird camera; ≥ 1, larger values leave more space. |
| `--follow-distance` | `0` | Backward distance d. Positive values fix it and bypass automatic distance settings; 0 selects automatic distance. |
| `--distance-ratio` | `0.12` | Automatic d = this positive ratio × full-trajectory radius, then clamped below. Radius is the 90th-percentile distance from the median center, with a floor of 0.5. |
| `--distance-min` | `0.6` | Positive lower bound for automatic follow distance. |
| `--distance-max` | `2` | Upper bound for automatic follow distance; must be ≥ the lower bound. |
| `--height-ratio` | `0.2` | Nonnegative height above the smoothed center, as a fraction of d along local up. |
| `--look-ahead` | `0.35` | Target offset as a fraction of d along local forward; increasing it looks farther ahead. |
| `--smooth-radius` | `6` | Center/forward smoothing radius in inference frames: normally `2r+1` samples. 0 uses the current frame only. |
| `--smooth-frames` | `6` | Positive window length for smoothing the observer's up direction; 1 uses only the current frame. |
| `--frustum-step` | `1` | Draw a frustum at every Nth inference index, counted from index 0; 0 hides frustums, not the trajectory. |
| `--frustum-size` | `0.05` | Positive display-frustum length along the reconstruction camera's local Z axis. |
| `--trajectory-width` | `2` | Positive trajectory line width in output pixels. |
| `--frustum-width` | `1` | Positive frustum line width in output pixels. |
| `--center-size` | `2` | Square camera-center marker size in output pixels; 0 hides center markers. |

### Outputs and runtime

| Parameter | Default | Setting / effect |
|---|---|---|
| `--fps` | `30` | Positive playback frame rate for both videos, independent of input FPS. |
| `--transition-frames` | `120` | Additional bird-transition frames; 0 disables. Reconstruction time freezes, only the observer moves, and points fade out. |
| `--outputs` | `video trajectory pointcloud` | One or more of `video`, `trajectory`, `pointcloud`, `rgb`. `rgb` requires source images; choose only `trajectory pointcloud` for PNGs without videos. |
| `--device` | `cpu` | Point rasterizer device: `cpu` or `cuda:N`. |
| `--chunk-points` | `200000` | Maximum points per rasterization batch; controls temporary working memory, not cloud density. |
| `--max-resident-points` | `20000000` | Ground-cloud point-count budget, not bytes. Checked during insertion, including before history/pruning cleanup; exceeding it raises an error. |
| `--crf` | `20` | H.264 quality setting, 0–51; lower values generally mean higher quality and larger files. |
| `--preset` | `medium` | Encoder preset: `ultrafast`, `fast`, `medium`, or `slow`; slower presets favor compression. |
| `--overwrite` | `false` | Replace this script's named render artifacts in an existing output directory; leave inference and unrelated files untouched. |
| `-h`, `--help` | — | Show command-line help and exit. |

## Outputs

| File | Content |
|---|---|
| `ground_pointcloud.mp4` | Ground-follow video with optional bird transition |
| `ground_rgb.mp4` | Matching source RGB video; only with `--outputs ... rgb` |
| `last_frame.png` | Final rendered frame; a transition also produces `last_frame_transparent.png` |
| `birdseye_trajectory.png` | Full-scope trajectory and cameras, transparent background |
| `birdseye_pointcloud.png` | Full-scope point cloud, transparent background |
| `render_manifest.json` | Settings, frame mapping, and completion status |
| `filter_stats.jsonl` | Per-frame filtering counts for each rendering pass |
| `frustum_intrinsics.json` | Intrinsics fitted for display frustums and their fit residuals |

Bird images and the transition endpoint share the same camera for aligned overlays. Use a new output directory or explicitly pass `--overwrite`; inference results are not modified.

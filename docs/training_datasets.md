# Training datasets and preprocessing

This guide lists the 18 supported sources, their processed formats and setup
requirements. For training commands and the DL3DV/TartanGround examples, see
the [README](../README.md#data-preparation).

## Choose a preparation route

- **16 CUT3R-compatible sources:** use the upstream preprocessors and auxiliary
  stages below. DL3DV and VKITTI2 instead use CUT3R's published processed data.
- **ScanNet++ Seq:** use the bundled [sequence preprocessor](../preprocess/README.md),
  not CUT3R's pair exporter.
- **TartanGround:** use the [four-directory example](../README.md#tartanground),
  also recommended as a template for custom RGB-D sequences.

The upstream reference is
[CUT3R commit `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`](https://github.com/CUT3R/CUT3R/tree/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/datasets_preprocess).
Compatibility means its **final processed artifacts**, including required indices
and downloads, can be loaded without further format conversion. It does not
imply identical cleaning or training scene selection. Keep each native disk
layout and follow the dataset's download and usage terms.

## Layout and sampling

Paths are relative to the configured `root`; `f` denotes a frame basename.
Camera poses are C2W. The table summarizes sampling; algorithm details are in
paper Appendix A.1–A.2 and the individual adapters.

| Config key | Preprocessing | Required layout / camera keys | Sampling |
|---|---|---|---|
| `dl3dv` | CUT3R processed downloads, merged | `<bucket>/<scene>/dense/{rgb/f.png,depth/f.npy,cam/f.npz,sky_mask/f.png,outlier_mask/f.png}`; `intrinsic`, `pose` | Adaptive temporal, max interval 20 |
| `tartanground` | Project four-directory layout | `sequence/{images/f.jpg,depths/f.npy,cameras/f.npz,masks/f.npy}`; `camera_intrinsics`, `camera_pose` | Foldback, environment-specific intervals |
| `tartanair` | `preprocess_tartanair.py` | `env/{Easy,Hard}/sequence/{f_rgb.png,f_depth.npy,f_cam.npz}`; `camera_intrinsics`, `camera_pose` | TartanAir v1; forward/revisit, interval 1–20 |
| `pointodyssey` | `preprocess_point_odyssey.py` | `{train,val,test}/scene/{rgb/f.jpg,depth/f.npy,cam/f.npz}`; `intrinsics`, `pose` | Forward/revisit, interval 1–4; built-in scene whitelist |
| `spring` | `preprocess_spring.py` | `sequence/{rgb/f.png,depth/f.npy,cam/f.npz}`; `intrinsics`, `pose` | Foldback, interval 1–4 |
| `mvs_synth` | `preprocess_mvs_synth.py` | `sequence/{rgb/f.jpg,depth/f.npy,cam/f.npz}`; `intrinsics`, `pose` | Foldback, interval 1–4 |
| `dynamic_replica` | `preprocess_dynamic_replica.py` | `split/sequence/left/{rgb/f.png,depth/f.npy,cam/f.npz}`; `intrinsics`, `pose` | Left camera, forward, interval 1–16 |
| `uasol` | `preprocess_uasol.py` | `scene/{rgb/f.png,depth/f.npy,cam/f.npz}`; `intrinsics`, `pose` | Forward/revisit, interval 1–40 |
| `arkit_hr` | `preprocess_arkitscenes_highres.py` | `{Training,Validation}/scene/{scene_metadata.npz,vga_wide/*.jpg,highres_depth/*.png}`; metadata: `images`, `intrinsics`, `trajectories` | Timestamp segments, stride-1 foldback |
| `wildrgbd` | `preprocess_wildrgbd.py` | `selected_seqs_{train,test}.json`, `category/sequence/{rgb/f.jpg,depth/f.png,masks/f.png,metadata/f.npz}`; `camera_intrinsics`, `camera_pose` | Foldback, interval 1–4 |
| `unreal4k_seq` | `preprocess_unreal4k.py` | `scene/{0,1}/{f_rgb.png,f_depth.npy,f.npz}`; `intrinsics`, `cam2world` | Spatial pose graph, not filename chronology |
| `scannet` | `preprocess_scannet.py` + `generate_set_scannet.py` | `scans_{train,test}/scene*/{color/f.jpg,depth/f.png,cam/f.npz,new_scene_metadata.npz}`; `intrinsics`, `pose` | Adaptive forward/revisit, interval 1–30 |
| `waymo` | `preprocess_waymo.py` + `invalid_files.h5` download | `scene/00000_1.{jpg,exr,npz}`, root `invalid_files.h5`; `intrinsics`, `cam2world` | Separate cameras; interval 1–8 (camera 4: 1–4); excludes camera 5 |
| `vkitti2` | CUT3R processed archive | `SceneXX/variant/Camera_{0,1}/00000_{rgb.jpg,depth.png,cam.npz}`; `camera_intrinsics`, `camera_pose` | Separate camera/variant streams, interval 1–5 |
| `hypersim_seq` | `preprocess_hypersim.py` | `scene/cam_*/{f_rgb.png,f_depth.npy,f_cam.npz}`; `intrinsics`, `pose` | Pose graph within each scene |
| `blendedmvs_seq` | `preprocess_blendedmvs.py` + `new_overlap.h5` download | `scene/{f.jpg,f.exr,f.npz}`, root `new_overlap.h5`; `intrinsics`, `R_cam2world`, `t_cam2world` | Pose graph, upright-pose filter |
| `arkit` | `preprocess_arkitscenes.py` + `generate_set_arkitscenes.py` | `{Training,Test}/all_metadata.npz`, `scene/{new_scene_metadata.npz,vga_wide/*.jpg,lowres_depth/*.png}` | Timestamp segments, stride-1 foldback; camera-only by default in config |
| `scannetpp_seq` | Bundled `preprocess_scannet_seq.py` | `all_metadata.npz`, `scene_device/{images/f.jpg,depths/f.npy,masks/f.npy,cameras/f.npz}`; `camera_intrinsics`, `camera_pose` | Separate iPhone/DSLR streams, stride-1 foldback |

Depth units are handled by the adapters: ARKit/ARKit HR, WildRGBD and ScanNet
PNG depth is in millimetres; VKITTI2 PNG depth is in centimetres. Waymo EXR and
HyperSim/ScanNet++ NPY depth are metric. DL3DV and BlendedMVS retain the export's
scale shared with camera translations, not necessarily metres. Do not apply
unit or pose conversions again after upstream preprocessing.

TartanGround and ScanNet++ boolean masks use `True` for **invalid** pixels;
DL3DV mask values >=127 are invalid. WildRGBD masks instead denote valid
foreground. The [loaded-data interface](../README.md#shared-loaded-data-interface-and-geometry-conventions)
normalizes these conventions for training.

### DL3DV and VKITTI2: processed downloads

For DL3DV, follow [CUT3R's download-and-merge instructions](https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/docs/preprocess.md#dl3dv).
Merge matching RGB/camera and depth/mask paths under one root, retaining all five
directories. Set `data.dl3dv.root` to the directory containing the buckets.
The upstream raw `preprocess_dl3dv.py` route is deprecated.

For VKITTI2, extract [CUT3R's complete processed archive](https://drive.google.com/file/d/1KdAH4ztRkzss1HCkGrPjQNnMg5c-f3aD/view?usp=sharing).
Set `data.vkitti2.root` to the directory containing `SceneXX` folders.
`split: null` uses all scenes; `train` excludes the last sorted scene and `test`
selects it. No additional index stage is needed. The reference CUT3R tree does
not provide a VKITTI2 raw conversion script.

### ScanNet: two preprocessing stages

Run commands from CUT3R's `datasets_preprocess` directory with its dependencies.
First extract sensor recordings to CUT3R's expected raw layout; see its
[preparation instructions](https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/docs/preprocess.md).

```sh
python preprocess_scannet.py --scannet_dir RAW_ROOT --output_dir PROCESSED_ROOT
python generate_set_scannet.py --root PROCESSED_ROOT \
  --splits scans_test scans_train --max_interval 150 --num_workers 8
```

Set `data.scannet.root: PROCESSED_ROOT`, not its split subdirectory, and choose
`split: train` or `test`. The second stage supplies required
`new_scene_metadata.npz`. Its collection interval is not the training sampling
interval, which is measured in positions of the ordered image list.

### Waymo: preprocessing and invalid-pair metadata

Use raw Perception 1.4.2 TFRecords and the upstream-required precomputed pairs:

```sh
python preprocess_waymo.py --waymo_dir RAW_TFRECORD_ROOT \
  --precomputed_pairs PAIRS.npz --output_dir PROCESSED_ROOT --workers 1
```

Separately download [invalid_files.h5](https://drive.google.com/file/d/1xI2SHHoXw1Bm7Lqrn7v56x30stCNuhlv/view?usp=sharing)
to `PROCESSED_ROOT`; preprocessing does not generate it. Set
`data.waymo.root: PROCESSED_ROOT`, `split: null`. Raw TFRecords or the temporary
extraction folder are not training roots. The loader needs h5py and an
OpenCV build with OpenEXR; the TensorFlow/Waymo SDK is only for preprocessing.

### HyperSim Seq and BlendedMVS Seq

For HyperSim, run `preprocess_hypersim.py --hypersim_dir RAW_ROOT
--output_dir PROCESSED_ROOT`. No additional pair-index pass is consumed.
The producer already converts radial range to camera-z depth and applies
metric scale and camera-axis conversions.

For BlendedMVS, run `preprocess_blendedmvs.py --blendedmvs_dir RAW_ROOT
--precomputed_pairs PAIRS.npz --output_dir PROCESSED_ROOT`, then download
[new_overlap.h5](https://drive.google.com/file/d/1anBQhF9BgOvgaWgAwWnf70tzspQZHBBB/view?usp=sharing)
to the output root. The loader uses its frame lists to build pose graphs;
preprocessing does not generate this auxiliary file. OpenCV OpenEXR and h5py
are required. Depth and translation share an arbitrary scene scale; the
default `z_far: 0` avoids an additional metric cutoff.

### ARKit: two stages and camera-only supervision

Run both upstream stages:

```sh
python preprocess_arkitscenes.py --arkitscenes_dir RAW_ROOT \
  --precomputed_pairs PAIRS_ROOT --output_dir PROCESSED_ROOT
python generate_set_arkitscenes.py --root PROCESSED_ROOT \
  --splits Training Test --max_interval 5 --num_workers 8
```

Keep split-level `all_metadata.npz` and per-scene `new_scene_metadata.npz`,
including `image_collection`. Only load trusted metadata: this upstream member
is a pickled dictionary. `split: train` selects `Training`; `test` selects `Test`.

The adapter excludes high-resolution scene IDs using `ROOT_highres/Training`
or `ROOT_highres/Validation`, or an explicit `highres_root`. These HR directories
must exist. Timestamp gaps over 1 second or non-increasing times split sequences.

The training config defaults to `camera_only: true`: depth files are not read,
placeholder depth is only for geometric preprocessing, and point/normal/confidence
supervision is skipped. Set it to `false` to use original RGB-D supervision.
Other adapters still require their documented depth inputs.

### ScanNet++ Seq

Follow the [bundled preprocessing guide](../preprocess/README.md). It exports
separate `<scene>_iphone` / `<scene>_dslr` sequences and generates root
`all_metadata.npz`; no CUT3R pair-generation pass is needed.

Sampling uses processed stride 1 with foldback, even when `allow_repeat: false`;
`forward_only: true` requires forward headroom instead. The producer defaults
to a 40 m depth bound; the loader's `z_far: 0` adds no further far cutoff.
Current producer defaults do not exactly reproduce historical training exports.

### Bundled training blacklists

These exclusions are specific to processed-data versions and quality checks,
not universal judgments about the raw datasets. Reassess them after reprocessing.
No object-storage access is needed for bundled defaults.

| Dataset | Default exclusions | Training config field |
|---|---:|---|
| DL3DV | 542 sequences, bundled TXT | `data.dl3dv.blacklist` |
| BlendedMVS Seq | 38 scenes, bundled TXT | `data.blendedmvs_seq.scene_blacklist_path` |
| HyperSim Seq | 4 camera folders, inline | `data.hypersim_seq.sequence_blacklist_path` |
| ScanNet++ Seq | 8 device sequences: 7 extra + 1 built-in | `data.scannetpp_seq.sequence_blacklist_path` |

- `null` / `auto` (default): use a dataset-root sidecar if present, otherwise
  the packaged TXT or inline defaults. Root filenames are
  `dl3dv_geometry_blacklist_20260810.txt`, `blendedmvs_scene_blacklist.json`,
  `hypersim_geometry_blacklist_20260810.txt` and `scannetpp_geometry_blacklist_20260810.txt`.
- An explicit path **replaces** the automatic extra list; missing files error.
  Use absolute paths for portability. Relative paths are resolved against the
  dataset root, except DL3DV, which uses the working directory.
- `""` disables extra blacklists, not longstanding built-in exclusions or other
  validity/orientation filters.

Bundled TXT files are in `abot_recon/training/datasets/metadata/` and resolved
relative to the loader, not the working directory. One exact ID per line;
blank lines and whole-line `#` comments are ignored. IDs are `<bucket>/<scene>`
for DL3DV, `scene/cam_*` for HyperSim, `<scene>_iphone` / `<scene>_dslr` for
ScanNet++, and scene IDs for BlendedMVS. BlendedMVS also accepts a JSON list or
`{"drop_scenes": [...]}`. No prefix matching is performed.

### Splits, short sequences and validity

- PointOdyssey and DynamicReplica use their native split folders. ARKit HR maps
  `train`/`test` to `Training`/`Validation`; WildRGBD uses the split JSON.
- TartanAir, Spring, MVS-Synth, UASOL and HyperSim/BlendedMVS/ScanNet++ Seq use the supplied
  root/metadata as the scene set: `split` does not create a holdout.
  UnrealStereo4K and Waymo require `split: null`.
- Short-sequence/repeat rules are adapter-specific. Ordinary non-repeat indices
  require at least `num_frames`; repeat-enabled indices usually require
  `max(num_frames // 3, 3)`. ARKit timestamp segments and pose-graph connectivity
  impose additional eligibility checks. ScanNet++ Seq has its own foldback rule.
- Depth cleanup remains dataset-specific. `z_far` is an additional validity
  cutoff, not a unit conversion. WildRGBD's `mask_bg: rand` applies foreground
  masking to the whole sample with probability 0.1; `true`/`false` force it on/off.
- Normal loss is enabled for TartanGround, TartanAir, PointOdyssey, ScanNet and
  VKITTI2, not the HyperSim/BlendedMVS/ScanNet++ **Seq** labels.

## Fine-tune on your own RGB-D sequences

Use the [TartanGround format and custom config](../README.md#fine-tune-on-your-own-data).
Undistort and register RGB/depth/masks, use metric camera-z depth and matching
C2W poses, and verify multi-frame geometry before training.

Reusing `data.tartanground` also adopts metric supervision, normal loss and
static-scene foldback sampling. Environment-specific interval overrides use the
prefix before `__` in the sequence name; unmatched names use configured
`min_interval` / `max_interval`. The public YAML does not expose `interval_by_env`.
For other sampling or supervision assumptions, add a dedicated adapter.

Validation uses only TartanGround-compatible data: 128-frame streams at strides
1 and 6. Each validation sequence needs at least 128 frames. Set held-out names
in **both** `eval_scenes` and `exclude_scenes`; the former does not remove them
from training. `null` uses the original TartanGround holdout names, normally
unsuitable for a new dataset. Adjust `eval_starts_per_scene` as needed.

Without suitable validation data, set top-level `validation_enabled: false`.
Use `exclude_scenes: []` only when intentionally training without that holdout.
Camera-only is an adapter-specific path for calibrated images and poses, not
support for arbitrary RGB-only folders.

## Enable a source

Install `.[train]`, edit [configs/finetune_cut3r.yaml](../configs/finetune_cut3r.yaml)
with the checkpoint and filesystem roots, then assign positive dataset weights.
Follow the [training commands](../README.md#start-training). The example initially
enables only TartanAir and disables validation. Weights select batches, not
sequence counts; adding a training source does not add a validation benchmark.

## Tests and attribution

Dataset tests (`tests/test_datasets_*.py`) cover layout fixtures, sampling,
geometry and training integration; they do not replace checking a real export.
See [Third-Party Notices](../THIRD_PARTY_NOTICES.md) for attribution. Code licenses
do not grant access to or redistribution rights for the source datasets.

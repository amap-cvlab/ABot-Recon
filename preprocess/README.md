# ScanNet++ sequence preprocessing

Use `preprocess_scannet_seq.py` for the sequence format expected by the
ScanNet++ Seq adapter. It renders depth from the aligned mesh, not iPhone sensor
depth, and is distinct from CUT3R's pair-oriented exporter.

## Input and installation

Use Linux with a working EGL/OpenGL renderer:

```sh
pip install -r preprocess/requirements.txt
```

Acquire ScanNet++ under its own terms. The raw root must contain
`splits/nvs_sem_train.txt` and `data/<scene_id>/` with:

- `scans/mesh_aligned_0.05.ply`;
- iPhone: `iphone/rgb/frame_*.jpg`, `iphone/rgb_masks/frame_*.png`,
  `iphone/colmap/{cameras,images}.txt`, `iphone/pose_intrinsic_imu.json`;
- DSLR: `dslr/resized_images/`, `dslr/resized_anon_masks/`,
  `dslr/colmap/{cameras,images,points3D}.txt`.

Extract official image/video and anonymization-mask assets first; the script
does not download/extract them or run COLMAP. `--stream iphone` / `--stream dslr`
requires only that device's files plus the mesh and split.

## Run

Check one scene before processing the full split:

```sh
python preprocess/preprocess_scannet_seq.py \
  --scannetpp_dir /path/to/scannetpp \
  --output_dir /path/to/processed_scannetpp_seq \
  --single_scene SCENE_ID --num_workers 1 --egl_devices 0
```

Remove `--single_scene` for the full split. Defaults: both device streams,
iPhone raw-frame stride 30, width 504 / maximum height 280, depth up to 40 m,
JPEG quality 95, and longest eligible DSLR segment of at least 100 frames.
Devices remain separate sequences. See `--help` for selection/rendering options.

### Current defaults and existing artifacts

Current defaults differ from historical training exports; a default run is not
an exact reproduction of those artifacts. Percentile clipping is removed:
`--depth_percentile` accepts only `0`, not the historical value `98`. Record input
versions and CLI settings, and use a new output directory when changing recipes.

### Output and restart behavior

```text
processed_scannetpp_seq/
  all_metadata.npz
  all_metadata.pkl
  <scene_id>_iphone/              # and/or <scene_id>_dslr/
    images/00000.jpg
    depths/00000.npy             # float32 camera-z depth, metres
    masks/00000.npy              # bool, True = invalid
    cameras/00000.npz            # camera_intrinsics, camera_pose (OpenCV C2W)
    sequence_metadata.npz
    sequence_manifest.json
    .complete.json
```

Root metadata is generated from completed sequences; no extra pair-generation
pass is needed. Set `data.scannetpp_seq.root` to this root and enable its weight.
Loader stride 1 means **processed** frames, not raw iPhone frames.

Rerun with the same paths/settings to resume. `--verify` checks and repairs frame
files; `--validate_only --verify` avoids frame rendering/writes but may update
summaries and root metadata. `--rebuild_metadata_only` rebuilds root indices
from matching completion markers. Fingerprints reject mixed settings; `--force`
can overwrite derived sequence files and is not a routine resume option.

See [Third-Party Notices](../THIRD_PARTY_NOTICES.md) for attribution.

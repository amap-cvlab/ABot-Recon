#!/usr/bin/env python3
"""Generate native-resolution sky rejection masks before offline rendering.

This is the only visualization-stage command that runs SkySeg. PNG values are
0 (keep) or 255 (reject), with thresholding and dilation already applied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from render import ReconstructionReader


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, help="default: RESULT_DIR/sky_masks")
    p.add_argument("--image-dir", type=Path, help="original inference images; otherwise use saved native RGB")
    p.add_argument("--image-start", type=int, default=0)
    p.add_argument("--image-end", type=int)
    p.add_argument("--image-stride", type=int, default=1)
    p.add_argument("--frame-list", type=Path, help="JSON list of exactly N source image paths in inference order")
    p.add_argument("--model-path", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints" / "skyseg.onnx")
    p.add_argument("--download-model", action="store_true", help="explicitly download missing weights from JianyuanWang/skyseg; respects HF_ENDPOINT")
    p.add_argument("--device", default="cpu", help="cpu or cuda:N; requested CUDA never silently falls back")
    p.add_argument("--non-sky-threshold", type=float, default=0.3, help="reject normalized non-sky scores <= this value")
    p.add_argument("--sky-dilate", type=int, default=3, help="native-pixel dilation radius; 3 gives a 7x7 ellipse")
    p.add_argument("--overwrite", action="store_true", help="replace owned masks and manifest; preserve unrelated files")
    return p


def file_identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, data):
    """Replace one owned output without exposing partial PNG/JSON contents."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".sky-mask-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_manifest(path, manifest):
    atomic_write(path, (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode())


def protect_inputs(output, paths):
    for path in paths:
        if path is not None:
            source = Path(path).resolve()
            if output == source or output in source.parents:
                raise ValueError(f"output-dir must not equal or contain an input: {source}")


def validate_output(output, indices, overwrite):
    if not output.exists():
        return set()
    if not output.is_dir():
        raise ValueError("output-dir is not a directory")
    if not any(output.iterdir()):
        return set()
    if not overwrite:
        raise FileExistsError("output-dir is not empty; use a fresh directory or --overwrite")
    # Ownership is recorded by our manifest, not inferred from arbitrary names.
    manifest_path = output / "manifest.json"
    owned = set()
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text())
            if old.get("schema_version") != 1 or old.get("mask_kind") != "sky_rejected":
                raise ValueError("unrecognized manifest")
            mapping = old["dense_output_indices"]
            if not isinstance(mapping, list) or not all(type(i) is int and i >= 0 for i in mapping):
                raise ValueError("invalid prior mapping")
            owned = {f"{i:08d}.png" for i in mapping}
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise FileExistsError("Refusing to overwrite an unrelated or invalid manifest.json") from exc
    for index in indices:
        path = output / f"{int(index):08d}.png"
        if (path.exists() or path.is_symlink()) and path.name not in owned:
            raise FileExistsError(f"Refusing to overwrite an unowned file: {path}")
    return owned


def load_session(a):
    if not a.model_path.is_file():
        if not a.download_model:
            raise FileNotFoundError(f"SkySeg weights missing: {a.model_path}; provide --model-path or explicitly use --download-model")
        from huggingface_hub import hf_hub_download
        a.model_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(hf_hub_download(repo_id="JianyuanWang/skyseg", filename="skyseg.onnx", local_dir=str(a.model_path.parent)))
        # The hub filename is fixed; honor a user-selected destination basename.
        if downloaded.resolve() != a.model_path.resolve():
            shutil.copyfile(downloaded, a.model_path)
    import onnxruntime as ort
    providers = ["CPUExecutionProvider"]
    if a.device.startswith("cuda:"):
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("SkySeg CUDA provider unavailable; install compatible onnxruntime-gpu or explicitly use --device cpu")
        providers = [("CUDAExecutionProvider", {"device_id": int(a.device.split(":", 1)[1])}), "CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(a.model_path), sess_options=options, providers=providers)
    expected = "CUDAExecutionProvider" if a.device.startswith("cuda:") else "CPUExecutionProvider"
    if session.get_providers()[0] != expected:
        raise RuntimeError(f"SkySeg unexpectedly substituted the requested {expected}")
    session.disable_fallback()  # Also prohibit retry-on-provider-error at run time.
    return session


def predict_mask(session, input_name, rgb, reader, threshold=0.3, dilation_radius=3):
    """Legacy SkySeg numerics, including exact source-to-native crop/pad mapping."""
    small = np.asarray(Image.fromarray(rgb).resize((320, 320), Image.Resampling.BILINEAR), np.float32) / 255
    small = (small - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    raw = np.asarray(session.run(None, {input_name: small.transpose(2, 0, 1)[None].astype(np.float32)})[0]).squeeze()
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("SkySeg returned invalid scores")
    sky = (raw - raw.min()) / max(float(np.ptp(raw)), 1e-6)
    if reader.paths is not None:
        _, _, _, rh, crop, pad = reader.geometry
        sky = cv2.resize(sky, (reader.w, rh), interpolation=cv2.INTER_LINEAR)
        native = np.ones((reader.h, reader.w), np.float32)
        native[pad:pad + min(rh, reader.h)] = sky[crop:crop + min(rh, reader.h)]
    else:
        native = cv2.resize(sky, (reader.w, reader.h), interpolation=cv2.INTER_LINEAR)
    rejected = (1 - native) <= threshold
    if dilation_radius:
        size = 2 * dilation_radius + 1
        rejected = cv2.dilate(rejected.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
    return rejected.astype(np.uint8) * 255


def generate(a):
    started = time.monotonic()
    a.result_dir = Path(a.result_dir).resolve()
    a.model_path = Path(a.model_path).resolve()
    output = (Path(a.output_dir) if a.output_dir is not None else a.result_dir / "sky_masks").resolve()
    if a.image_dir and a.frame_list:
        raise ValueError("image-dir and frame-list are mutually exclusive")
    if a.image_start < 0 or a.image_stride < 1 or (a.image_end is not None and a.image_end <= a.image_start):
        raise ValueError("invalid source image start/end/stride")
    if not np.isfinite(a.non_sky_threshold) or not 0 <= a.non_sky_threshold <= 1 or a.sky_dilate < 0:
        raise ValueError("threshold must be in [0,1] and sky-dilate must be nonnegative")
    if a.device != "cpu" and not (a.device.startswith("cuda:") and a.device[5:].isascii() and a.device[5:].isdigit()):
        raise ValueError("device must be cpu or cuda:N")
    protect_inputs(output, [a.result_dir, a.image_dir, a.frame_list, a.model_path])
    reader = ReconstructionReader(SimpleNamespace(
        result_dir=a.result_dir, poses="final", confidence=0,
        image_dir=a.image_dir, image_start=a.image_start, image_end=a.image_end,
        image_stride=a.image_stride, frame_list=a.frame_list, resolution="native"))
    protect_inputs(output, reader.paths or [])
    previous_owned = validate_output(output, reader.indices, a.overwrite)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": 1, "status": "running", "mask_kind": "sky_rejected",
        "native_hw": [reader.h, reader.w], "frames": len(reader.poses),
        "dense_output_indices": reader.indices.tolist(),
        "color_sha256": [], "mask_sha256": [],
        "threshold": a.non_sky_threshold, "dilation_radius": a.sky_dilate,
        "input_mode": "source" if reader.paths is not None else "native",
        "device": a.device, "provider": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": {"result_dir": str(a.result_dir), "colors": file_identity(a.result_dir / "colors.pt"),
                   "source_hw": reader.source_hw, "image_start": a.image_start,
                   "image_end": a.image_end, "image_stride": a.image_stride,
                   "frame_list": str(a.frame_list.resolve()) if a.frame_list else None,
                   "image_dir": str(a.image_dir.resolve()) if a.image_dir else None,
                   "files": [file_identity(p) for p in reader.paths] if reader.paths is not None else []},
    }
    # Invalidate an older complete manifest BEFORE touching any of its masks.
    write_manifest(manifest_path, manifest)
    try:
        current_names = {f"{int(index):08d}.png" for index in reader.indices}
        for name in previous_owned - current_names:
            # Only prior-manifest-owned names are removed. unlink never follows
            # symlinks, and unrelated contents of the output directory stay put.
            (output / name).unlink(missing_ok=True)
        if reader.paths is not None:
            from abot_recon.preprocessing import preprocess_image
        downloaded = not a.model_path.is_file() and a.download_model
        session = load_session(a)
        manifest["provider"] = session.get_providers()[0]
        manifest["model"] = {**file_identity(a.model_path), "sha256": file_sha256(a.model_path),
                             "downloaded_this_run": downloaded,
                             "download_repo": "JianyuanWang/skyseg" if downloaded else None,
                             "download_filename": "skyseg.onnx" if downloaded else None,
                             "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co") if downloaded else None}
        input_name = session.get_inputs()[0].name
        cv2.setNumThreads(1)
        inference_started = time.monotonic()
        for row, index in enumerate(reader.indices):
            colors = reader.colors[row].numpy()
            rgb = colors if reader.paths is None else reader.rgb(int(index))
            if reader.paths is not None:
                # The reader samples alignment for fast setup; generation must
                # bind EVERY source-derived mask to its published native RGB.
                expected, _ = preprocess_image(Image.fromarray(rgb), height=reader.h, width=reader.w)
                expected = (expected.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
                error = np.abs(expected.astype(float) - colors).mean()
                if error > 1.0:
                    raise ValueError(f"Source image does not match colors.pt at dense row {row} (MAE {error:.2f}); check source selection")
            mask = predict_mask(session, input_name, rgb, reader, a.non_sky_threshold, a.sky_dilate)
            encoded = BytesIO()
            Image.fromarray(mask).save(encoded, format="PNG")
            data = encoded.getvalue()
            atomic_write(output / f"{int(index):08d}.png", data)
            manifest["color_sha256"].append(hashlib.sha256(colors.tobytes()).hexdigest())
            manifest["mask_sha256"].append(hashlib.sha256(data).hexdigest())
            if (row + 1) % 100 == 0 or row + 1 == reader.m:
                print(f"Sky masks: {row + 1}/{reader.m} (pose {int(index)})", flush=True)
        manifest["inference_seconds"] = time.monotonic() - inference_started
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "complete"
        write_manifest(manifest_path, manifest)  # Commit only after every PNG exists.
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_manifest(manifest_path, manifest)
        raise
    return manifest


if __name__ == "__main__":
    generate(parser().parse_args())

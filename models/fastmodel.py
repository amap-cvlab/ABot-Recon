import inspect

import torch
import torch.nn as nn

from contextlib import contextmanager, nullcontext
from typing import Optional, Dict

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os
import sys


@contextmanager
def _temporarily_disable_dense_heads(module, head_names=("point_head", "depth_head")):
    """Disable independent dense readouts for a pose-only call, then restore them."""
    originals = {name: getattr(module, name) for name in head_names if hasattr(module, name)}
    try:
        for name in originals:
            setattr(module, name, None)
        yield
    finally:
        for name, head in originals.items():
            setattr(module, name, head)


class _DiscardedHorizonDepthHead(nn.Module):
    """Cheap shape-compatible replacement for Horizon's state-independent DPT head."""

    def forward(self, aggregated_tokens_list, *, images, **kwargs):
        batch, frames = images.shape[:2]
        depth = images.new_zeros((batch, frames, 1, 1, 1))
        confidence = images.new_zeros((batch, frames, 1, 1))
        return depth, confidence


@contextmanager
def _horizon_pose_only_depth_readout(model):
    """Skip Horizon's DPT readout without changing camera or streaming state."""
    core = getattr(model, "horizonstream", model)
    depth_head = getattr(core, "dpt_decoder", None)
    if depth_head is None:
        yield
        return
    core.dpt_decoder = _DiscardedHorizonDepthHead()
    try:
        yield
    finally:
        core.dpt_decoder = depth_head


class CUT3R(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        source_root: str = "third_party/CUT3R",
        input_size: int = 512,
    ):
        super().__init__()
        self.input_size = int(input_size)

        if pretrained_model_name_or_path is not None:
            _prepend_official_src(source_root)
            _expose_official_croco_models(source_root)
            from dust3r.model import ARCroco3DStereo

            self.model = ARCroco3DStereo.from_pretrained(pretrained_model_name_or_path)
            self.model.config.model_update_type = "cut3r"
        else:
            raise NotImplementedError

    @torch.no_grad()
    def inference_recurrent_lighter(self, groups, model, device, verbose=True):
        # if verbose:
        #     print(f">> Inference with model on {len(groups)} image/raymaps")

        with torch.cuda.amp.autocast(enabled=False):
            preds, batch, state_args = model.forward_recurrent_lighter(
                groups, device, ret_state=True
            )
            res = dict(views=batch, pred=preds)
        return res, state_args

    def forward(self, views):
        return _forward_official_cut3r(views, self.model)


class TTT3R(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        source_root: str = "third_party/TTT3R",
        input_size: int = 512,
    ):
        super().__init__()
        self.input_size = int(input_size)

        if pretrained_model_name_or_path is not None:
            _prepend_official_src(source_root)
            _expose_official_croco_models(source_root)
            from dust3r.model import ARCroco3DStereo

            self.model = ARCroco3DStereo.from_pretrained(pretrained_model_name_or_path)
            self.model.config.model_update_type = "ttt3r"
        else:
            raise NotImplementedError

    @torch.no_grad()
    def inference_recurrent_lighter(self, groups, model, device, verbose=True):
        # if verbose:
        #     print(f">> Inference with model on {len(groups)} image/raymaps")

        with torch.cuda.amp.autocast(enabled=False):
            preds, batch, state_args = model.forward_recurrent_lighter(
                groups, device, ret_state=True
            )
            res = dict(views=batch, pred=preds)
        return res, state_args

    def forward(self, views):
        return _forward_official_cut3r(views, self.model)


def _forward_official_cut3r(views, model):
    """Run the public forward path used by the official mv_recon evaluator."""
    device = next(model.parameters()).device
    ignore_keys = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
    for view in views:
        for name, value in view.items():
            if name in ignore_keys:
                continue
            if isinstance(value, (tuple, list)):
                view[name] = [item.to(device, non_blocking=True) for item in value]
            elif torch.is_tensor(value):
                view[name] = value.to(device, non_blocking=True)
    with torch.amp.autocast(device.type, enabled=False):
        output = model(views)
    return {"views": output.views, "pred": output.ress}, None


class LingBotMAP(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        source_root: str = "third_party/LingBot-Map",
        img_size: int = 518,
        patch_size: int = 14,
        mode: str = "streaming",
        enable_3d_rope: bool = True,
        max_frame_num: int = 8000,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        camera_num_iterations: int = 4,
        use_sdpa: bool = False,
        num_scale_frames: int = 8,
        keyframe_interval: Optional[int] = None,
        window_size: int = 64,
        overlap_size: int = 16,
        # Official bench preprocess (benchmark/configs/methods/lingbot_map.yaml)
        preprocess_mode: str = "area_budget",  # area_budget | crop
        area_budget: int = 255000,
        align: int = 14,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.mode = mode
        self.num_scale_frames = num_scale_frames
        self.keyframe_interval = keyframe_interval
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.preprocess_mode = preprocess_mode
        self.area_budget = int(area_budget)
        self.align = int(align)

        _prepend_official_src(source_root)

        if mode == "windowed":
            from lingbot_map.models.gct_stream_window import GCTStream
        else:
            from lingbot_map.models.gct_stream import GCTStream

        self.model = GCTStream(
            img_size=img_size,
            patch_size=patch_size,
            enable_3d_rope=enable_3d_rope,
            max_frame_num=max_frame_num,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=kv_cache_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=use_sdpa,
            camera_num_iterations=camera_num_iterations,
        )

        if pretrained_model_name_or_path is not None:
            if not os.path.exists(pretrained_model_name_or_path):
                print(f"Pretrained model path {pretrained_model_name_or_path} does not exist!")
            else:
                raw = torch.load(
                    pretrained_model_name_or_path, map_location="cpu", weights_only=True
                )
                state_dict = raw.get("state_dict", raw.get("model", raw))
                keys = list(state_dict.keys())
                if keys and all(str(k).startswith("module.") for k in keys):
                    state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"Missing keys: {len(missing)}")
                if unexpected:
                    print(f"Unexpected keys: {len(unexpected)}")
        self.model.eval()

    def reset_kv_cache_manager(self):
        """Destroy the KV cache manager so it will be re-created with the correct
        tokens_per_frame on the next forward pass.  Call this when switching to a
        dataset whose images have a different aspect ratio."""
        aggregator = getattr(self.model, "aggregator", self.model)
        if hasattr(aggregator, "kv_cache_manager"):
            aggregator.kv_cache_manager = None
        if hasattr(aggregator, "clean_kv_cache"):
            aggregator.clean_kv_cache()

    def forward(
        self,
        images: torch.Tensor,
        dense_output_indices=None,
        pose_only: bool = False,
    ):
        use_amp = images.device.type == "cuda"
        capability = (
            torch.cuda.get_device_capability(images.device)
            if images.device.type == "cuda"
            else (0, 0)
        )
        amp_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16

        # Resolve keyframe_interval from the sequence axis. Inputs may be
        # [S,C,H,W] or [B,S,C,H,W], matching GCTStream.inference_streaming.
        keyframe_interval = self.keyframe_interval
        if keyframe_interval is None:
            num_frames = images.shape[1] if images.ndim == 5 else images.shape[0]
            if self.mode == "streaming" and num_frames > 320:
                keyframe_interval = (num_frames + 319) // 320
            else:
                keyframe_interval = 1

        dense_context = _temporarily_disable_dense_heads(self.model) if pose_only else nullcontext()
        with dense_context, torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                if self.mode == "streaming":
                    parameters = inspect.signature(
                        self.model.inference_streaming
                    ).parameters
                    supports_selective_dense = "dense_output_indices" in parameters
                    inference_kwargs = {
                        "num_scale_frames": self.num_scale_frames,
                        "keyframe_interval": keyframe_interval,
                        "output_device": torch.device("cpu"),
                    }
                    if supports_selective_dense:
                        inference_kwargs["dense_output_indices"] = dense_output_indices
                    predictions = self.model.inference_streaming(
                        images.float(), **inference_kwargs
                    )
                    predictions["dense_output_indices_applied"] = bool(
                        supports_selective_dense and dense_output_indices is not None
                    )
                else:
                    predictions = self.model.inference_windowed(
                        images.float(),
                        window_size=self.window_size,
                        overlap_size=self.overlap_size,
                        num_scale_frames=self.num_scale_frames,
                    )
        return predictions


class LongStream(nn.Module):
    """Wrapper for LongStream model that preserves the full inference pipeline.

    The forward method strictly follows the original infer.py logic:
      KeyframeSelector -> run_batch_refresh/run_streaming_refresh
      -> compose_abs_from_rel -> pose_encoding_to_extri_intri (w2c output)
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        window_size: int = 48,
        keyframe_stride: int = 8,
        refresh: int = 4,
        inference_mode: str = "batch_refresh",
        streaming_mode: str = "causal",
        rel_pose_num_iterations: int = 4,
        # LongStream model architecture config
        use_role_embedding: bool = False,
        enable_scale_token: bool = True,
        disable_keyframe_distinction: bool = True,
        use_segment_mask: bool = False,
        enable_camera_head: bool = False,
        freeze: str = "none",
        use_rel_pose_head: bool = True,
        rel_pose_head_cfg: Optional[dict] = None,
        source_root: str = "third_party/LongStream",
        strict_load: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.keyframe_stride = keyframe_stride
        self.refresh = refresh
        self.inference_mode = inference_mode
        self.streaming_mode = streaming_mode
        self.window_size = window_size
        self.rel_pose_num_iterations = rel_pose_num_iterations

        import sys

        longstream_pkg = os.path.normpath(source_root)
        if longstream_pkg not in sys.path:
            sys.path.insert(0, longstream_pkg)

        import longstream

        loaded_root = os.path.realpath(longstream.__file__)
        expected_root = os.path.realpath(longstream_pkg) + os.sep
        if not loaded_root.startswith(expected_root):
            raise RuntimeError(
                f"LongStream import collision: expected {expected_root}, got {loaded_root}"
            )

        from longstream.core.model import LongStreamModel
        from longstream.utils.depth import unproject_depth_to_points
        from longstream.utils.vendor.dust3r.utils.image import load_images_for_eval

        self.image_loader = load_images_for_eval
        self.unproject_depth = unproject_depth_to_points

        # Build model config dict matching longstream_infer.yaml structure
        longstream_cfg = dict(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            window_size=window_size,
            use_role_embedding=use_role_embedding,
            enable_scale_token=enable_scale_token,
            disable_keyframe_distinction=disable_keyframe_distinction,
            use_segment_mask=use_segment_mask,
            enable_camera_head=enable_camera_head,
            freeze=freeze,
            use_rel_pose_head=use_rel_pose_head,
        )
        if use_rel_pose_head and rel_pose_head_cfg is not None:
            longstream_cfg["rel_pose_head_cfg"] = dict(rel_pose_head_cfg)

        model_cfg = {
            "checkpoint": checkpoint,
            "strict_load": bool(strict_load),
            "longstream_cfg": longstream_cfg,
        }

        self.model = LongStreamModel(model_cfg)
        self.model.eval()

    def forward(
        self,
        images: torch.Tensor,
        pose_only: bool = False,
        dense_output_indices=None,
        output_device="cpu",
        output_float32: bool = False,
        empty_cache_between_batches: bool = True,
    ):
        """Run full LongStream inference pipeline.

        Args:
            images: (B, S, C, H, W) tensor in [0, 1] range.

        Returns:
            dict with keys:
              - extrinsic_w2c: (S, 3, 4) w2c extrinsic matrices
              - intrinsic: (S, 3, 3) intrinsic matrices
              - depth: (S, H, W) depth maps (if available)
        """
        from longstream.streaming.keyframe_selector import KeyframeSelector
        from longstream.streaming.refresh import run_batch_refresh, run_streaming_refresh
        from longstream.utils.camera import compose_abs_from_rel
        from longstream.utils.vendor.models.components.utils.pose_enc import (
            pose_encoding_to_extri_intri,
        )

        if images.dim() == 4:
            images = images.unsqueeze(0)

        batch_size, num_frames, channels, height, width = images.shape

        selector = KeyframeSelector(
            min_interval=self.keyframe_stride,
            max_interval=self.keyframe_stride,
            force_first=True,
            mode="fixed",
        )
        is_keyframe, keyframe_indices = selector.select_keyframes(
            num_frames, batch_size, images.device
        )

        rel_pose_cfg = {"num_iterations": self.rel_pose_num_iterations}

        dense_context = _temporarily_disable_dense_heads(self.model) if pose_only else nullcontext()
        refresh_parameters = {}
        with dense_context, torch.no_grad():
            if self.inference_mode == "batch_refresh":
                refresh_parameters = inspect.signature(run_batch_refresh).parameters
                refresh_kwargs = {}
                if "dense_output_indices" in refresh_parameters:
                    refresh_kwargs["dense_output_indices"] = (
                        None if pose_only else dense_output_indices
                    )
                if "output_device" in refresh_parameters:
                    refresh_kwargs["output_device"] = output_device
                if "empty_cache_between_batches" in refresh_parameters:
                    refresh_kwargs["empty_cache_between_batches"] = empty_cache_between_batches
                outputs = run_batch_refresh(
                    self.model,
                    images,
                    is_keyframe,
                    keyframe_indices,
                    self.streaming_mode,
                    self.keyframe_stride,
                    self.refresh,
                    rel_pose_cfg,
                    **refresh_kwargs,
                )
            elif self.inference_mode in ("streaming_refresh", "streaming"):
                outputs = run_streaming_refresh(
                    self.model,
                    images,
                    is_keyframe,
                    keyframe_indices,
                    self.streaming_mode,
                    self.window_size,
                    self.refresh,
                    rel_pose_cfg,
                )
            else:
                raise ValueError(f"Unsupported inference mode: {self.inference_mode}")

        # Decode poses: rel_pose_enc -> abs -> w2c extrinsic + intrinsic
        if "rel_pose_enc" in outputs:
            rel_pose_enc = outputs["rel_pose_enc"][0]
            abs_pose_enc = compose_abs_from_rel(rel_pose_enc, keyframe_indices[0])
            extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(
                abs_pose_enc[None], image_size_hw=(height, width)
            )
        elif "pose_enc" in outputs:
            pose_enc = outputs["pose_enc"][0]
            extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(
                pose_enc[None], image_size_hw=(height, width)
            )
        else:
            raise RuntimeError("LongStream outputs contain neither rel_pose_enc nor pose_enc")

        def prepare_output(value: torch.Tensor) -> torch.Tensor:
            value = value.detach()
            if output_float32 and value.is_floating_point():
                value = value.float()
            return value.to(output_device)

        result = {
            "extrinsic_w2c": prepare_output(extrinsic_w2c[0]),  # (S, 3, 4)
            "intrinsic": prepare_output(intrinsic[0]),  # (S, 3, 3)
        }
        result["dense_output_indices_applied"] = (
            not pose_only
            and dense_output_indices is not None
            and "dense_output_indices" in refresh_parameters
        )

        if not pose_only and "depth" in outputs:
            depth = prepare_output(outputs["depth"][0, :, :, :, 0])
            result["depth"] = depth
        if not pose_only and "world_points" in outputs:
            result["camera_points"] = prepare_output(outputs["world_points"][0])

        return result


def _prepend_official_src(source_root: str) -> str:
    source_root = os.path.abspath(os.path.expanduser(source_root))
    src_root = os.path.join(source_root, "src")
    if os.path.isdir(src_root):
        import_root = src_root
    elif os.path.isdir(source_root):
        import_root = source_root
    else:
        raise FileNotFoundError(f"Official model source does not exist: {source_root}")
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    return source_root


def _expose_official_croco_models(source_root: str) -> None:
    """Resolve CroCo's official ``models.*`` imports alongside the evaluation package."""
    croco_models = os.path.join(os.path.abspath(source_root), "src", "croco", "models")
    if not os.path.isdir(croco_models):
        raise FileNotFoundError(f"Official CroCo models do not exist: {croco_models}")
    models_package = sys.modules.get("models")
    package_path = getattr(models_package, "__path__", None)
    if package_path is None:
        raise RuntimeError("The evaluation models package is not initialized")
    if croco_models not in package_path:
        package_path.append(croco_models)


class InfiniteVGGTEval(nn.Module):
    """Pinned official InfiniteVGGT with its native pruning budget."""

    family = "infinitevggt"

    def __init__(
        self,
        checkpoint: str,
        source_root: str,
        img_size: int = 518,
        patch_size: int = 14,
        total_budget: int = 1_200_000,
        preprocess_mode: str = "crop",
        strict_load: bool = True,
    ):
        super().__init__()
        _prepend_official_src(source_root)
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.load_fn import load_and_preprocess_images
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.preprocess_mode = preprocess_mode
        self.image_loader = load_and_preprocess_images
        self.pose_decoder = pose_encoding_to_extri_intri
        self.model = StreamVGGT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            total_budget=int(total_budget),
        )
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state, strict=bool(strict_load))
        del state
        self.model.eval()


class OVGGTOfficialEval(nn.Module):
    """Pinned official OVGGT with released cache and anchor defaults."""

    family = "ovggt"

    def __init__(
        self,
        checkpoint: str,
        source_root: str,
        img_size: int = 518,
        patch_size: int = 14,
        total_budget: int = 200_000,
        camera_budget: int = 384,
        eviction_strategy: str = "repr_shift_spatial",
        intra_frame_keep_ratio: float = 1.0,
        spatial_alpha: float = 0.5,
        importance_weight: float = 0.5,
        preprocess_mode: str = "crop",
        strict_load: bool = True,
    ):
        super().__init__()
        _prepend_official_src(source_root)
        from ovggt.models.ovggt import OVGGT
        from ovggt.utils.load_fn import load_and_preprocess_images
        from ovggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.preprocess_mode = preprocess_mode
        self.image_loader = load_and_preprocess_images
        self.pose_decoder = pose_encoding_to_extri_intri
        self.model = OVGGT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            total_budget=int(total_budget),
            camera_budget=int(camera_budget),
            eviction_strategy=eviction_strategy,
            intra_frame_keep_ratio=float(intra_frame_keep_ratio),
            spatial_alpha=float(spatial_alpha),
            importance_weight=float(importance_weight),
        )
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state, strict=bool(strict_load))
        del state
        self.model.eval()


class STream3ROfficialEval(nn.Module):
    """Official STream3R causal/window inference with bounded RGB chunks."""

    family = "stream3r"

    def __init__(
        self,
        checkpoint: str,
        source_root: str,
        mode: str = "causal",
        chunk_size: int = 8,
        img_size: int = 518,
        patch_size: int = 14,
        preprocess_mode: str = "crop",
    ):
        super().__init__()
        if mode not in {"causal", "window"}:
            raise ValueError(f"STream3R eval mode must be causal or window, got {mode}")
        _prepend_official_src(source_root)
        from stream3r.models.stream3r import STream3R
        from stream3r.models.components.utils.geometry import unproject_depth_map_to_point_map
        from stream3r.models.components.utils.load_fn import load_and_preprocess_images
        from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
        from stream3r.stream_session import StreamSession

        checkpoint_dir = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
        self.model = STream3R.from_pretrained(checkpoint_dir)
        self.model.eval()
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.preprocess_mode = preprocess_mode
        self.image_loader = load_and_preprocess_images
        self.pose_decoder = pose_encoding_to_extri_intri
        self.unproject = unproject_depth_map_to_point_map
        self._session_type = StreamSession

    def make_session(self):
        return self._session_type(self.model, self.mode)


class HorizonStreamEval(nn.Module):
    """Evaluation wrapper for HorizonStream official release.

    Uses the official streaming chunk schedule (window_size / sliding_size) and
    online motion-averaged camera maps, then exposes depth + w2c for PC eval.
    """

    def __init__(
        self,
        checkpoint: str,
        horizonstream_root: Optional[str] = None,
        img_size: int = 518,
        patch_size: int = 14,
        crop: bool = True,
        window_size: int = 10,
        sliding_size: int = 21,
        abs_pose_source: str = "online",
        enable_offline_motion_averaging: bool = False,
        strict_load: bool = False,
        amp_dtype: str = "auto",
        horizonstream_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.crop = bool(crop)
        self.window_size = int(window_size)
        self.sliding_size = int(sliding_size)
        self.abs_pose_source = str(abs_pose_source).strip().lower()
        self.enable_offline_motion_averaging = bool(enable_offline_motion_averaging)
        self.amp_dtype = str(amp_dtype).strip().lower()
        if self.amp_dtype not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError(f"amp_dtype must be one of auto, bf16, fp16, fp32; got {amp_dtype!r}")
        self.pretrained_model_name_or_path = checkpoint

        if horizonstream_root is None:
            horizonstream_root = os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "..",
                    "HorizonStream",
                )
            )
        self.horizonstream_root = os.path.normpath(horizonstream_root)

        import sys

        if self.horizonstream_root not in sys.path:
            sys.path.insert(0, self.horizonstream_root)

        from horizonstream.core.model import HorizonStreamModel
        from horizonstream.utils.vendor.dust3r.utils.image import load_images_for_eval

        self.image_loader = load_images_for_eval

        if horizonstream_cfg is None:
            # Defaults mirror configs/horizonstream_infer.yaml
            horizonstream_cfg = dict(
                load_pretrained=False,
                use_xyz_head=False,
                enable_metric_readout_token=True,
                frames_chunk_size=1,
                use_chunkwise_checkpoint=True,
                agg_regator_cfg=dict(
                    patch_embed="dinov2_vitl14_reg",
                    img_size=self.img_size,
                    patch_size=self.patch_size,
                    embed_dim=1024,
                    intermediate_layer_idx=[4, 11, 17, 23],
                    num_heads=16,
                    depth=24,
                    gate_attn="headwise",
                    rope_type="3d",
                    rope_temporal_period=21,
                    fuse_grm_priori=False,
                    num_pose_tokens=32,
                    use_register_token=False,
                    chunk_block_num=3,
                    global_attn_arch="gla",
                    global_attn_impl="softmax",
                    gla_expand_v=1.0,
                    gla_use_short_conv=True,
                    gla_conv_size=4,
                    gla_conv_bias=False,
                    gla_allow_neg_eigval=False,
                    gla_norm_eps=1.0e-5,
                    gla_serial_layers=[4, 11, 17, 23],
                ),
                cam_decoder_cfg=dict(
                    pose_encoding_type="absT_quaR_FoV",
                    dim_in=2048,
                ),
                metric_readout_head_cfg=dict(
                    hidden_dims=[256, 128],
                    init_bias=0.0,
                ),
                dpt_decoder_cfg=dict(
                    dim_in=2048,
                    output_dim=2,
                    activation="exp",
                    conf_activation="expp1",
                    intermediate_layer_idx=[4, 11, 17, 23],
                ),
            )

        model_cfg = {
            "checkpoint": checkpoint,
            "strict_load": bool(strict_load),
            "horizonstream_cfg": dict(horizonstream_cfg),
        }
        self.model = HorizonStreamModel(model_cfg)
        self.model.eval()

    @staticmethod
    def _chunk_schedule(num_frames: int, window_size: int, sliding_size: int):
        if num_frames <= 0:
            return []
        if num_frames <= window_size:
            return [(0, num_frames)]
        chunks = [(0, window_size)]
        start = window_size
        while start < num_frames:
            end = min(start + sliding_size, num_frames)
            chunks.append((start, end))
            start = end
        return chunks

    def _autocast_settings(self, device: torch.device):
        if device.type != "cuda" or self.amp_dtype == "fp32":
            return False, torch.float32
        if self.amp_dtype == "bf16":
            return True, torch.bfloat16
        if self.amp_dtype == "fp16":
            return True, torch.float16
        capability = torch.cuda.get_device_capability(device)
        return True, torch.bfloat16 if capability[0] >= 8 else torch.float16

    def forward(
        self,
        images: torch.Tensor,
        pose_only: bool = False,
        dense_output_indices=None,
        output_device="cpu",
        output_float32: bool = True,
        run_pose_postprocess: bool = True,
        emit_progress: bool = True,
    ):
        """Run HorizonStream streaming inference.

        Args:
            images: (B, S, C, H, W) or (S, C, H, W) in [0, 1].

        Returns:
            dict with extrinsic_w2c (S,3,4), intrinsic (S,3,3), depth (S,H,W).
        """
        from horizonstream.runtime.motion_averaging import compute_motion_averaged_camera_maps
        from horizonstream.utils.vendor.models.components.utils.pose_enc import (
            pose_encoding_to_extri_intri,
        )

        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.shape[0] != 1:
            raise ValueError("HorizonStreamEval currently supports batch_size=1")

        device = images.device
        result_device = device if output_device is None else torch.device(output_device)
        _, num_frames, _, height, width = images.shape
        chunks = self._chunk_schedule(num_frames, self.window_size, self.sliding_size)

        use_amp, amp_dtype = self._autocast_settings(device)

        depth_context = _horizon_pose_only_depth_readout(self.model) if pose_only else nullcontext()
        with depth_context, torch.no_grad():
            if len(chunks) == 1:
                with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                    outputs = self.model.forward_window(images.float())
                pose_enc = outputs["pose_enc"]
                depth = None if pose_only else outputs["depth"]
                if depth is not None and dense_output_indices is not None:
                    select = torch.as_tensor(
                        dense_output_indices, device=depth.device, dtype=torch.long
                    )
                    depth = depth.index_select(1, select)
            else:
                state = self.model.build_sequence_state()
                depth_parts = [] if not pose_only else None
                chunk_cam_maps = [] if run_pose_postprocess else None
                for chunk_idx, (start, end) in enumerate(chunks):
                    chunk_images = images[:, start:end]
                    current_window_size = end - start if chunk_idx == 0 else self.window_size
                    with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                        outputs = self.model.forward_chunk(
                            chunk_images.float(),
                            window_size=current_window_size,
                            chunk_idx=chunk_idx,
                            state=state,
                        )
                    if depth_parts is not None:
                        if dense_output_indices is None:
                            depth_parts.append(outputs["depth"].detach().to(result_device))
                        else:
                            local = [
                                index - start
                                for index in dense_output_indices
                                if start <= index < end
                            ]
                            if local:
                                select = torch.as_tensor(local, device=outputs["depth"].device)
                                depth_parts.append(
                                    outputs["depth"]
                                    .index_select(1, select)
                                    .detach()
                                    .to(result_device)
                                )
                    if "chunk_cam_map" not in outputs:
                        raise RuntimeError("HorizonStream chunk omitted chunk_cam_map")
                    if chunk_cam_maps is not None:
                        chunk_cam_maps.append(
                            outputs["chunk_cam_map"].detach().to(dtype=torch.float32)
                        )
                    self.model.advance_sequence_state(
                        state, is_last_chunk=chunk_idx == len(chunks) - 1
                    )
                    if emit_progress:
                        print(
                            f"[HorizonStream] pose inference: {end}/{num_frames} frames "
                            f"({chunk_idx + 1}/{len(chunks)} chunks)",
                            flush=True,
                        )
                depth = torch.cat(depth_parts, dim=1) if depth_parts is not None else None
                if run_pose_postprocess:
                    motion_maps = compute_motion_averaged_camera_maps(
                        chunk_cam_maps,
                        frames_num=int(num_frames),
                        window_size=self.window_size,
                        dtype=torch.float32,
                        enable_offline=self.enable_offline_motion_averaging,
                    )
                    if (
                        self.abs_pose_source == "offline"
                        and motion_maps.get("offline_cam_map") is not None
                    ):
                        pose_enc = motion_maps["offline_cam_map"]
                    else:
                        pose_enc = motion_maps["online_cam_map"]
                    pose_enc = pose_enc.to(device=device)
                else:
                    pose_enc = None

        if not run_pose_postprocess:
            result = {"raw_pose_frames": int(num_frames)}
            if depth is not None:
                depth = depth[0]
                if depth.dim() == 4 and depth.shape[-1] == 1:
                    depth = depth[..., 0]
                result["depth"] = depth.detach().to(result_device)
            return result

        extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(height, width)
        )
        result = {
            "extrinsic_w2c": extrinsic_w2c[0].detach().to(result_device),
            "intrinsic": intrinsic[0].detach().to(result_device),
        }
        if depth is not None:
            depth = depth[0]
            if depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth = depth.detach()
            if output_float32:
                depth = depth.float()
            result["depth"] = depth.to(result_device)
        return result

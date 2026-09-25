import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope, causal_frame_attention_mask
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .layers.adjacent_pose_head import AdjacentPoseHead
from .depth_utils import depth_from_log_z
from .dinov2.hub.backbones import dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file


def _load_raw_pi3_checkpoint(ckpt_path: str):
    """Load trusted local PyTorch or safetensors weights onto the CPU."""
    ckpt_path = str(ckpt_path)
    if "://" in ckpt_path:
        raise ValueError(
            "Checkpoint URIs are not supported by this loader. "
            "Download the checkpoint first and pass a local filesystem path."
        )
    if ckpt_path.lower().endswith(".safetensors"):
        return load_file(ckpt_path, device="cpu")
    return torch.load(ckpt_path, weights_only=False, map_location="cpu")


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def _resolve_confidence_training_flags(
    *,
    train_conf: bool,
    enable_confidence: Optional[bool],
    confidence_only_train: Optional[bool],
) -> Tuple[bool, bool]:
    """Resolve new confidence controls while preserving legacy ``train_conf`` semantics.

    Historically, ``train_conf=True`` both created the confidence branch and froze
    every prediction branch except confidence.  New training recipes need to enable
    confidence during joint full fine-tuning, so construction and freezing are now
    separate opt-in controls.  Leaving both new arguments unset is exactly backward
    compatible.
    """

    legacy_train_conf = bool(train_conf)
    enabled = legacy_train_conf if enable_confidence is None else bool(enable_confidence)
    confidence_only = (
        legacy_train_conf
        if confidence_only_train is None
        else bool(confidence_only_train)
    )
    if confidence_only and not enabled:
        raise ValueError(
            "confidence_only_train=True requires enable_confidence=True "
            "(or legacy train_conf=True)."
        )
    return enabled, confidence_only

class Pi3(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_vggt=True,
            freeze_encoder=True,
            freeze_prediction_heads=False,
            use_global_points=False,
            train_conf=False,
            enable_confidence: Optional[bool] = None,
            confidence_only_train: Optional[bool] = None,
            init_conf_decoder_from_point: bool = False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            decoder_depth_override: Optional[int] = None,
            causal_global_attn: bool = True,
            use_packaged_flash_attn: bool = False,
            camera_pose_mode: str = "absolute",
            relative_camera_head_cfg: Optional[Dict[str, Any]] = None,
            point_z_log_max: Optional[float] = None,
            _deferred_ckpt_available: bool = False,
        ):
        super().__init__()
        (
            self.enable_confidence,
            self.confidence_only_train,
        ) = _resolve_confidence_training_flags(
            train_conf=train_conf,
            enable_confidence=enable_confidence,
            confidence_only_train=confidence_only_train,
        )
        # Backward-compatible attribute used by existing forward/inference paths.
        self.train_conf = self.enable_confidence
        self.init_conf_decoder_from_point = bool(init_conf_decoder_from_point)
        if (
            self.confidence_only_train
            and ckpt is None
            and not bool(_deferred_ckpt_available)
        ):
            raise ValueError(
                "Confidence-only training requires model.ckpt. ABot-Recon may defer "
                "that checkpoint load until its extra modules are registered."
            )
        # Hydra passes all keys under cfg.model into __init__; trainer may also read cfg.model.causal_global_attn.
        self.causal_global_attn = causal_global_attn
        self.point_z_log_max = None if point_z_log_max is None else float(point_z_log_max)
        self.camera_pose_mode = str(camera_pose_mode).lower()
        if self.camera_pose_mode not in ("absolute", "relative_adjacent"):
            raise ValueError(
                "camera_pose_mode must be 'absolute' or 'relative_adjacent', "
                f"got {camera_pose_mode!r}"
            )

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            # pos_embed always defines RoPE2D (CUDA ext or PyTorch fallback); no ImportError.
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError

        if decoder_depth_override is not None:
            d_ = int(decoder_depth_override)
            if d_ <= 0 or d_ % 2 != 0:
                raise ValueError("decoder_depth_override must be a positive even number")
            dec_depth = d_

        attn_cls = (
            partial(FlashAttentionRope, use_packaged_flash_attn=True)
            if use_packaged_flash_attn
            else FlashAttentionRope
        )
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=attn_cls,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.relative_camera_head_type = "none"
        if self.camera_pose_mode == "relative_adjacent":
            rel_cfg = dict(relative_camera_head_cfg or {})
            head_type = str(rel_cfg.pop("head_type", "token_pair")).lower()
            if head_type != "token_pair":
                raise ValueError(
                    "relative_camera_head_cfg.head_type must be 'token_pair', "
                    f"got {head_type!r}"
                )
            self.camera_head = AdjacentPoseHead(dim=512, **rel_cfg)
            self.relative_camera_head_type = head_type
        else:
            self.camera_head = CameraHead(dim=512)
        

        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        if load_vggt:
            vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1 

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

        if self.enable_confidence:
            # ----------------------
            #     Conf Decoder
            # ----------------------
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        if freeze_encoder:
            print("Freezing the encoder.", flush=True)
            freeze_all_params([self.encoder])
            print("[Pi3] encoder requires_grad set (freeze done).", flush=True)

        if freeze_prediction_heads:
            if self.confidence_only_train:
                print(
                    "[Pi3] freeze_prediction_heads=True is redundant during confidence-only training "
                    "(conf_decoder / conf_head must remain trainable).",
                    flush=True,
                )
            else:
                head_modules = [
                    self.point_decoder,
                    self.point_head,
                    self.camera_decoder,
                    self.camera_head,
                    self.register_token,
                ]
                if self.use_global_points:
                    head_modules.extend(
                        [self.global_points_decoder, self.global_point_head]
                    )
                freeze_all_params(head_modules)
                print(
                    "[Pi3] freeze_prediction_heads: frozen register_token, point/camera "
                    "(and global point) heads; trainable: mainly self.decoder "
                    f"(encoder frozen={freeze_encoder}).",
                    flush=True,
                )

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if ckpt is not None:
            ckpt_path = str(ckpt)
            print(
                "[Pi3] Reading checkpoint (local path; can be slow)…\n"
                f"      path={ckpt_path}",
                flush=True,
            )
            checkpoint = _load_raw_pi3_checkpoint(ckpt_path)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                elif "model" in checkpoint and isinstance(
                    checkpoint["model"], dict
                ):
                    checkpoint = checkpoint["model"]

            checkpoint_has_conf_decoder = any(
                key.startswith("conf_decoder.") or ".conf_decoder." in key
                for key in checkpoint.keys()
            )
            print("[Pi3] Applying state_dict to model…", flush=True)
            res = self.load_state_dict(checkpoint, strict=False)
            if (
                self.enable_confidence
                and self.init_conf_decoder_from_point
                and not checkpoint_has_conf_decoder
            ):
                self._initialize_conf_decoder_from_point()
            print(f"[Pi3] Load checkpoints from {ckpt}: {res}", flush=True)

            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("[Pi3] model.ckpt is None — no pretrained file load.", flush=True)

        if self.confidence_only_train:
            self._freeze_for_confidence_only()

    def _initialize_conf_decoder_from_point(self) -> None:
        """Warm-start a missing confidence decoder from the trained point decoder."""

        res = self.conf_decoder.load_state_dict(self.point_decoder.state_dict(), strict=True)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(
                "conf_decoder and point_decoder are expected to be isomorphic; "
                f"got {res}"
            )
        print(
            "[Pi3] Checkpoint has no conf_decoder weights; initialized conf_decoder "
            "from the loaded point_decoder.",
            flush=True,
        )

    def _freeze_for_confidence_only(self) -> Tuple[str, ...]:
        """Freeze the complete model except ``conf_decoder`` and ``conf_head``."""

        prefixes = ("conf_decoder.", "conf_head.")
        trainable = []
        for name, param in self.named_parameters():
            should_train = name.startswith(prefixes)
            param.requires_grad_(should_train)
            if should_train:
                trainable.append(name)
        if not trainable:
            raise RuntimeError(
                "Confidence-only training found no conf_decoder/conf_head parameters."
            )
        leaked = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and not name.startswith(prefixes)
        ]
        if leaked:
            raise RuntimeError(
                "Confidence-only parameter freeze leaked non-confidence parameters: "
                + ", ".join(leaked[:16])
            )
        print(
            "[Pi3] Confidence-only training: trainable parameters are restricted to "
            f"conf_decoder/conf_head ({sum(self.get_parameter(n).numel() for n in trainable) / 1e6:.3f}M).",
            flush=True,
        )
        return tuple(trainable)

    @property
    def num_global_decoder_blocks(self) -> int:
        return len(self.decoder) // 2

    def decode(
        self,
        hidden,
        N,
        H,
        W,
        causal_global: bool = True,
        stream_use_cache: bool = False,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N, *self.register_token.shape[-2:]
        )

        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith("rope"):
            pos = self.position_getter(
                B * N, H // self.patch_size, W // self.patch_size, hidden.device
            )

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        if stream_use_cache and past_key_values is None:
            past_key_values = [None] * self.num_global_decoder_blocks

        global_blk_idx = 0

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            attn_mask = None
            if i % 2 == 1 and causal_global and not stream_use_cache:
                attn_mask = causal_frame_attention_mask(
                    N, hw, hidden.dtype, hidden.device
                )

            use_blk_cache = stream_use_cache and (i % 2 == 1)
            past_kv = (
                past_key_values[global_blk_idx]
                if use_blk_cache and past_key_values is not None
                else None
            )

            if use_blk_cache:
                hidden, new_kv = blk(
                    hidden,
                    xpos=pos,
                    attn_mask=None,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                assert past_key_values is not None
                past_key_values[global_blk_idx] = new_kv
            elif (
                i >= self.num_dec_blk_not_to_checkpoint
                and self.training
                and not stream_use_cache
            ):
                # Same as Pi3_train: `checkpoint(blk, hidden, xpos=pos, ...)`.
                # Do NOT wrap in `lambda ... xpos=pos ...`: `pos` is reassigned every iteration;
                # Python late-binds `pos` so backward recompute sees the wrong shape (725 vs N*hw).
                hidden = checkpoint(
                    blk,
                    hidden,
                    xpos=pos,
                    attn_mask=attn_mask,
                    past_key_values=None,
                    use_cache=False,
                    use_reentrant=False,
                )
            else:
                hidden = blk(
                    hidden,
                    xpos=pos,
                    attn_mask=attn_mask,
                    past_key_values=None,
                    use_cache=False,
                )

            if i % 2 == 1:
                global_blk_idx += 1

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        fused = torch.cat([final_output[0], final_output[1]], dim=-1)
        pos_out = pos.reshape(B * N, hw, -1)
        return fused, pos_out, past_key_values

    def _predict_camera_poses(
        self,
        camera_tokens: torch.Tensor,
        *,
        B: int,
        N: int,
        patch_h: int,
        patch_w: int,
        camera_state: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        if self.camera_pose_mode == "relative_adjacent":
            feat = camera_tokens.reshape(B, N, -1, camera_tokens.shape[-1])
            return self.camera_head(
                feat,
                camera_state=camera_state,
                return_state=return_state,
            )
        poses = self.camera_head(camera_tokens, patch_h, patch_w).reshape(B, N, 4, 4)
        if return_state:
            return poses, None
        return poses

    def forward(
        self,
        imgs: torch.Tensor,
        causal_global_attn: Optional[bool] = None,
        stream_use_cache: bool = False,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        ref_hidden: Optional[torch.Tensor] = None,
        camera_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if causal_global_attn is None:
            causal_global_attn = self.causal_global_attn

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        if stream_use_cache and N != 1:
            raise ValueError("stream_use_cache requires N==1 (single-frame steps with KV cache)")
        patch_h, patch_w = H // 14, W // 14
        tks = patch_h * patch_w + self.patch_start_idx

        imgs_bn = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs_bn, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos, past_key_values = self.decode(
            hidden,
            N,
            H,
            W,
            causal_global=causal_global_attn,
            stream_use_cache=stream_use_cache,
            past_key_values=past_key_values,
        )

        new_ref_hidden: Optional[torch.Tensor] = None
        if self.use_global_points:
            if stream_use_cache and N == 1:
                hf = hidden.reshape(B, tks, -1)
                if ref_hidden is None:
                    context = hf
                    new_ref_hidden = hf.detach()
                else:
                    context = ref_hidden
                    new_ref_hidden = ref_hidden
            else:
                context = (
                    hidden.reshape(B, N, tks, -1)[:, 0:1]
                    .repeat(1, N, 1, 1)
                    .reshape(B * N, tks, -1)
                )
            global_point_hidden = self.global_points_decoder(
                hidden, context, xpos=pos, ypos=pos
            )

        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            point_hidden = point_hidden.float()
            ret = self.point_head(
                [point_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = depth_from_log_z(z, self.point_z_log_max)
            local_points = torch.cat([xy * z, z], dim=-1)

            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head(
                    [conf_hidden[:, self.patch_start_idx :]], (H, W)
                ).reshape(B, N, H, W, -1)
            else:
                conf = None

            camera_hidden = camera_hidden.float()
            camera_tokens_for_head = camera_hidden[:, self.patch_start_idx :]
            if (
                self.camera_pose_mode == "relative_adjacent"
                and self.relative_camera_head_type == "token_pair"
            ):
                camera_tokens_for_head = camera_hidden
            camera_poses, new_camera_state = self._predict_camera_poses(
                camera_tokens_for_head,
                B=B,
                N=N,
                patch_h=patch_h,
                patch_w=patch_w,
                camera_state=camera_state,
                return_state=True,
            )

            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head(
                    [global_point_hidden[:, self.patch_start_idx :]], (H, W)
                ).reshape(B, N, H, W, -1)
            else:
                global_points = None

            points = torch.einsum(
                "bnij, bnhwj -> bnhwi",
                camera_poses,
                homogenize_points(local_points),
            )[..., :3]

        out: Dict[str, Any] = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points,
        )
        if stream_use_cache:
            out["past_key_values"] = past_key_values
        if new_ref_hidden is not None:
            out["ref_hidden"] = new_ref_hidden
        if new_camera_state is not None:
            out["camera_state"] = new_camera_state
        return out

    def inference_stream(
        self,
        frames: List[torch.Tensor],
        return_stacked: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * self.num_global_decoder_blocks
        ref_hidden: Optional[torch.Tensor] = None
        camera_state: Optional[Dict[str, torch.Tensor]] = None
        outputs: List[Dict[str, Any]] = []
        for fr in frames:
            x = fr
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.unsqueeze(1)
            pred = self.forward(
                x,
                causal_global_attn=True,
                stream_use_cache=True,
                past_key_values=past_key_values,
                ref_hidden=ref_hidden,
                camera_state=camera_state,
            )
            past_key_values = pred["past_key_values"]
            ref_hidden = pred.get("ref_hidden")
            camera_state = pred.get("camera_state", camera_state)
            clean = {
                k: v
                for k, v in pred.items()
                if k not in ("past_key_values", "ref_hidden", "camera_state")
            }
            outputs.append(clean)
        if not return_stacked:
            return outputs
        stacked: Dict[str, Any] = {}
        for k in outputs[0].keys():
            vals = [o[k] for o in outputs]
            if vals[0] is None:
                stacked[k] = None
            else:
                stacked[k] = torch.cat(vals, dim=1)
        return stacked

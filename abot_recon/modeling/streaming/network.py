"""ABot-Recon network with causal windowed attention and 3D rotary positions."""

from __future__ import annotations

import os
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint

from abot_recon.modeling.pi3.models.pi3 import Pi3, _load_raw_pi3_checkpoint
from abot_recon.modeling.pi3.models.depth_utils import depth_from_log_z
from abot_recon.modeling.pi3.models.layers.attention import (
    FlashAttentionRope,
    causal_frame_attention_mask,
)
from abot_recon.modeling.pi3.utils.geometry import homogenize_points

from abot_recon.modeling.streaming.attention import (
    StreamingFlashAttention,
)
from abot_recon.modeling.streaming.sdpa import (
    stream_window_attention,
)
from abot_recon.modeling.streaming.rope3d import RoPE3D
from abot_recon.modeling.streaming.kv_state import (
    StreamingKVState,
    prune_window_carry_pk_pv,
    streamed_chunk_key_positions_pi3,
    streaming_carry_from_rect_kv,
)
from abot_recon.modeling.streaming.core.cache_utils import detach_carry
from abot_recon.modeling.streaming.core.chunk_attention import kv_carry_fingerprint_for_bias_cache


def _log_pi3_streaming_attn_mode(
    *,
    memory_mode: str,
    temporal_window_frames: int,
    num_reference_frames: int,
    num_summary_tokens: int,
    global_pos_encoding: str,
) -> None:
    print(
        f"[ABotReconNetwork] attention=causal-window "
        f"window_frames={temporal_window_frames} "
        f"position_encoding={global_pos_encoding}",
        flush=True,
    )


class ABotReconNetwork(Pi3):
    """Checkpoint-exact ABot-Recon network for causal frame-by-frame inference.

    Global decoder blocks retain a fixed local K/V window, while the camera
    head predicts adjacent relative transforms and composes them online.
    """

    def __init__(self, **kwargs):
        # Attention gates are registered after the base network is built, so a
        # constructor-provided checkpoint must be deferred until that point.
        _deferred_ckpt = kwargs.pop("ckpt", None)
        # ── Paged KV (FlashInfer) streaming path ──────────────────────────────────
        # Default ON; falls back to _inference_stream_sdpa if unavailable.
        self.use_paged_kv = bool(kwargs.pop("use_paged_kv", True))
        self.paged_max_total_frames = int(kwargs.pop("paged_max_total_frames", 4096))
        self.paged_max_summary_frames = int(kwargs.pop("paged_max_summary_frames", 0))
        self.paged_force_fp32 = bool(kwargs.pop("paged_force_fp32", False))
        self._paged_manager = None
        self._paged_manager_sig: Optional[Tuple[int, int, int, int, str, str]] = None
        infer_mode_kw = kwargs.pop("infer_mode", kwargs.pop("stream_infer_mode", "full"))
        self.infer_mode = str(infer_mode_kw).lower()
        if self.infer_mode not in ("full", "stream"):
            raise ValueError(f"infer_mode must be 'full' or 'stream', got {self.infer_mode!r}")
        # Back-compat YAML may still emit this key; discard.
        kwargs.pop("stream_frame_max_history_frames", None)

        self.memory_mode = str(kwargs.pop("memory_mode", "streaming")).lower()
        if self.memory_mode not in ("streaming", "window"):
            raise ValueError(
                f"memory_mode must be 'streaming' or 'window', got {self.memory_mode!r}"
            )
        self.num_reference_frames = int(kwargs.pop("num_reference_frames", 0))

        _lwf = kwargs.pop("local_window_frames", None)
        if _lwf is None:
            raise ValueError(
                "ABotReconNetwork requires `local_window_frames` (int). "
                "training runtime sets it from the first element of `train.local_window_sample_values`; "
                "for scripts/tests, pass local_window_frames=… to hydra.instantiate."
            )
        self.local_window_frames = int(_lwf)

        _gate_layers_cfg = kwargs.pop("gate_layers", None) or []

        _num_summary = kwargs.pop("num_summary_tokens", 0)
        self.global_pos_encoding = str(kwargs.pop("global_pos_encoding", "pi3_2d")).lower()
        rope3d_raw = kwargs.pop("rope3d_config", None)
        if rope3d_raw is not None and isinstance(rope3d_raw, DictConfig):
            rope3d_raw = OmegaConf.to_container(rope3d_raw, resolve=True)
        self._rope3d_cfg: Optional[Dict[str, Any]] = rope3d_raw

        super().__init__(
            ckpt=None,
            _deferred_ckpt_available=_deferred_ckpt is not None,
            **kwargs,
        )

        if _num_summary is None:
            self.streaming_num_summary_tokens = int(self.patch_start_idx)
        elif type(_num_summary) is int:
            self.streaming_num_summary_tokens = _num_summary
        else:
            raise TypeError(
                "ABotReconNetwork: `num_summary_tokens` must be YAML null/absent or a plain int "
                f"(got {type(_num_summary).__name__}: {_num_summary!r})."
            )

        self.rope3d_embed: Optional[RoPE3D] = None
        if self.global_pos_encoding == "rope3d":
            if self._rope3d_cfg is None:
                raise ValueError(
                    "global_pos_encoding=rope3d requires model.rope3d_config config"
                )
            blk1 = self.decoder[1]
            old_attn = blk1.attn
            head_dim = old_attn.qkv.in_features // old_attn.num_heads
            cfg = self._rope3d_cfg
            fhw = cfg.get("fhw_dim")
            if fhw is None:
                raise ValueError("rope3d_config.fhw_dim is required (sum must equal head_dim)")
            fhw_t = tuple(int(x) for x in fhw)
            if sum(fhw_t) != head_dim:
                raise ValueError(
                    f"rope3d_config.fhw_dim sum {sum(fhw_t)} != head_dim {head_dim} "
                    f"(dec_embed_dim={self.dec_embed_dim}, num_heads={old_attn.num_heads})"
                )
            self.rope3d_embed = RoPE3D(
                attention_head_dim=head_dim,
                patch_size=(1, int(self.patch_size), int(self.patch_size)),
                max_seq_len=int(cfg.get("max_seq_len", 2048)),
                theta=float(cfg.get("theta", 10000.0)),
                fhw_dim=fhw_t,
            )

        for i in range(1, len(self.decoder), 2):
            blk = self.decoder[i]
            old = blk.attn
            if not isinstance(old, FlashAttentionRope):
                raise TypeError(f"Expected FlashAttentionRope at decoder[{i}]")
            blk.attn = StreamingFlashAttention.from_flash_attn(
                old,
                memory_mode=self.memory_mode,
                num_reference_frames=self.num_reference_frames,
                local_window_frames=self.local_window_frames,
                num_summary_tokens=self.streaming_num_summary_tokens,
                global_pos_encoding=self.global_pos_encoding,
                rope3d_embed=self.rope3d_embed,
            )

        _log_pi3_streaming_attn_mode(
            memory_mode=self.memory_mode,
            temporal_window_frames=self.local_window_frames,
            num_reference_frames=self.num_reference_frames,
            num_summary_tokens=self.streaming_num_summary_tokens,
            global_pos_encoding=self.global_pos_encoding,
        )


        self._gate_layers = [int(i) for i in _gate_layers_cfg]
        if self._gate_layers:
            for i in self._gate_layers:
                if i < 0 or i >= len(self.decoder):
                    raise ValueError(
                        f"gate_layers index {i} out of range [0, {len(self.decoder) - 1}]"
                    )
                blk = self.decoder[i]
                blk.attn.use_gate = True
                blk.attn.gate_proj = torch.nn.Linear(
                    self.dec_embed_dim, self.dec_embed_dim, bias=False
                )
            try:
                import torch.distributed as dist

                if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                    print(
                        f"[ABotReconNetwork] gate_layers={self._gate_layers} "
                        f"(elementwise, bias=True, init bias=5.0)",
                        flush=True,
                    )
            except Exception:
                pass

        if _deferred_ckpt is not None:
            self._load_deferred_streaming_checkpoint(_deferred_ckpt)

        # ABot-Recon registers/replaces attention, gate modules
        # after the base Pi3 constructor. Re-apply the confidence-only allowlist
        # here so none of those late-created parameters can accidentally train.
        if self.confidence_only_train:
            self._freeze_for_confidence_only()

        self._default_local_window_frames = int(self.local_window_frames)

    @property
    def stream_infer_mode(self) -> str:
        """Backward compat for configs / logs (``full`` | ``stream``)."""
        return self.infer_mode

    def _postprocess_stream_carry(
        self,
        carry: Optional[List[Optional[Any]]],
        *,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Optional[List[Optional[Any]]]:
        """Between stream steps: ``window`` drops whole frames; ``streaming`` packs KV (true VRAM shrink)."""
        if carry is None:
            return None
        mm = str(self.memory_mode).lower()
        out: List[Optional[Any]] = []
        for kv in carry:
            if kv is None:
                out.append(None)
                continue
            if isinstance(kv, StreamingKVState):
                if mm != "streaming":
                    raise ValueError(
                        "StreamingKVState requires memory_mode=streaming"
                    )
                # append 已在 stream_chunk_forward 内完成；此处仅 detach。
                out.append(kv.clone_buffers())
                continue
            pk, pv = kv[0], kv[1]
            pk = pk.contiguous()
            pv = pv.contiguous()
            T_total = int(pk.shape[2])
            if mm == "streaming":
                out.append(
                    streaming_carry_from_rect_kv(
                        pk,
                        pv,
                        num_reference_frames=int(self.num_reference_frames),
                        local_window_frames=int(self.local_window_frames),
                        num_summary_tokens=int(self.streaming_num_summary_tokens),
                        max_summary_frames=int(self.paged_max_summary_frames),
                    )
                )
            elif mm == "window":
                pk2, pv2, _new_t = prune_window_carry_pk_pv(
                    pk,
                    pv,
                    T_total=T_total,
                    num_reference_frames=int(self.num_reference_frames),
                    local_window_frames=int(self.local_window_frames),
                )
                out.append((pk2, pv2))
            else:
                raise ValueError(f"Unknown memory_mode={self.memory_mode!r}")
        return out

    def set_local_window_frames(self, w: int) -> None:
        """Set sliding temporal window (streaming + streaming bias); reference unchanged."""
        w = max(int(w), 1)
        self.local_window_frames = w
        for i in range(1, len(self.decoder), 2):
            attn = self.decoder[i].attn
            if isinstance(attn, StreamingFlashAttention):
                attn.local_window_frames = w

    def _build_pos_grid(
        self, B: int, T_total: int, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        pos = self.position_getter(B * T_total, patch_h, patch_w, device)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(
                B * T_total, self.patch_start_idx, 2, device=device, dtype=pos.dtype
            )
            pos = torch.cat([pos_special, pos], dim=1)
        hw = pos.shape[1]
        return pos.reshape(B, T_total * hw, 2)

    def _stream_packed_key_positions(
        self,
        hs_carry: StreamingKVState,
        *,
        B: int,
        N: int,
        H: int,
        W: int,
        tpf: int,
        device: torch.device,
        pos_cache: Dict[str, Any],
    ) -> torch.Tensor:
        """2D RoPE key positions for packed past + new frame (``pi3_2d`` stream only)."""
        Fp = int(hs_carry.total_frames_seen)
        sig = ("hs_pos", B, tuple(hs_carry.bias_fingerprint_tokens()), N, H, W, device)
        if pos_cache.get("pos_k_sig") != sig or "pos_k" not in pos_cache:
            pos_cache["pos_k"] = self._build_pos_grid(B, Fp + N, H, W, device)
            pos_cache["pos_k_sig"] = sig
        if str(getattr(self, "global_pos_encoding", "pi3_2d")) != "pi3_2d":
            return pos_cache["pos_k"]
        return streamed_chunk_key_positions_pi3(
            pos_cache["pos_k"],
            hs_carry,
            tpf_new=tpf,
            current_global_frame_idx=Fp,
        )

    def _global_attn_stream_packed(
        self,
        h: torch.Tensor,
        pos_q: torch.Tensor,
        hs_carry: StreamingKVState,
        blk,
        attn_mod: StreamingFlashAttention,
        B: int,
        N: int,
        H: int,
        W: int,
        pos_cache: Dict[str, Any],
    ) -> Tuple[torch.Tensor, StreamingKVState]:
        """Stream infer: packed streaming carry, no attn bias (see ``infer/stream/attn_sdpa``)."""
        tpf = h.shape[1] // N
        chunk_k_pos = self._stream_packed_key_positions(
            hs_carry, B=B, N=N, H=H, W=W, tpf=tpf, device=h.device, pos_cache=pos_cache
        )
        xn = blk.norm1(h)
        out, new_carry = stream_window_attention(
            attn_mod,
            xn,
            pos_q=pos_q,
            hs_carry=hs_carry,
            num_new_frames=N,
            chunk_key_positions=chunk_k_pos,
            image_hw=(H, W),
            patch_start_idx=int(self.patch_start_idx),
        )
        h_out = h + blk.ls1(out)
        h_out = h_out + blk.ls2(blk.mlp(blk.norm2(h_out)))
        return h_out, new_carry

    def decode(
        self,
        hidden,
        N,
        H,
        W,
        causal_global: bool = True,
        stream_use_cache: bool = False,
        past_key_values: Optional[List[Optional[Tuple]]] = None,
        long_sequence_parallel: bool = False,
        streaming_inference: bool = False,
    ):
        if not long_sequence_parallel:
            return super().decode(hidden, N, H, W, causal_global, stream_use_cache, past_key_values)
        if stream_use_cache:
            raise ValueError("long_sequence_parallel cannot be used with stream_use_cache=True")

        if streaming_inference:
            if past_key_values is None:
                past_key_values = [None] * self.num_global_decoder_blocks
        else:
            past_key_values = None

        BN, hw_enc, _ = hidden.shape
        B = BN // N

        final_output: List[torch.Tensor] = []

        hidden = hidden.reshape(B * N, hw_enc, -1)

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
            pos_special = (
                torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        global_attn_bias_cache: Dict[str, Any] = {}

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
            if i % 2 == 1 and causal_global and not stream_use_cache and not long_sequence_parallel:
                attn_mask = causal_frame_attention_mask(N, hw, hidden.dtype, hidden.device)

            use_blk_cache = stream_use_cache and (i % 2 == 1)
            past_kv_blk = (
                past_key_values[global_blk_idx]
                if use_blk_cache and past_key_values is not None
                else None
            )

            if use_blk_cache:
                hidden, new_kv = blk(
                    hidden,
                    xpos=pos,
                    attn_mask=None,
                    past_key_values=past_kv_blk,
                    use_cache=True,
                )
                assert past_key_values is not None
                past_key_values[global_blk_idx] = new_kv
            elif long_sequence_parallel and (i % 2 == 1):
                attn_mod = blk.attn
                if not isinstance(attn_mod, StreamingFlashAttention):
                    raise TypeError("Expected StreamingFlashAttention on global blocks")
                nh = attn_mod.num_heads
                dim_in = attn_mod.qkv.in_features
                hd = dim_in // nh
                tpf = hidden.shape[1] // N

                if streaming_inference:
                    past_kv_ent = (
                        past_key_values[global_blk_idx] if past_key_values is not None else None
                    )
                else:
                    past_kv_ent = None

                hs_carry: Optional[StreamingKVState] = (
                    past_kv_ent if isinstance(past_kv_ent, StreamingKVState) else None
                )

                if streaming_inference and hs_carry is not None:
                    hidden, new_kv = self._global_attn_stream_packed(
                        hidden,
                        pos,
                        hs_carry,
                        blk,
                        attn_mod,
                        B,
                        N,
                        H,
                        W,
                        global_attn_bias_cache,
                    )
                else:
                    if past_kv_ent is None:
                        pk = hidden.new_zeros((B, nh, 0, tpf, hd), requires_grad=False)
                        pv = hidden.new_zeros((B, nh, 0, tpf, hd), requires_grad=False)
                    elif isinstance(past_kv_ent, tuple) and len(past_kv_ent) == 2:
                        pk, pv = past_kv_ent[0], past_kv_ent[1]
                    else:
                        raise ValueError(
                            "KV carry expects (pk, pv) tuples or StreamingKVState, "
                            f"got {type(past_kv_ent)}"
                        )

                    def _global_attn_cp(
                        h,
                        p_q,
                        pk_,
                        pv_,
                        *,
                        _blk=blk,
                        _attn=attn_mod,
                        _B=B,
                        _N=N,
                        _H=H,
                        _W=W,
                        _bias_cache=global_attn_bias_cache,
                    ):
                        t_pst_frames = int(pk_.shape[2])
                        tpf_win = h.shape[1] // _N
                        W_blk = int(getattr(_attn, "local_window_frames", self.local_window_frames))
                        bias_dtype = h.dtype
                        bias_sig = (
                            t_pst_frames,
                            _N,
                            tpf_win,
                            W_blk,
                            h.device,
                            bias_dtype,
                            kv_carry_fingerprint_for_bias_cache(None),
                            str(_attn.memory_mode),
                            int(_attn.num_reference_frames),
                            int(_attn.local_window_frames),
                            int(_attn.num_summary_tokens),
                            str(_attn.global_pos_encoding),
                        )
                        if _bias_cache.get("bias_sig") != bias_sig or "bias" not in _bias_cache:
                            _bias_cache["bias"] = _attn.compute_streaming_attn_bias(
                                t_past=t_pst_frames,
                                num_new_frames=_N,
                                tokens_per_frame=tpf_win,
                                dtype=bias_dtype,
                                device=h.device,
                                temporal_window_frames=W_blk,
                                kv_carry_meta=None,
                            )
                            _bias_cache["bias_sig"] = bias_sig
                            bias_cached = _bias_cache["bias"]
                        else:
                            bias_cached = _bias_cache["bias"]

                        if t_pst_frames == 0:
                            chunk_k_pos = p_q
                        else:
                            pos_k_sig = (_B, t_pst_frames, _N, _H, _W, h.device)
                            if (
                                _bias_cache.get("pos_k_sig") != pos_k_sig
                                or "pos_k" not in _bias_cache
                            ):
                                _bias_cache["pos_k"] = self._build_pos_grid(
                                    _B, t_pst_frames + _N, _H, _W, h.device
                                )
                                _bias_cache["pos_k_sig"] = pos_k_sig
                            chunk_k_pos = _bias_cache["pos_k"]

                        past_eff = (pk_, pv_) if t_pst_frames > 0 else None
                        xn = _blk.norm1(h)
                        out, new_kv_ = _attn.stream_chunk_forward(
                            xn,
                            pos_q=p_q,
                            past_key_values=past_eff,
                            num_new_frames=_N,
                            chunk_key_positions=chunk_k_pos,
                            kv_carry_meta=None,
                            cached_streaming_attn_bias=bias_cached,
                            image_hw=(_H, _W),
                            patch_start_idx=int(self.patch_start_idx),
                        )
                        h_out = h + _blk.ls1(out)
                        h_out = h_out + _blk.ls2(_blk.mlp(_blk.norm2(h_out)))
                        return h_out, new_kv_

                    if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                        hidden, new_kv = checkpoint(
                            _global_attn_cp,
                            hidden,
                            pos,
                            pk,
                            pv,
                            use_reentrant=False,
                        )
                    else:
                        hidden, new_kv = _global_attn_cp(hidden, pos, pk, pv)
                if streaming_inference:
                    assert past_key_values is not None
                    past_key_values[global_blk_idx] = new_kv
            elif i >= self.num_dec_blk_not_to_checkpoint and self.training and not stream_use_cache:
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
        if long_sequence_parallel and not streaming_inference:
            return fused, pos_out, None
        return fused, pos_out, past_key_values

    def forward(
        self,
        imgs: torch.Tensor,
        causal_global_attn: Optional[bool] = None,
        stream_use_cache: bool = False,
        past_key_values: Optional[List[Optional[Tuple]]] = None,
        ref_hidden: Optional[torch.Tensor] = None,
        camera_state: Optional[Dict[str, torch.Tensor]] = None,
        long_sequence_parallel: bool = False,
        streaming_inference: bool = False,
    ) -> Dict[str, Any]:
        if causal_global_attn is None:
            causal_global_attn = self.causal_global_attn

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        if long_sequence_parallel:
            if stream_use_cache:
                raise ValueError("long_sequence_parallel excludes stream_use_cache")
        elif stream_use_cache and N != 1:
            raise ValueError("stream_use_cache requires N==1 (single-frame steps with KV cache)")

        patch_h, patch_w = H // 14, W // 14
        tks = patch_h * patch_w + self.patch_start_idx

        imgs_bn = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs_bn, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos, pkv = self.decode(
            hidden,
            N,
            H,
            W,
            causal_global=causal_global_attn,
            stream_use_cache=stream_use_cache,
            past_key_values=past_key_values,
            long_sequence_parallel=long_sequence_parallel,
            streaming_inference=streaming_inference,
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
                    hidden.reshape(B, N, tks, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B * N, tks, -1)
                )
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx :]], (H, W)).reshape(
                B, N, H, W, -1
            )
            xy, z = ret.split([2, 1], dim=-1)
            z = depth_from_log_z(z, self.point_z_log_max)
            local_points = torch.cat([xy * z, z], dim=-1)

            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx :]], (H, W)).reshape(
                    B, N, H, W, -1
                )
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
            out["past_key_values"] = pkv
        elif streaming_inference and pkv is not None:
            out["past_key_values"] = pkv
        if new_ref_hidden is not None:
            out["ref_hidden"] = new_ref_hidden
        if new_camera_state is not None:
            out["camera_state"] = new_camera_state
        return out

    @staticmethod
    def _camera_state_output_slice(
        pred: Dict[str, Any], num_frames: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        state = pred.get("camera_state")
        if not isinstance(state, dict):
            return None
        raw_rel = state.get("raw_adjacent_rel_poses")
        source_idx = state.get("source_frame_indices")
        if raw_rel is None or source_idx is None:
            return None
        if raw_rel.shape[1] < num_frames or source_idx.shape[1] < num_frames:
            return None
        out = {
            "raw_adjacent_rel_poses": raw_rel[:, -num_frames:],
            "source_frame_indices": source_idx[:, -num_frames:],
        }
        for bias_key in ("rotation_residual"):
            rot_bias = state.get(bias_key)
            if rot_bias is not None and rot_bias.numel() > 0:
                if rot_bias.shape[1] == raw_rel.shape[1] - 1:
                    pad = rot_bias.new_zeros(rot_bias.shape[0], 1, rot_bias.shape[-1])
                    rot_bias = torch.cat([pad, rot_bias], dim=1)
                if rot_bias.shape[1] >= num_frames:
                    out[bias_key] = rot_bias[:, -num_frames:]
        return out

    @staticmethod
    def _attach_camera_state_output(
        stacked: Dict[str, Any], chunks: List[Optional[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Any]:
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            return stacked
        metrics = {
            "raw_adjacent_rel_poses": torch.cat([c["raw_adjacent_rel_poses"] for c in chunks], dim=1),
            "source_frame_indices": torch.cat([c["source_frame_indices"] for c in chunks], dim=1),
        }
        for bias_key in ("rotation_residual"):
            if any(c.get(bias_key) is not None for c in chunks):
                bias_chunks = []
                for c in chunks:
                    bias = c.get(bias_key)
                    if bias is None:
                        raw = c["raw_adjacent_rel_poses"]
                        bias = raw.new_zeros(raw.shape[0], raw.shape[1], 3)
                    bias_chunks.append(bias)
                metrics[bias_key] = torch.cat(bias_chunks, dim=1)
        stacked["camera_state_metrics"] = metrics
        return stacked

    @torch.inference_mode()
    def _inference_stream_full(
        self,
        imgs: torch.Tensor,
        causal_global_attn: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Single ``forward``: matches ``training runtime.forward_batch`` (no KV carry).

        Streaming / window SDPA biases are built once per layer
        inside ``decode`` and cached for the duration of that ``forward`` (signature ``t_past, N, …``).
        """
        if causal_global_attn is None:
            causal_global_attn = self.causal_global_attn
        _, N, _, _, _ = imgs.shape
        if N < 1:
            raise ValueError("_inference_stream_full requires sequence length N >= 1")
        pred = self.forward(
            imgs,
            causal_global_attn=causal_global_attn,
            long_sequence_parallel=True,
            streaming_inference=False,
        )
        out = {
            k: v
            for k, v in pred.items()
            if k not in ("past_key_values", "ref_hidden", "camera_state")
        }
        return self._attach_camera_state_output(out, [self._camera_state_output_slice(pred, N)])

    @torch.inference_mode()
    def _inference_stream_sdpa(
        self,
        imgs: torch.Tensor,
        causal_global_attn: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Causal streaming with streaming **packed** KV (true VRAM savings) or dense window prune.

        Uses the same causal SDPA mask rules as the paged backend; ``memory_mode=streaming`` summaries mid-memory
        frames to the summary register band. ``global_pos_encoding=rope3d`` is supported (gathered
        3D RoPE on packed keys).
        """
        if causal_global_attn is None:
            causal_global_attn = self.causal_global_attn
        _, N, _, H_img, W_img = imgs.shape
        if N < 1:
            raise ValueError("_inference_stream_sdpa requires sequence length N >= 1")

        carry: Optional[List[Optional[Tuple]]] = None
        ref_h: Optional[torch.Tensor] = None
        camera_state: Optional[Dict[str, torch.Tensor]] = None
        preds: List[Dict[str, Any]] = []
        camera_state_chunks: List[Optional[Dict[str, torch.Tensor]]] = []
        _no_pbar = os.environ.get("ABOT_RECON_NO_TQDM", "0") == "1"
        for t in tqdm(
            range(N),
            desc="ABotReconNetwork stream",
            unit="frm",
            leave=True,
            disable=_no_pbar,
        ):
            frame = imgs[:, t : t + 1]
            pred = self.forward(
                frame,
                causal_global_attn=causal_global_attn,
                long_sequence_parallel=True,
                streaming_inference=True,
                past_key_values=carry,
                ref_hidden=ref_h,
                camera_state=camera_state,
            )
            carry = self._postprocess_stream_carry(
                detach_carry(pred.get("past_key_values")),
                spatial_hw=(int(H_img), int(W_img)),
            )
            if pred.get("ref_hidden") is not None:
                ref_h = pred["ref_hidden"]
            camera_state = pred.get("camera_state", camera_state)
            camera_state_chunks.append(self._camera_state_output_slice(pred, frame.shape[1]))
            preds.append(
                {
                    k: v
                    for k, v in pred.items()
                    if k not in ("past_key_values", "ref_hidden", "camera_state")
                }
            )
        stacked: Dict[str, Any] = {}
        for k in preds[0].keys():
            vals = [p[k] for p in preds]
            if vals[0] is None:
                stacked[k] = None
            else:
                stacked[k] = torch.cat(vals, dim=1)
        return self._attach_camera_state_output(stacked, camera_state_chunks)

    # =========================================================================
    # FlashInfer paged KV streaming (B=1 only)
    # =========================================================================

    def _can_use_paged_kv(self, imgs: torch.Tensor) -> bool:
        """Check whether the paged-KV streaming path applies to this call."""
        if not getattr(self, "use_paged_kv", False):
            return False
        if imgs.shape[0] != 1:
            return False
        if self.memory_mode not in ("streaming", "window"):
            return False
        try:
            from abot_recon.modeling.streaming.paged_kv import (
                flashinfer_available,
            )
        except ImportError:
            return False
        if not self.paged_force_fp32 and not flashinfer_available():
            return False
        return True

    def _ensure_paged_manager(
        self,
        *,
        H: int,
        W: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Create or recreate the per-model paged KV cache manager.

        Rebuilds if the (resolution, dtype, memory_mode, window, reference, summary)
        signature changed; otherwise just calls ``reset()``.
        """
        from abot_recon.modeling.streaming.paged_kv import (
            PagedKVCacheManager,
        )

        paged_dtype = dtype
        if not self.paged_force_fp32 and paged_dtype not in (torch.float16, torch.bfloat16):
            # FlashInfer FA2/FA3 kernels do not accept fp32 q/kv dtypes. Validation
            # can call stream inference with fp32 images outside autocast, so keep
            # the model path usable by storing/running paged attention in bf16.
            paged_dtype = torch.bfloat16

        ps = int(self.patch_size)
        tpf = (H // ps) * (W // ps) + int(self.patch_start_idx)
        # Probe first global attn for head config.
        attn0 = self.decoder[1].attn
        num_heads = attn0.num_heads
        head_dim = attn0.qkv.in_features // num_heads
        sig = (
            tpf,
            num_heads,
            head_dim,
            int(self.num_reference_frames),
            int(self.local_window_frames),
            int(self.streaming_num_summary_tokens),
            int(self.paged_max_summary_frames),
            self.memory_mode,
            str(paged_dtype),
            str(device),
        )
        if self._paged_manager is not None and self._paged_manager_sig == sig:
            self._paged_manager.reset()
            return

        self._paged_manager = PagedKVCacheManager(
            num_layers=self.num_global_decoder_blocks,
            tpf=tpf,
            num_heads=num_heads,
            head_dim=head_dim,
            num_reference_frames=int(self.num_reference_frames),
            local_window_frames=int(self.local_window_frames),
            num_summary_tokens=int(self.streaming_num_summary_tokens),
            memory_mode=self.memory_mode,
            max_total_frames=int(self.paged_max_total_frames),
            max_summary_frames=int(self.paged_max_summary_frames),
            dtype=paged_dtype,
            device=device,
            force_fp32=self.paged_force_fp32,
        )
        self._paged_manager_sig = sig

    def _decode_paged(
        self,
        hidden: torch.Tensor,
        *,
        H: int,
        W: int,
        frame_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-frame decode (N=1, B=1) using paged KV for global layers.

        Mirrors the ``long_sequence_parallel=True, streaming_inference=True``
        slice of ``decode()`` for B=N=1, replacing the global ``stream_chunk_forward``
        with ``StreamingFlashAttention.paged_stream_forward``.

        Returns:
            (fused_hidden, pos): ``fused`` is the cat of the last two decoder
            layers' outputs along the channel dim (matches ``decode()``).
        """
        manager = self._paged_manager
        if manager is None:
            raise RuntimeError("paged manager not initialised; call _ensure_paged_manager first")

        B = 1
        BN, hw_enc, _ = hidden.shape
        if BN != B:
            raise ValueError(f"_decode_paged requires B=N=1; got hidden batch={BN}")

        # Prepend register tokens → hidden shape (1, tpf, C)
        register_token = self.register_token.repeat(B, 1, 1, 1).reshape(
            B, *self.register_token.shape[-2:]
        )
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        # Position grid (same layout as decode()): (1, tpf, 2) for pi3_2d
        if self.pos_type.startswith("rope"):
            pos = self.position_getter(B, H // self.patch_size, W // self.patch_size, hidden.device)
        else:
            pos = None
        if pos is not None and self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(
                B, self.patch_start_idx, 2, device=hidden.device, dtype=pos.dtype
            )
            pos = torch.cat([pos_special, pos], dim=1)
        # pos shape: (1, tpf, 2)

        final_output: List[torch.Tensor] = []
        global_blk_idx = 0
        n_layers = len(self.decoder)

        for i in range(n_layers):
            blk = self.decoder[i]
            if i % 2 == 0:
                # Local attention: standard block call, hidden in (B*N, hw, C) = (1, tpf, C)
                hidden = blk(
                    hidden,
                    xpos=pos,
                    attn_mask=None,
                    past_key_values=None,
                    use_cache=False,
                )
            else:
                # Global attention via paged KV manager.
                if not isinstance(blk.attn, StreamingFlashAttention):
                    raise TypeError(
                        f"Expected StreamingFlashAttention at decoder[{i}], "
                        f"got {type(blk.attn).__name__}"
                    )
                x_norm = blk.norm1(hidden)
                attn_out = blk.attn.paged_stream_forward(
                    x_norm,
                    pos_q=pos,
                    manager=manager,
                    layer_idx=global_blk_idx,
                    frame_idx=int(frame_idx),
                    image_hw=(H, W),
                    patch_start_idx=int(self.patch_start_idx),
                )
                hidden = hidden + blk.ls1(attn_out)
                hidden = hidden + blk.ls2(blk.mlp(blk.norm2(hidden)))
                global_blk_idx += 1

            if i + 1 in (n_layers - 1, n_layers):
                final_output.append(hidden.reshape(B, hw, -1))

        fused = torch.cat([final_output[0], final_output[1]], dim=-1)
        return fused, pos

    def _forward_frame_paged(
        self,
        frame: torch.Tensor,
        *,
        frame_idx: int,
        ref_hidden: Optional[torch.Tensor],
        camera_state: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, Any]:
        """One-frame forward using paged KV.  Mirrors ``forward()`` for B=N=1."""
        imgs = (frame - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        if B != 1 or N != 1:
            raise ValueError(f"_forward_frame_paged requires B=N=1, got B={B} N={N}")

        patch_h, patch_w = H // 14, W // 14
        tks = patch_h * patch_w + self.patch_start_idx

        imgs_bn = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs_bn, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self._decode_paged(hidden, H=H, W=W, frame_idx=frame_idx)

        new_ref_hidden: Optional[torch.Tensor] = None
        global_point_hidden = None
        if self.use_global_points:
            hf = hidden.reshape(B, tks, -1)
            if ref_hidden is None:
                context = hf
                new_ref_hidden = hf.detach()
            else:
                context = ref_hidden
                new_ref_hidden = ref_hidden
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos) if self.train_conf else None
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx :]], (H, W)).reshape(
                B, N, H, W, -1
            )
            xy, z = ret.split([2, 1], dim=-1)
            z = depth_from_log_z(z, self.point_z_log_max)
            local_points = torch.cat([xy * z, z], dim=-1)

            conf = None
            if self.train_conf and conf_hidden is not None:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx :]], (H, W)).reshape(
                    B, N, H, W, -1
                )

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

            global_points = None
            if self.use_global_points and global_point_hidden is not None:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head(
                    [global_point_hidden[:, self.patch_start_idx :]], (H, W)
                ).reshape(B, N, H, W, -1)

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
        if new_ref_hidden is not None:
            out["ref_hidden"] = new_ref_hidden
        if new_camera_state is not None:
            out["camera_state"] = new_camera_state
        return out

    def _forward_frame_camera_only(
        self,
        frame: torch.Tensor,
        *,
        frame_idx: int,
        ref_hidden: Optional[torch.Tensor],
        camera_state: Optional[Dict[str, torch.Tensor]],
        causal_global_attn: Optional[bool] = None,
        past_key_values: Optional[List[Optional[Tuple]]] = None,
        use_paged: bool = True,
    ) -> Dict[str, Any]:
        """One-frame streaming forward for camera poses only.

        This uses the same encoder/decoder/camera-head path as the normal
        frame forward, but skips point/conf/global point heads for long pose eval.
        """
        del ref_hidden  # Global point reference is only needed for point outputs.
        if causal_global_attn is None:
            causal_global_attn = self.causal_global_attn

        imgs = (frame - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        if B != 1 or N != 1:
            raise ValueError(f"_forward_frame_camera_only requires B=N=1, got B={B} N={N}")

        patch_h, patch_w = H // 14, W // 14
        imgs_bn = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs_bn, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        if use_paged:
            hidden, pos = self._decode_paged(hidden, H=H, W=W, frame_idx=frame_idx)
            pkv = None
        else:
            hidden, pos, pkv = self.decode(
                hidden,
                N,
                H,
                W,
                causal_global=causal_global_attn,
                stream_use_cache=False,
                past_key_values=past_key_values,
                long_sequence_parallel=True,
                streaming_inference=True,
            )

        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        with torch.amp.autocast(device_type="cuda", enabled=False):
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
            if new_camera_state is not None:
                new_camera_state.pop("_aux_rel_pose_pred", None)
                new_camera_state.pop("_history_rel_pose_pred", None)

        out: Dict[str, Any] = {"camera_poses": camera_poses}
        if pkv is not None:
            out["past_key_values"] = pkv
        if new_camera_state is not None:
            out["camera_state"] = new_camera_state
        return out

    @torch.inference_mode()
    def _inference_stream_paged(
        self,
        imgs: torch.Tensor,
        causal_global_attn: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """FlashInfer paged-KV per-frame streaming. B=1 only."""
        del causal_global_attn  # paged path is intrinsically causal (cache only sees past)
        B, N, _, H_img, W_img = imgs.shape
        if B != 1:
            raise ValueError(f"_inference_stream_paged requires B=1, got {B}")
        if N < 1:
            raise ValueError("_inference_stream_paged requires N >= 1")

        self._ensure_paged_manager(H=H_img, W=W_img, dtype=imgs.dtype, device=imgs.device)

        ref_h: Optional[torch.Tensor] = None
        camera_state: Optional[Dict[str, torch.Tensor]] = None
        preds: List[Dict[str, Any]] = []
        camera_state_chunks: List[Optional[Dict[str, torch.Tensor]]] = []
        _no_pbar = os.environ.get("ABOT_RECON_NO_TQDM", "0") == "1"
        for t in tqdm(
            range(N),
            desc="ABotReconNetwork paged stream",
            unit="frm",
            leave=True,
            disable=_no_pbar,
        ):
            frame = imgs[:, t : t + 1]
            pred = self._forward_frame_paged(
                frame, frame_idx=t, ref_hidden=ref_h, camera_state=camera_state
            )
            if pred.get("ref_hidden") is not None:
                ref_h = pred["ref_hidden"]
            camera_state = pred.get("camera_state", camera_state)
            camera_state_chunks.append(self._camera_state_output_slice(pred, frame.shape[1]))
            preds.append({k: v for k, v in pred.items() if k not in ("ref_hidden", "camera_state")})

        stacked: Dict[str, Any] = {}
        for k in preds[0].keys():
            vals = [p[k] for p in preds]
            if vals[0] is None:
                stacked[k] = None
            else:
                stacked[k] = torch.cat(vals, dim=1)
        return self._attach_camera_state_output(stacked, camera_state_chunks)

    @torch.inference_mode()
    def inference_stream_iter(
        self,
        frames: Iterable[torch.Tensor],
        *,
        num_frames: Optional[int] = None,
        causal_global_attn: Optional[bool] = None,
        output_keys: Optional[Iterable[str]] = None,
        dense_output_indices: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        """Streaming inference over already-loaded single-frame tensors.

        Each item must be ``[B,1,3,H,W]``. This avoids staging the full image
        sequence on CPU/GPU before inference. Only ``infer_mode=stream`` is
        supported because ``full`` mode requires the complete tensor.
        """
        mode = str(getattr(self, "infer_mode", "full")).lower()
        if mode != "stream":
            raise ValueError("inference_stream_iter requires infer_mode='stream'")

        iterator = iter(frames)
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("inference_stream_iter requires at least one frame") from exc

        if first.dim() != 5 or first.shape[1] != 1:
            raise ValueError(f"expected frame shape [B,1,3,H,W], got {tuple(first.shape)}")

        allowed = set(output_keys) if output_keys is not None else None
        dense_indices = (
            None if dense_output_indices is None else {int(index) for index in dense_output_indices}
        )
        dense_keys = {"points", "local_points", "conf", "global_points"}
        collect_camera_state = allowed is None or "camera_state" in allowed
        camera_only = allowed is not None and allowed.issubset({"camera_poses", "camera_state"})

        def keep_pred(
            pred: Dict[str, Any], *, skip_state: Tuple[str, ...], frame_index: int
        ) -> Dict[str, Any]:
            return {
                k: v
                for k, v in pred.items()
                if k not in skip_state
                and (allowed is None or k in allowed)
                and (dense_indices is None or k not in dense_keys or frame_index in dense_indices)
            }

        def stack_preds(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not preds:
                raise ValueError("no predictions were collected; check output_keys")
            stacked: Dict[str, Any] = {}
            for k in set().union(*(pred.keys() for pred in preds)):
                vals = [pred[k] for pred in preds if k in pred]
                if vals[0] is None:
                    stacked[k] = None
                else:
                    stacked[k] = torch.cat(vals, dim=1)
            return stacked

        frames_iter = chain((first,), iterator)
        total = num_frames if num_frames is not None else None
        _no_pbar = os.environ.get("ABOT_RECON_NO_TQDM", "0") == "1"

        if self._can_use_paged_kv(first):
            B, _, _, H_img, W_img = first.shape
            if B != 1:
                raise ValueError(f"paged inference_stream_iter requires B=1, got {B}")
            self._ensure_paged_manager(H=H_img, W=W_img, dtype=first.dtype, device=first.device)
            ref_h: Optional[torch.Tensor] = None
            camera_state: Optional[Dict[str, torch.Tensor]] = None
            preds: List[Dict[str, Any]] = []
            camera_state_chunks: List[Optional[Dict[str, torch.Tensor]]] = []
            for t, frame in enumerate(
                tqdm(
                    frames_iter,
                    total=total,
                    desc="ABotReconNetwork paged stream",
                    unit="frm",
                    leave=True,
                    disable=_no_pbar,
                )
            ):
                frame_camera_only = camera_only or (
                    dense_indices is not None and t not in dense_indices
                )
                if frame_camera_only:
                    pred = self._forward_frame_camera_only(
                        frame,
                        frame_idx=t,
                        ref_hidden=ref_h,
                        camera_state=camera_state,
                        use_paged=True,
                    )
                else:
                    pred = self._forward_frame_paged(
                        frame, frame_idx=t, ref_hidden=ref_h, camera_state=camera_state
                    )
                if pred.get("ref_hidden") is not None:
                    ref_h = pred["ref_hidden"]
                camera_state = pred.get("camera_state", camera_state)
                if collect_camera_state:
                    camera_state_chunks.append(
                        self._camera_state_output_slice(pred, frame.shape[1])
                    )
                preds.append(
                    keep_pred(
                        pred,
                        skip_state=("ref_hidden", "camera_state"),
                        frame_index=t,
                    )
                )
            out = stack_preds(preds)
            return (
                self._attach_camera_state_output(out, camera_state_chunks)
                if collect_camera_state
                else out
            )

        if causal_global_attn is None:
            causal_global_attn = self.causal_global_attn
        _, _, _, H_img, W_img = first.shape
        carry: Optional[List[Optional[Tuple]]] = None
        ref_h: Optional[torch.Tensor] = None
        camera_state: Optional[Dict[str, torch.Tensor]] = None
        preds: List[Dict[str, Any]] = []
        camera_state_chunks: List[Optional[Dict[str, torch.Tensor]]] = []
        for frame_index, frame in enumerate(
            tqdm(
                frames_iter,
                total=total,
                desc="ABotReconNetwork stream",
                unit="frm",
                leave=True,
                disable=_no_pbar,
            )
        ):
            frame_camera_only = camera_only or (
                dense_indices is not None and frame_index not in dense_indices
            )
            if frame_camera_only:
                pred = self._forward_frame_camera_only(
                    frame,
                    frame_idx=frame_index,
                    ref_hidden=ref_h,
                    camera_state=camera_state,
                    causal_global_attn=causal_global_attn,
                    past_key_values=carry,
                    use_paged=False,
                )
            else:
                pred = self.forward(
                    frame,
                    causal_global_attn=causal_global_attn,
                    long_sequence_parallel=True,
                    streaming_inference=True,
                    past_key_values=carry,
                    ref_hidden=ref_h,
                    camera_state=camera_state,
                )
            carry = self._postprocess_stream_carry(
                detach_carry(pred.get("past_key_values")),
                spatial_hw=(int(H_img), int(W_img)),
            )
            if pred.get("ref_hidden") is not None:
                ref_h = pred["ref_hidden"]
            camera_state = pred.get("camera_state", camera_state)
            if collect_camera_state:
                camera_state_chunks.append(self._camera_state_output_slice(pred, frame.shape[1]))
            preds.append(
                keep_pred(
                    pred,
                    skip_state=("past_key_values", "ref_hidden", "camera_state"),
                    frame_index=frame_index,
                )
            )
        out = stack_preds(preds)
        return (
            self._attach_camera_state_output(out, camera_state_chunks)
            if collect_camera_state
            else out
        )

    def _load_deferred_streaming_checkpoint(self, ckpt: str) -> None:
        """Load pretrained weights after all ABot-Recon-specific modules exist.

        The base Pi3 constructor cannot load ABot-Recon checkpoints safely because
        gate modules are registered in this subclass after ``super().__init__``.
        Keeping this loader here makes ``model.ckpt`` safe for training and eval.
        """
        ckpt_path = str(ckpt)
        src = (
            "OSS (buffered in RAM via ossutil.open)"
            if ckpt_path.startswith("oss://")
            else "local/NAS path"
        )
        print(
            f"[ABotReconNetwork] Reading deferred checkpoint ({src}; can be slow)…\\n"
            f"      path={ckpt_path}",
            flush=True,
        )
        checkpoint = _load_raw_pi3_checkpoint(ckpt_path)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]

        checkpoint_has_conf_decoder = any(
            key.startswith("conf_decoder.") or ".conf_decoder." in key for key in checkpoint.keys()
        )
        print("[ABotReconNetwork] Applying deferred state_dict to complete streaming model…", flush=True)
        res = self.load_state_dict(checkpoint, strict=False)
        if (
            self.enable_confidence
            and self.init_conf_decoder_from_point
            and not checkpoint_has_conf_decoder
        ):
            self._initialize_conf_decoder_from_point()
        missing = list(getattr(res, "missing_keys", []))
        unexpected = list(getattr(res, "unexpected_keys", []))
        gate_bad = [k for k in missing + unexpected if ".gate_proj." in k]
        if gate_bad:
            preview = ", ".join(gate_bad[:16])
            raise RuntimeError(
                "Streaming checkpoint gate weights were not loaded cleanly after deferred load: "
                f"{preview}. Check model.gate_layers and checkpoint compatibility."
            )
        print(f"[ABotReconNetwork] Load checkpoints from {ckpt}: {res}", flush=True)
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def inference_stream(
        self,
        imgs: torch.Tensor,
        causal_global_attn: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run streaming inference per ``infer_mode``: ``full`` (one-shot) or ``stream`` (per-frame KV)."""
        mode = str(getattr(self, "infer_mode", "full")).lower()
        if mode == "full":
            return self._inference_stream_full(imgs, causal_global_attn=causal_global_attn)
        if mode == "stream":
            if self._can_use_paged_kv(imgs):
                return self._inference_stream_paged(imgs, causal_global_attn=causal_global_attn)
            return self._inference_stream_sdpa(imgs, causal_global_attn=causal_global_attn)
        raise ValueError(f"infer_mode must be 'full' or 'stream', got {mode!r}")

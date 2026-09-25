"""Dense sliding-window KV carry for chunk-parallel training (no sparse keyframes).

1. Forward one chunk with optional **past KV** (last ``keep_last`` frames from the previous chunk,
   typically **detached** in the trainer).
2. Concatenate along time: ``[past_kv][new_kv]`` for attention.
3. Slice to the last ``keep_last`` frames and detach → next chunk's past.

Attention mask (``layers.chunk_mask``) uses **concatenated frame slots** ``[past][new]``:
causal + last ``W`` slots (``model.chunk_temporal_window``). ``dense_kv_carry_meta`` is optional
metadata for debugging or other code paths, not required for the bias.

**Loss (trainer):** chunk-internal pairs use ``Pi3Loss`` with ``camera_max_pair_distance`` = lookback
``dist``. Cross-chunk pairs where the reference index lies in detached past use
``camera_chunk_bridge_loss`` (GT pose on the past side); see ``trainers/stream_chunk_trainer.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

KVLayerItem = Optional[Any]


def dense_kv_carry_meta(
    device: torch.device,
    chunk_start: int,
    num_past_frames: int,
) -> Dict[str, Any]:
    """Global frame ids for ``streaming_chunk_attention_bias`` (contiguous past window).

    Past slots (oldest→newest): ``chunk_start - num_past_frames .. chunk_start - 1``.
    New frames in this chunk start at global index ``chunk_start`` (``new_base_time``).
    """
    if num_past_frames <= 0:
        raise ValueError("dense_kv_carry_meta requires num_past_frames > 0")
    cs = int(chunk_start)
    tp = int(num_past_frames)
    logical = torch.arange(cs - tp, cs, device=device, dtype=torch.long)
    return {
        "logical_frame_ids": logical,
        "new_base_time": cs,
    }


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def slice_kv_carry(
    past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    keep_last_frames: int,
) -> Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
    """Keep the last ``keep_last_frames`` along time dim (dim=2) of 5D ``(B,H,T,tpf,d)`` KV.

    Typical Pi3 carry has the same ``T`` on every non-``None`` layer; we short-circuit when
    no truncation is needed. Per-layer work remains (separate storages); ``.contiguous()`` is
    skipped when the time-tail view is already contiguous.
    """
    if past_key_values is None:
        return None
    keep = int(keep_last_frames)
    if all(item is None or item[0].shape[2] <= keep for item in past_key_values):
        return list(past_key_values)

    sl = slice(-keep, None)

    def _one(
        item: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if item is None:
            return None
        k, v = item[0], item[1]
        if k.shape[2] <= keep:
            return (k, v)
        kt = k[:, :, sl, :, :]
        vt = v[:, :, sl, :, :]
        return (_maybe_contiguous(kt), _maybe_contiguous(vt))

    return [_one(item) for item in past_key_values]


def _detach_streaming_stream_carry_if_any(obj: Any) -> Optional[Any]:
    try:
        from abot_recon.modeling.streaming.kv_state import StreamingKVState
    except ImportError:
        return None
    if not isinstance(obj, StreamingKVState):
        return None
    return StreamingKVState(
        num_reference_frames=int(obj.num_reference_frames),
        local_window_frames=int(obj.local_window_frames),
        num_summary_tokens=int(obj.num_summary_tokens),
        tpf=int(obj.tpf),
        reference_pk=obj.reference_pk.detach(),
        reference_pv=obj.reference_pv.detach(),
        reference_count=int(obj.reference_count),
        summary_pk=obj.summary_pk.detach(),
        summary_pv=obj.summary_pv.detach(),
        window_pk=obj.window_pk.detach(),
        window_pv=obj.window_pv.detach(),
        window_count=int(obj.window_count),
        total_frames_seen=int(obj.total_frames_seen),
        max_summary_frames=int(getattr(obj, "max_summary_frames", 0)),
        summary_frame_base=int(getattr(obj, "summary_frame_base", 0)),
    )


def detach_carry(
    past_key_values: Optional[List[KVLayerItem]],
) -> Optional[List[KVLayerItem]]:
    if past_key_values is None:
        return None
    out: List[KVLayerItem] = []
    for item in past_key_values:
        if item is None:
            out.append(None)
            continue
        hs = _detach_streaming_stream_carry_if_any(item)
        if hs is not None:
            out.append(hs)
            continue
        if len(item) == 3:
            k, v, meta = item
            meta_d = {
                kk: (vv.detach() if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in meta.items()
            }
            out.append((k.detach(), v.detach(), meta_d))
        else:
            k, v = item[0], item[1]
            out.append((k.detach(), v.detach()))
    return out

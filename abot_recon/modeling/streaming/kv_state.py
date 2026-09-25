"""SDPA state for causal windowed streaming attention.

Past state excludes the current frame. For a window of ``W`` frames, the
state retains at most ``W - 1`` complete past-frame K/V blocks; the current
frame is supplied by the active query step. Absolute frame indices are kept
for 3D rotary positions even after older frames leave the window.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


def token_width_for_query(
    fk: int,
    *,
    T_query: int,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
    tpf: int,
) -> int:
    ka = int(num_reference_frames)
    Wloc = max(int(local_window_frames), 1)
    n_comp = max(int(num_summary_tokens), 0)
    if fk < ka or fk >= T_query - Wloc + 1:
        return int(tpf)
    return min(n_comp, int(tpf))


def _nc_eff(num_summary_tokens: int, tpf: int) -> int:
    return min(max(int(num_summary_tokens), 0), int(tpf))


def _summary_row_count(total_past_frames: int, ka: int, W: int) -> int:
    """非锚定、只存 summary 前缀的过去帧个数（对已提交 ``total_past_frames`` 帧的 past）。"""
    if total_past_frames <= ka:
        return 0
    wc = max(0, int(W) - 1)
    return max(0, int(total_past_frames) - int(ka) - wc)


@dataclass
class StreamingKVState:
    """pre-norm K/V carry（与 dense ``pk,pv`` 同空间：层内先于 k_norm / RoPE 的 rawKV）."""

    num_reference_frames: int
    local_window_frames: int
    num_summary_tokens: int
    tpf: int

    reference_pk: Tensor
    reference_pv: Tensor
    reference_count: int

    summary_pk: Tensor
    summary_pv: Tensor

    window_pk: Tensor
    window_pv: Tensor
    window_count: int

    total_frames_seen: int
    max_summary_frames: int = 0
    summary_frame_base: int = 0

    @property
    def window_cap(self) -> int:
        return max(0, int(self.local_window_frames) - 1)

    @property
    def nc_eff(self) -> int:
        return _nc_eff(self.num_summary_tokens, self.tpf)

    @classmethod
    def empty(
        cls,
        *,
        batch_heads_ref: Tensor,
        tpf: int,
        num_reference_frames: int,
        local_window_frames: int,
        num_summary_tokens: int,
        max_summary_frames: int = 0,
    ) -> "StreamingKVState":
        """batch_heads_ref: any (B, nh, ...) 张量，提供 device/dtype。"""
        B, nh = int(batch_heads_ref.shape[0]), int(batch_heads_ref.shape[1])
        hd = int(batch_heads_ref.shape[-1])
        ka = int(num_reference_frames)
        Wcap = max(0, int(local_window_frames) - 1)
        nc = _nc_eff(num_summary_tokens, tpf)
        z = torch.zeros
        dtype, device = batch_heads_ref.dtype, batch_heads_ref.device
        wp = (
            z((B, nh, Wcap, tpf, hd), dtype=dtype, device=device) if Wcap > 0 else z((B, nh, 0, tpf, hd), dtype=dtype, device=device)
        )
        wv = (
            z((B, nh, Wcap, tpf, hd), dtype=dtype, device=device) if Wcap > 0 else z((B, nh, 0, tpf, hd), dtype=dtype, device=device)
        )
        return cls(
            num_reference_frames=ka,
            local_window_frames=int(local_window_frames),
            num_summary_tokens=int(num_summary_tokens),
            max_summary_frames=max(0, int(max_summary_frames)),
            tpf=int(tpf),
            reference_pk=z((B, nh, ka, tpf, hd), dtype=dtype, device=device),
            reference_pv=z((B, nh, ka, tpf, hd), dtype=dtype, device=device),
            reference_count=0,
            summary_pk=z((B, nh, 0, nc, hd), dtype=dtype, device=device),
            summary_pv=z((B, nh, 0, nc, hd), dtype=dtype, device=device),
            window_pk=wp,
            window_pv=wv,
            window_count=0,
            total_frames_seen=0,
        )

    def clone_buffers(self) -> "StreamingKVState":
        return deepcopy(self)

    def _first_stored_summary_fk(self, T_q: int) -> int:
        ka = int(self.num_reference_frames)
        nc_rows = _summary_row_count(T_q, ka, int(self.local_window_frames))
        stored = int(self.summary_pk.shape[2])
        return ka + max(0, nc_rows - stored)

    def tokens_per_frame_past(self) -> List[int]:
        """past 帧 0 … T_p−1 的宽度列表；``T_query = T_p``（本步 query 帧下标）。"""
        T_p = int(self.total_frames_seen)
        T_q = T_p
        ka = int(self.num_reference_frames)
        Wloc = max(int(self.local_window_frames), 1)
        tpf = int(self.tpf)
        nc = self.nc_eff
        first_summary = self._first_stored_summary_fk(T_q)
        nc_rows = _summary_row_count(T_q, ka, int(self.local_window_frames))
        last_summary = ka + nc_rows - 1
        out: List[int] = []
        for fk in range(T_p):
            if fk < ka or fk >= T_q - Wloc + 1:
                out.append(
                    token_width_for_query(
                        fk,
                        T_query=T_q,
                        num_reference_frames=ka,
                        local_window_frames=Wloc,
                        num_summary_tokens=int(self.num_summary_tokens),
                        tpf=tpf,
                    )
                )
            elif nc > 0 and first_summary <= fk <= last_summary:
                out.append(min(nc, tpf))
            else:
                out.append(0)
        return out

    def bias_fingerprint_tokens(self) -> Tuple[int, ...]:
        return tuple(self.tokens_per_frame_past())

    def _gather_frame_kv(self, fk: int) -> Tuple[Tensor, Tensor, int]:
        tpf = int(self.tpf)
        ka = int(self.num_reference_frames)
        T_q = int(self.total_frames_seen)
        Wloc = max(int(self.local_window_frames), 1)
        need = token_width_for_query(
            fk,
            T_query=T_q,
            num_reference_frames=ka,
            local_window_frames=Wloc,
            num_summary_tokens=int(self.num_summary_tokens),
            tpf=tpf,
        )

        if need == 0:
            B, nh = int(self.reference_pk.shape[0]), int(self.reference_pk.shape[1])
            hd = int(self.reference_pk.shape[-1])
            z = torch.empty((B, nh, 0, hd), dtype=self.reference_pk.dtype, device=self.reference_pk.device)
            return z, z, 0

        if fk < ka:
            if fk >= self.reference_count:
                raise RuntimeError(f"reference miss fk={fk} reference_count={self.reference_count}")
            kk = self.reference_pk[:, :, fk, :need, :].contiguous()
            vv = self.reference_pv[:, :, fk, :need, :].contiguous()
            return kk, vv, need

        nc_rows = _summary_row_count(T_q, ka, int(self.local_window_frames))
        first_summary_fk = self._first_stored_summary_fk(T_q)
        last_summary_fk = ka + nc_rows - 1
        if nc_rows > 0 and fk >= first_summary_fk and fk <= last_summary_fk:
            r = fk - first_summary_fk
            if r >= self.summary_pk.shape[2]:
                raise RuntimeError(f"summary row miss fk={fk} Nc={self.summary_pk.shape[2]}")
            kk = self.summary_pk[:, :, r, :need, :].contiguous()
            vv = self.summary_pv[:, :, r, :need, :].contiguous()
            return kk, vv, need

        win_first = ka + nc_rows
        if fk >= win_first and fk < T_q:
            r = fk - win_first
            if r < 0 or r >= self.window_count:
                raise RuntimeError(
                    f"window miss fk={fk} win_first={win_first} window_count={self.window_count}"
                )
            kk = self.window_pk[:, :, r, :need, :].contiguous()
            vv = self.window_pv[:, :, r, :need, :].contiguous()
            return kk, vv, need

        raise RuntimeError(
            f"T_q={T_q} ka={ka} fk={fk} reference_count={self.reference_count} "
            f"Nc={self.summary_pk.shape[2]} win={self.window_count} cap={self.window_cap}"
        )

    def _append_summary_row(self, row_k: Tensor, row_v: Tensor) -> None:
        cap = int(self.max_summary_frames)
        if cap > 0 and self.summary_pk.shape[2] >= cap:
            self.summary_pk = self.summary_pk[:, :, 1:].contiguous()
            self.summary_pv = self.summary_pv[:, :, 1:].contiguous()
            self.summary_frame_base += 1
        self.summary_pk = torch.cat([self.summary_pk, row_k], dim=2).contiguous()
        self.summary_pv = torch.cat([self.summary_pv, row_v], dim=2).contiguous()

    def stored_frame_token_widths(self) -> List[Tuple[int, int]]:
        """Return ``(absolute_frame_idx, stored_token_count)`` for visible past frames.

        Unlike :meth:`tokens_per_frame_past`, this representation is sparse: frames
        that have been dropped entirely (for example, old frames when
        ``num_summary_tokens=0``) are omitted.  Absolute frame indices are retained
        so temporal RoPE never renumbers a sliding window to ``0..W-1``.
        """
        T_p = int(self.total_frames_seen)
        spans: List[Tuple[int, int]] = []

        for frame_idx in range(int(self.reference_count)):
            spans.append((frame_idx, int(self.tpf)))

        stored_summary = int(self.summary_pk.shape[2])
        if self.nc_eff > 0 and stored_summary > 0:
            first_summary = self._first_stored_summary_fk(T_p)
            spans.extend(
                (first_summary + row, int(self.nc_eff))
                for row in range(stored_summary)
            )

        if int(self.window_count) > 0:
            first_window = T_p - int(self.window_count)
            spans.extend(
                (first_window + row, int(self.tpf))
                for row in range(int(self.window_count))
            )

        spans.sort(key=lambda item: item[0])
        return spans

    def flatten_past_kv_indexed(self) -> Tuple[Tensor, Tensor, List[Tuple[int, int]]]:
        """Flatten stored K/V and return their absolute frame-index/token spans."""
        spans = self.stored_frame_token_widths()
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        B, nh = int(self.reference_pk.shape[0]), int(self.reference_pk.shape[1])
        hd = int(self.reference_pk.shape[-1])
        if not spans:
            z = torch.empty((B, nh, 0, hd), dtype=self.reference_pk.dtype, device=self.reference_pk.device)
            return z, z, spans
        for fk, nw in spans:
            kk, vv, nk = self._gather_frame_kv(fk)
            if nk != nw:
                raise RuntimeError(
                    f"stored span width mismatch for frame {fk}: span={nw}, gathered={nk}"
                )
            ks.append(kk.reshape(B, nh, nk, hd))
            vs.append(vv.reshape(B, nh, nk, hd))
        return torch.cat(ks, dim=2).contiguous(), torch.cat(vs, dim=2).contiguous(), spans

    def flatten_past_kv(self) -> Tuple[Tensor, Tensor, List[int]]:
        """Backward-compatible dense frame-width metadata plus packed stored K/V."""
        k, v, spans = self.flatten_past_kv_indexed()
        lens = [0] * int(self.total_frames_seen)
        for frame_idx, token_count in spans:
            lens[frame_idx] = token_count
        return k, v, lens

    def append_committed_frame(self, k_frm: Tensor, v_frm: Tensor) -> None:
        """提交刚算完的这一帧全局下标 ``g = total_frames_seen`` 的全 ``tpf`` raw K/V."""
        ka = int(self.num_reference_frames)
        g = int(self.total_frames_seen)
        tpf = int(self.tpf)
        Wcap = self.window_cap
        nc = self.nc_eff

        if k_frm.shape != (k_frm.shape[0], k_frm.shape[1], tpf, k_frm.shape[-1]):
            raise ValueError(f"k_frm shape {tuple(k_frm.shape)} != (B,nh,tpf,{k_frm.shape[-1]})")

        if g < ka:
            self.reference_pk[:, :, g].copy_(k_frm)
            self.reference_pv[:, :, g].copy_(v_frm)
            self.reference_count = g + 1
            self.total_frames_seen = g + 1
            return

        if Wcap <= 0:
            if nc > 0:
                self._append_summary_row(
                    k_frm[:, :, :nc, :].unsqueeze(2),
                    v_frm[:, :, :nc, :].unsqueeze(2),
                )
            self.total_frames_seen = g + 1
            return

        if self.window_count < Wcap:
            self.window_pk[:, :, self.window_count].copy_(k_frm)
            self.window_pv[:, :, self.window_count].copy_(v_frm)
            self.window_count += 1
        else:
            ev_k = self.window_pk[:, :, 0].clone()
            ev_v = self.window_pv[:, :, 0].clone()
            if nc > 0:
                self._append_summary_row(
                    ev_k[:, :, :nc, :].unsqueeze(2),
                    ev_v[:, :, :nc, :].unsqueeze(2),
                )
            # Source and destination overlap on dim 2 -- without an explicit clone,
            # ``copy_`` produces undefined results (silently corrupts older window
            # entries on CUDA). The bug compounds across evictions; symptom is paged-
            # stream-vs-SDPA-stream growing divergence after the first eviction.
            shifted_k = self.window_pk[:, :, 1:Wcap].clone()
            shifted_v = self.window_pv[:, :, 1:Wcap].clone()
            self.window_pk[:, :, : Wcap - 1].copy_(shifted_k)
            self.window_pv[:, :, : Wcap - 1].copy_(shifted_v)
            self.window_pk[:, :, Wcap - 1].copy_(k_frm)
            self.window_pv[:, :, Wcap - 1].copy_(v_frm)
        self.total_frames_seen = g + 1


def streamed_chunk_key_positions_pi3(
    pos_full_bt2: Tensor,
    carry: StreamingKVState,
    *,
    tpf_new: int,
    current_global_frame_idx: int,
) -> Tensor:
    """dense ``pos_full_bt2`` 形状 ``(B, Nframes * tpf_new, 2)``（帧优先），拼接与 ``flatten_past_kv`` 同序。"""
    if pos_full_bt2.ndim != 3 or pos_full_bt2.shape[-1] != 2:
        raise ValueError(f"pos_full_bt2 expected (B,L,2), got {tuple(pos_full_bt2.shape)}")
    lens = carry.tokens_per_frame_past()
    parts: List[Tensor] = []
    for fk, nw in enumerate(lens):
        base = fk * int(tpf_new)
        parts.append(pos_full_bt2[:, base : base + int(nw), :])
    g = int(current_global_frame_idx)
    base_new = g * int(tpf_new)
    parts.append(pos_full_bt2[:, base_new : base_new + int(tpf_new), :])
    return torch.cat(parts, dim=1).contiguous()


def streaming_carry_from_rect_kv(
    pk: Tensor,
    pv: Tensor,
    *,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
    max_summary_frames: int = 0,
) -> StreamingKVState:
    """dense ``(B,nh,T,tpf,d)`` 一次性 bootstrap（首段 forward 后出现）。"""
    B, nh, T_tot, tpf, hd = pk.shape
    m = StreamingKVState.empty(
        batch_heads_ref=pk[:, :, 0, 0, :],
        tpf=int(tpf),
        num_reference_frames=num_reference_frames,
        local_window_frames=local_window_frames,
        num_summary_tokens=num_summary_tokens,
        max_summary_frames=max_summary_frames,
    )
    for t in range(int(T_tot)):
        m.append_committed_frame(pk[:, :, t].contiguous(), pv[:, :, t].contiguous())
    return m


def prune_window_carry_pk_pv(
    pk: Tensor,
    pv: Tensor,
    *,
    T_total: int,
    num_reference_frames: int,
    local_window_frames: int,
) -> Tuple[Tensor, Tensor, int]:
    """memory_mode=window：整帧 index_select。"""
    T_q = T_total - 1
    ka = int(num_reference_frames)
    Wloc = max(int(local_window_frames), 1)
    keep = [fk for fk in range(T_total) if fk < ka or fk >= T_q - Wloc + 1]
    if len(keep) == T_total:
        return pk, pv, T_total
    idx_t = torch.tensor(keep, device=pk.device, dtype=torch.long)
    pk2 = pk.index_select(2, idx_t).contiguous()
    pv2 = pv.index_select(2, idx_t).contiguous()
    return pk2, pv2, int(len(keep))

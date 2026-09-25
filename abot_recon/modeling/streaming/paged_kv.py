"""FlashInfer paged KV cache for ABotReconNetwork streaming inference.

The released model retains full K/V tokens for the active causal window. Each
frame occupies one page per global layer. Once the window is full, appending a
new frame recycles the oldest page. The visible page table therefore follows
the temporal order of the current local window and does not grow with the
processed sequence length.

FlashInfer planning is performed once per frame and reused across global
layers, whose page identifiers evolve in lockstep.
"""

from __future__ import annotations

import collections
import math
from typing import Dict, List

import torch
from torch import Tensor

try:
    import flashinfer  # type: ignore
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False


def flashinfer_available() -> bool:
    return _FLASHINFER_AVAILABLE


class PagedKVCacheManager:
    """Paged K/V storage for the model's fixed causal window.

    The release constructor fixes the inactive compatibility fields to zero;
    only ``local_window_frames`` contributes visible history. ``force_fp32``
    is retained as a validation path that gathers the visible pages and runs
    PyTorch SDPA instead of a FlashInfer kernel.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        tpf: int,
        num_heads: int,
        head_dim: int,
        num_reference_frames: int,
        local_window_frames: int,
        num_summary_tokens: int,
        memory_mode: str,
        max_total_frames: int,
        max_summary_frames: int = 0,
        dtype: torch.dtype,
        device: torch.device,
        force_fp32: bool = False,
        fa3: bool = False,
    ):
        if memory_mode not in ("streaming", "window"):
            raise ValueError(f"memory_mode must be 'streaming' or 'window', got {memory_mode!r}")
        if local_window_frames < 1:
            raise ValueError(f"local_window_frames must be >= 1, got {local_window_frames}")
        if num_reference_frames < 0:
            raise ValueError(f"num_reference_frames must be >= 0, got {num_reference_frames}")
        if num_summary_tokens < 0 or num_summary_tokens > tpf:
            raise ValueError(
                f"num_summary_tokens must be in [0, tpf={tpf}], got {num_summary_tokens}"
            )

        if not force_fp32 and not _FLASHINFER_AVAILABLE:
            raise RuntimeError(
                "flashinfer is not installed. Install with `pip install flashinfer-python` "
                "or pass force_fp32=True for the debug gather+SDPA path."
            )

        self.num_layers = int(num_layers)
        self.tpf = int(tpf)
        self.page_size = int(tpf)  # one full frame per page
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.ka = int(num_reference_frames)
        self.W = int(local_window_frames)  # physical window capacity (includes current)
        self.n_summary = int(num_summary_tokens) if memory_mode == "streaming" else 0
        self.memory_mode = memory_mode
        self.max_summary_frames = max(0, int(max_summary_frames))
        self.device = device
        self.force_fp32 = bool(force_fp32)
        self.storage_dtype = torch.float32 if self.force_fp32 else dtype

        # ── Page pool sizing ────────────────────────────────────────────────
        # reference: ka pages, IDs [0, ka)
        # window+free recyclable patch pool: W pages plus a small headroom (16),
        #     IDs [ka, ka + W + 16)
        # summary: packs n_summary rows per page; sized for FIFO cap or full clip
        self._n_reference_pages = self.ka
        self._n_recyclable_pages = self.W + 16
        if self.n_summary > 0:
            if self.max_summary_frames > 0:
                pool_summary_frames = self.max_summary_frames
            else:
                pool_summary_frames = max(0, int(max_total_frames) - self.ka - self.W + 1)
            self._n_summary_pages = math.ceil(
                pool_summary_frames * self.n_summary / self.page_size
            ) + 16
        else:
            self._n_summary_pages = 0
        self.max_num_pages = (
            self._n_reference_pages + self._n_recyclable_pages + self._n_summary_pages
        )

        self._reference_id_lo = 0
        self._reference_id_hi = self.ka  # exclusive
        self._recyc_id_lo = self.ka
        self._recyc_id_hi = self.ka + self._n_recyclable_pages
        self._summary_id_lo = self._recyc_id_hi
        self._summary_id_hi = self.max_num_pages

        # ── Physical paged KV cache: one tensor per layer ───────────────────
        # Shape [max_num_pages, 2, page_size, H, D] (NHD layout, K=0, V=1)
        self.kv_caches: List[Tensor] = [
            torch.zeros(
                self.max_num_pages, 2, self.page_size, num_heads, head_dim,
                dtype=self.storage_dtype, device=device,
            )
            for _ in range(self.num_layers)
        ]

        # ── Per-layer state (identical across layers, evolved in lockstep) ──
        # Reference: ordered by frame index (0..ka-1)
        self.reference_pages: List[List[int]] = [[] for _ in range(self.num_layers)]
        # Window: ring buffer, leftmost = oldest, rightmost = newest = current
        self.window_pages: List[collections.deque] = [
            collections.deque() for _ in range(self.num_layers)
        ]
        self.free_recyclable: List[List[int]] = [
            list(range(self._recyc_id_lo, self._recyc_id_hi))
            for _ in range(self.num_layers)
        ]
        # Summary: append-only list of page IDs in chronological order
        self.summary_pages: List[List[int]] = [[] for _ in range(self.num_layers)]
        self.free_summary: List[List[int]] = [
            list(range(self._summary_id_lo, self._summary_id_hi))
            for _ in range(self.num_layers)
        ]
        self.summary_token_count: List[int] = [0] * self.num_layers
        self.summary_frame_base: List[int] = [0] * self.num_layers
        self.frame_count: List[int] = [0] * self.num_layers

        # ── FlashInfer wrapper (one, plan reused across layers per step) ───
        if not self.force_fp32:
            self.workspace_buffer = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout="NHD",
                backend="fa3" if fa3 else "fa2",
            )
        else:
            self.prefill_wrapper = None  # debug path uses gather + SDPA

        self._qo_indptr = torch.tensor(
            [0, self.page_size], dtype=torch.int32, device=device
        )

        # Per-step plan state (set by compute_attention when block_idx == 0)
        self._plan_step_id: int = -1

    # =========================================================================
    # Public API
    # =========================================================================

    def reset(self) -> None:
        """Reset all per-layer state. Pre-allocated KV tensors are NOT zeroed."""
        for layer_idx in range(self.num_layers):
            self.reference_pages[layer_idx].clear()
            self.window_pages[layer_idx].clear()
            self.summary_pages[layer_idx].clear()
            self.summary_token_count[layer_idx] = 0
            self.summary_frame_base[layer_idx] = 0
            self.frame_count[layer_idx] = 0
            self.free_recyclable[layer_idx] = list(range(self._recyc_id_lo, self._recyc_id_hi))
            self.free_summary[layer_idx] = list(range(self._summary_id_lo, self._summary_id_hi))
        self._plan_step_id = -1

    def append_frame(self, layer_idx: int, k_nhd: Tensor, v_nhd: Tensor) -> None:
        """Append one frame's post-RoPE K/V to layer ``layer_idx``'s pool.

        Args:
            layer_idx: global decoder layer index (0..num_layers-1).
            k_nhd, v_nhd: [tpf, num_heads, head_dim] (NHD), POST q_norm/k_norm + RoPE.
        """
        if k_nhd.shape != (self.page_size, self.num_heads, self.head_dim):
            raise ValueError(
                f"k_nhd shape {tuple(k_nhd.shape)} != "
                f"({self.page_size}, {self.num_heads}, {self.head_dim})"
            )
        if v_nhd.shape != k_nhd.shape:
            raise ValueError(f"v shape {tuple(v_nhd.shape)} != k shape {tuple(k_nhd.shape)}")

        g = self.frame_count[layer_idx]
        cache = self.kv_caches[layer_idx]
        k_typed = k_nhd.to(self.storage_dtype)
        v_typed = v_nhd.to(self.storage_dtype)

        # ── Reference: first ka frames ─────────────────────────────────────────
        if g < self.ka:
            pid = self._reference_id_lo + g
            cache[pid, 0].copy_(k_typed)
            cache[pid, 1].copy_(v_typed)
            self.reference_pages[layer_idx].append(pid)
            self.frame_count[layer_idx] = g + 1
            return

        # ── Window: non-reference frame ────────────────────────────────────────
        # If window full, evict oldest (with summary pickup if applicable).
        if len(self.window_pages[layer_idx]) >= self.W:
            self._evict_oldest_window(layer_idx)

        pid = self.free_recyclable[layer_idx].pop()
        cache[pid, 0].copy_(k_typed)
        cache[pid, 1].copy_(v_typed)
        self.window_pages[layer_idx].append(pid)
        self.frame_count[layer_idx] = g + 1

    def compute_attention(self, layer_idx: int, q_nhd: Tensor) -> Tensor:
        """Run paged attention for layer ``layer_idx``.

        Plan is built on layer_idx==0 and reused across remaining layers within
        the same frame step.

        Args:
            q_nhd: [tpf, num_heads, head_dim] (NHD), POST q_norm + RoPE.

        Returns:
            out: [tpf, num_heads, head_dim] (NHD).
        """
        if q_nhd.shape != (self.page_size, self.num_heads, self.head_dim):
            raise ValueError(
                f"q shape {tuple(q_nhd.shape)} != "
                f"({self.page_size}, {self.num_heads}, {self.head_dim})"
            )

        if self.force_fp32:
            return self._compute_attention_fp32_debug(layer_idx, q_nhd)

        # New plan per frame step: triggered by layer 0
        if layer_idx == 0:
            visible = self.build_visible_page_table(0)
            last_len = self.compute_last_page_len(0)
            if not visible:
                raise RuntimeError("visible page table empty -- call append_frame first")
            if not (1 <= last_len <= self.page_size):
                raise RuntimeError(
                    f"last_page_len {last_len} not in [1, {self.page_size}]"
                )

            kv_indices = torch.tensor(visible, dtype=torch.int32, device=self.device)
            kv_indptr = torch.tensor([0, len(visible)], dtype=torch.int32, device=self.device)
            kv_last = torch.tensor([last_len], dtype=torch.int32, device=self.device)

            self.prefill_wrapper.plan(
                self._qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                head_dim_qk=self.head_dim,
                page_size=self.page_size,
                causal=False,  # mask encoded in the visible-page selection
                pos_encoding_mode="NONE",  # RoPE applied externally before append
                q_data_type=self.storage_dtype,
            )
            self._plan_step_id += 1

        return self.prefill_wrapper.run(
            q=q_nhd.to(self.storage_dtype).contiguous(),
            paged_kv_cache=self.kv_caches[layer_idx],
        )

    def get_stats(self, layer_idx: int = 0) -> Dict[str, int]:
        return {
            "frames": int(self.frame_count[layer_idx]),
            "reference": len(self.reference_pages[layer_idx]),
            "window": len(self.window_pages[layer_idx]),
            "summary_pages": len(self.summary_pages[layer_idx]),
            "summary_tokens": int(self.summary_token_count[layer_idx]),
            "free_recyc": len(self.free_recyclable[layer_idx]),
            "free_summary": len(self.free_summary[layer_idx]),
        }

    # =========================================================================
    # Helpers -- visible table & layout
    # =========================================================================

    def build_visible_page_table(self, layer_idx: int) -> List[int]:
        """Visible pages at attention time = reference + window + summary.

        Summary placed LAST so ``paged_kv_last_page_len`` describes the partial
        summary tail. All reference and window pages are fully written ``page_size``
        tokens, so FlashInfer treats them as full and only the summary tail page
        uses ``last_page_len``.
        """
        return (
            list(self.reference_pages[layer_idx])
            + list(self.window_pages[layer_idx])
            + list(self.summary_pages[layer_idx])
        )

    def compute_last_page_len(self, layer_idx: int) -> int:
        if self.summary_pages[layer_idx]:
            tail = self.summary_token_count[layer_idx] % self.page_size
            return self.page_size if tail == 0 else tail
        return self.page_size

    # =========================================================================
    # Internal write helpers
    # =========================================================================

    def _evict_oldest_window(self, layer_idx: int) -> None:
        """Pop leftmost window page, optionally extract summary prefix, recycle."""
        evicted_pid = self.window_pages[layer_idx].popleft()
        if self.n_summary > 0:
            cache = self.kv_caches[layer_idx]
            comp_k = cache[evicted_pid, 0, : self.n_summary]
            comp_v = cache[evicted_pid, 1, : self.n_summary]
            self._write_summary_tokens(layer_idx, comp_k, comp_v)
        self.free_recyclable[layer_idx].append(evicted_pid)

    def _summary_frames_stored(self, layer_idx: int) -> int:
        if self.n_summary <= 0:
            return 0
        return int(self.summary_token_count[layer_idx]) // self.n_summary

    def _clear_summary_pool(self, layer_idx: int) -> None:
        for pid in self.summary_pages[layer_idx]:
            self.free_summary[layer_idx].append(pid)
        self.summary_pages[layer_idx].clear()
        self.summary_token_count[layer_idx] = 0

    def _drop_oldest_summary_frame(self, layer_idx: int) -> None:
        """Drop the oldest ``n_summary`` tokens from the summary stream (one evicted frame)."""
        nc = self.n_summary
        total = int(self.summary_token_count[layer_idx])
        if nc <= 0 or total < nc:
            return
        cache = self.kv_caches[layer_idx]
        chunks_k: List[Tensor] = []
        chunks_v: List[Tensor] = []
        pos = 0
        for i, pid in enumerate(self.summary_pages[layer_idx]):
            n = self.page_size if (pos + self.page_size <= total) else (total - pos)
            chunks_k.append(cache[pid, 0, :n].clone())
            chunks_v.append(cache[pid, 1, :n].clone())
            pos += n
        k_all = torch.cat(chunks_k, dim=0)[nc:]
        v_all = torch.cat(chunks_v, dim=0)[nc:]
        self._clear_summary_pool(layer_idx)
        self.summary_frame_base[layer_idx] += 1
        for off in range(0, int(k_all.shape[0]), nc):
            self._write_summary_tokens_one_frame(
                layer_idx, k_all[off : off + nc], v_all[off : off + nc]
            )

    def _enforce_summary_fifo_before_write(self, layer_idx: int) -> None:
        cap = int(self.max_summary_frames)
        if cap <= 0 or self.n_summary <= 0:
            return
        while self._summary_frames_stored(layer_idx) >= cap:
            self._drop_oldest_summary_frame(layer_idx)

    def _write_summary_tokens(self, layer_idx: int, comp_k: Tensor, comp_v: Tensor) -> None:
        """Append summary tokens to layer ``layer_idx``'s summary stream.

        ``comp_*`` may be length ``n_summary`` (one frame) or longer (bulk rewrite).
        Handles page-boundary crossing.  Pages packed contiguously, no padding
        except tail of the most recent page.
        """
        n_write = int(comp_k.shape[0])
        if n_write == 0:
            return
        if n_write % self.n_summary != 0:
            raise ValueError(
                f"summary write length {n_write} must be a multiple of n_summary={self.n_summary}"
            )
        for off in range(0, n_write, self.n_summary):
            self._enforce_summary_fifo_before_write(layer_idx)
            self._write_summary_tokens_one_frame(layer_idx, comp_k[off : off + self.n_summary], comp_v[off : off + self.n_summary])

    def _write_summary_tokens_one_frame(self, layer_idx: int, comp_k: Tensor, comp_v: Tensor) -> None:
        n_write = self.n_summary
        if n_write == 0:
            return

        cache = self.kv_caches[layer_idx]
        written = 0
        remaining = n_write

        while remaining > 0:
            tail = self.summary_token_count[layer_idx] % self.page_size
            if tail == 0:
                # need a fresh summary page
                if not self.free_summary[layer_idx]:
                    raise RuntimeError(
                        f"summary page pool exhausted (layer {layer_idx}, "
                        f"summary_tokens={self.summary_token_count[layer_idx]}). "
                        f"Increase max_summary_frames or paged_max_total_frames."
                    )
                new_pid = self.free_summary[layer_idx].pop(0)  # FIFO for chronology
                self.summary_pages[layer_idx].append(new_pid)

            page_pid = self.summary_pages[layer_idx][-1]
            space = self.page_size - tail
            n_chunk = min(remaining, space)
            cache[page_pid, 0, tail : tail + n_chunk].copy_(comp_k[written : written + n_chunk])
            cache[page_pid, 1, tail : tail + n_chunk].copy_(comp_v[written : written + n_chunk])

            self.summary_token_count[layer_idx] += n_chunk
            written += n_chunk
            remaining -= n_chunk

    # =========================================================================
    # Debug fp32 path (no FlashInfer)
    # =========================================================================

    def _compute_attention_fp32_debug(self, layer_idx: int, q_nhd: Tensor) -> Tensor:
        """Gather visible K,V into a dense tensor and run fp32 SDPA.

        Matches the reference paged-attention layout
        force_fp32 branch for numerical comparison.
        """
        import torch.nn.functional as F

        visible = self.build_visible_page_table(layer_idx)
        last_len = self.compute_last_page_len(layer_idx)
        cache = self.kv_caches[layer_idx]

        parts_k, parts_v = [], []
        for i, pid in enumerate(visible):
            n = last_len if i == len(visible) - 1 else self.page_size
            parts_k.append(cache[pid, 0, :n])  # [n, H, D]
            parts_v.append(cache[pid, 1, :n])

        k_flat = torch.cat(parts_k, dim=0).float()  # [Lk, H, D]
        v_flat = torch.cat(parts_v, dim=0).float()

        q = q_nhd.float().permute(1, 0, 2).unsqueeze(0)       # [1, H, Lq, D]
        k = k_flat.permute(1, 0, 2).unsqueeze(0)               # [1, H, Lk, D]
        v = v_flat.permute(1, 0, 2).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.squeeze(0).permute(1, 0, 2).to(q_nhd.dtype)  # [Lq, H, D]

    # =========================================================================
    # Introspection helpers (testing / debugging)
    # =========================================================================

    def visible_frame_indices(self, layer_idx: int) -> List[int]:
        """Frame indices contributing to visible K,V at attention time.

        Order matches build_visible_page_table.  Summary entries listed by
        their original frame index, repeated per summary slot if needed.
        Used by tests to construct an SDPA-equivalent dense baseline.
        """
        ka = self.ka
        out: List[int] = []
        # reference pages == frames 0..ka-1
        for i in range(len(self.reference_pages[layer_idx])):
            out.append(i)
        # window pages: chronological from oldest to newest
        n_evicted = max(0, self.frame_count[layer_idx] - self.ka - self.W)
        # oldest still-in-window frame index = ka + n_evicted (after eviction it's gone)
        oldest_win = ka + n_evicted
        n_win = len(self.window_pages[layer_idx])
        for i in range(n_win):
            out.append(oldest_win + i)
        # summary: only frames still stored after optional FIFO trim
        n_stored = self._summary_frames_stored(layer_idx)
        first_summary = ka + max(0, n_evicted - n_stored)
        for i in range(first_summary, ka + n_evicted):
            out.append(i)
        return out

    @property
    def installed_flashinfer(self) -> bool:
        return _FLASHINFER_AVAILABLE

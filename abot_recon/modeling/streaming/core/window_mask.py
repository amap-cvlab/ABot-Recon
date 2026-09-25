"""Additive attention bias: past K/V then new K/V; causal + sliding temporal window.

``temporal_window_frames`` (**W**) is the **number of frame slots in the window including the
query frame** (so **W−1** are strictly in the past relative to ``T_q``). A key frame ``T_k`` is
inside the window iff ``T_q - T_k <= W - 1`` (together with causality ``T_k <= T_q``). Matches
``streaming`` streaming dense window: ``T_k >= T_q - W + 1``.
"""

import torch


def streaming_chunk_attention_bias(
    num_past_frames: int,
    num_new_frames: int,
    tokens_per_frame: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    temporal_window_frames: int = 12,
    num_reference_frames: int = 0,
) -> torch.Tensor:
    """Return ``(Lq, Lk)`` bias for SDPA.

    Keys are laid out as ``[past frames][new chunk frames]`` along time (each frame has
    ``tokens_per_frame`` tokens). Queries are **only** the new-chunk tokens.

    Implemented in **frame space** `(t_N × (t_p+t_N))`, then Kronecker-expanded by
    ``tokens_per_frame``, so intermediate masks are tiny. This avoids O(L_q·L_k) int64
    broadcast temporaries (~8 bytes/token-pair), which exhausted GPU memory at 128+ frames.

    Index **frame slots** along that concatenation: past uses ``0 .. t_p-1``, new uses
    ``t_p .. t_p + t_n - 1``. For a query belonging to new frame ``fq`` (0-based in chunk),
    its slot index is ``T_q = t_p + fq``. A key at slot ``T_k`` is visible iff:

    - **Causal:** ``T_k <= T_q`` (no future).
    - **Window:** ``T_q - T_k < W`` i.e. ``T_q - T_k <= W - 1`` with ``W = temporal_window_frames``
      (**W** = frame slots in the window **including** the query frame; **W−1** older frames).

    - **Optional references:** if ``num_reference_frames = n > 0``, any key with frame index
      ``T_k < n`` is also allowed (still respecting causality ``T_k <= T_q``). Same semantics
      as the reference band in ``streaming_attention_bias``.

    This matches the usual dense carry when past KV is already the trailing window in time
    order; no separate global-id tensor is required.
    """
    tpf = int(tokens_per_frame)
    t_p = int(num_past_frames)
    t_n = int(num_new_frames)
    W = int(temporal_window_frames)
    if W < 1:
        raise ValueError(f"temporal_window_frames must be >= 1, got {W}")
    k_reference = max(int(num_reference_frames), 0)

    if not dtype.is_floating_point:
        dtype = torch.float32

    # Frame-factorized layout (cheap): visibility depends only on (T_q,T_k)
    # frame slots, identical for every token within each frame — so compute a
    # small (t_n)×(t_p+t_n) table and Kronecker-expand by tpf. This avoids O(Lq×Lk)
    # int64 temporals from broadcasting (e.g. 128×756 → ~9e9×8 B blows past large GPUs).
    Fq = t_n
    Fk = t_p + t_n
    fq = torch.arange(Fq, device=device, dtype=torch.long).unsqueeze(1)
    fk = torch.arange(Fk, device=device, dtype=torch.long).unsqueeze(0)

    # Query frames are strictly the new chunk: global slot T_q = t_p … t_p+t_n-1.
    T_q_frame = (t_p + fq).long()
    # Keys span the concatenated timeline: global slot equals fk ∈ [0, t_p+t_n).
    T_k_frame = fk.long()

    # One fused boolean grid (Fq×Fk — small); avoid keeping several temps alive.
    blocked_frames = (T_k_frame > T_q_frame) | (
        ((T_q_frame - T_k_frame) > (W - 1)) & ~(T_k_frame < int(k_reference))
    )

    min_v = torch.finfo(dtype).min
    bias_f = torch.zeros((Fq, Fk), dtype=dtype, device=device)
    bias_f.masked_fill_(blocked_frames, min_v)
    del blocked_frames

    Lq = t_n * tpf
    Lk = (t_p + t_n) * tpf
    # Views + one contiguous allocation; avoids repeat_interleave's extra staging.
    bias = (
        bias_f[:, None, :, None]
        .expand(Fq, tpf, Fk, tpf)
        .reshape(Lq, Lk)
        .contiguous()
    )
    return bias

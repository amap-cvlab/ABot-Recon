"""Dense additive bias for streaming attention (SDPA path only)."""

import torch


def streaming_attention_bias(
    num_past_frames: int,
    num_new_frames: int,
    tokens_per_frame: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
) -> torch.Tensor:
    """(Lq, Lk) bias for queries = new chunk only; keys = past frames + new chunk.

    ``local_window_frames`` (**W**) is the **number of frame slots in the dense local window
    including the query frame** (so **W−1** are strictly in the past). Dense-local visibility:
    ``T_k >= T_q - W + 1`` (and causal / reference / summary rules). Same **W** meaning as
    ``streaming_chunk_attention_bias(..., temporal_window_frames=W)`` in ``chunk_mask``.

    Built in **(query_frame × key_frame × key-token-in-frame)** space: visibility does not depend
    on which **query** token inside a frame (same ``T_q`` for the whole frame). Intermediate
    masks avoid O(Lq·Lk) int broadcast temporaries from a naive `(T_k vs T_q)` full grid.
    """
    tpf = int(tokens_per_frame)
    t_p = int(num_past_frames)
    t_n = int(num_new_frames)
    k_reference = int(num_reference_frames)
    Wloc = max(int(local_window_frames), 1)
    n_reg = int(num_summary_tokens)

    if not dtype.is_floating_point:
        dtype = torch.float32

    Lq = t_n * tpf
    Lk = (t_p + t_n) * tpf
    min_val = torch.finfo(dtype).min

    # Global key frame indices: 0..t_p+t_n-1 along the concatenated K layout.
    Fq = int(t_n)
    Fk = int(t_p + t_n)

    fq = torch.arange(Fq, device=device, dtype=torch.long).unsqueeze(1)  # (Fq,1)
    fk = torch.arange(Fk, device=device, dtype=torch.long).unsqueeze(0)  # (1,Fk)
    T_q = t_p + fq  # (Fq,1): query frame slots for the new chunk
    T_k = fk  # (1,Fk)

    causal = T_k > T_q  # (Fq,Fk)
    in_reference = T_k < k_reference
    in_window = T_k >= (T_q - Wloc + 1)

    causal_ok = ~causal
    region_ok = in_reference | in_window
    # Intra-frame key index r_k ∈ [0,tpf): summary register band.
    rk = torch.arange(tpf, device=device, dtype=torch.long).view(1, 1, tpf)
    summary_ok = rk < int(n_reg)  # (1,1,tpf)

    allowed = causal_ok.unsqueeze(-1) & (region_ok.unsqueeze(-1) | summary_ok)  # (Fq,Fk,tpf)

    bias_f = torch.zeros((Fq, Fk, tpf), dtype=dtype, device=device)
    bias_f.masked_fill_(~allowed, min_val)

    # Same bias for every query token row within frame fq → expand rq dimension then reshape Lq×Lk.
    bias_4 = bias_f.unsqueeze(1).expand(Fq, tpf, Fk, tpf).reshape(Lq, Lk).contiguous()
    return bias_4


def packed_attention_bias(
    tokens_per_past_frame: list,
    tpf_new: int,
    *,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """(Lq, Lk) additive bias for **packed** past keys + dense new frame (``tpf_new`` tokens).

    Past keys are laid out as ``concat_f [0..T_p-1]`` with ``tokens_per_past_frame[fk]`` tokens per
    logical frame. Trailing keys are the new frame with index ``T_p`` and ``tpf_new`` tokens.
    Queries are **only** the new frame (``Lq = tpf_new``). ``T_q = T_p`` (global index of the query frame).
    Visibility matches :func:`streaming_attention_bias` (causal / reference / window / summary).
    """
    tpf_n = int(tpf_new)
    lens = [int(x) for x in tokens_per_past_frame]
    T_p = len(lens)
    T_q = T_p  # query frame global index
    k_reference = int(num_reference_frames)
    Wloc = max(int(local_window_frames), 1)
    n_reg = int(num_summary_tokens)
    if not dtype.is_floating_point:
        dtype = torch.float32
    min_val = torch.finfo(dtype).min

    keys: list[tuple[int, int]] = []
    for fk, ln in enumerate(lens):
        for rk in range(ln):
            keys.append((fk, rk))
    for rk in range(tpf_n):
        keys.append((T_q, rk))
    Lk = len(keys)
    Lq = tpf_n

    row = torch.zeros((Lk,), device=device, dtype=dtype)
    for ik, (Tk_k, rk_k) in enumerate(keys):
        causal_bad = Tk_k > T_q
        if causal_bad:
            row[ik] = min_val
            continue
        in_reference = Tk_k < k_reference
        in_window = Tk_k >= T_q - Wloc + 1
        summary_ok = rk_k < n_reg
        if not (in_reference or in_window or summary_ok):
            row[ik] = min_val
    bias = row.unsqueeze(0).expand(Lq, Lk).clone()
    return bias.contiguous()

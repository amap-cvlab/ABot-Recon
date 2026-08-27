# Third-party evaluation patches

These patches capture the exact tracked modifications present in the pinned
third-party checkouts used by the formal evaluation runs. Apply each patch to
the commit encoded in its filename with `git apply`.

| Patch | Purpose | Numerical status |
|---|---|---|
| `cut3r_8bc15dc_eval_memory.patch` | Avoid retaining recurrent states when `ret_state=false` | Output path unchanged; current FP32 recurrent evaluation is authoritative |
| `ttt3r_edd6d8c_eval_memory.patch` | Avoid retaining recurrent states when `ret_state=false` | Output path unchanged; current FP32 gated recurrent evaluation is authoritative |
| `longstream_72399ad_selective_dense.patch` | Copy dense outputs to CPU only for requested metric frames | Retained-frame outputs match the full-output path |
| `infinitevggt_7f9a5a2_required_keys.patch` | Copy only adapter-requested result keys to CPU | Dense values for retained keys are unchanged |
| `ovggt_b582391_required_keys.patch` | Copy only adapter-requested result keys to CPU | Dense values for retained keys are unchanged; repeated runs are deterministic |
| `lingbot_1f480ae_selective_dense.patch` | Preserve every pose while retaining depth only for requested metric frames | Patched file exactly matches the formal checkout; retained `pose_enc`/`depth` are bit-identical in the 80-frame cross-window A/B |

Horizon's disabled offline-branch fix and the experimental LingBot `f720b42`
cache patch are archived under `unrelease/third_party_patches/`; neither is
used by the formal evaluation.

Formal LingBot runs use upstream commit `1f480ae` plus
`lingbot_1f480ae_selective_dense.patch`. The patch avoids retaining dense maps
for frames excluded from reconstruction metrics and is recommended for long
sequences. The adapter also detects a clean official checkout automatically;
in that case it requests the standard full dense output and selects metric
frames afterward, preserving correctness at a higher peak-memory cost.

`SHA256SUMS` covers the patch files. Untracked files in the original model
checkouts are intentionally excluded because they were not imported by the
formal adapters.

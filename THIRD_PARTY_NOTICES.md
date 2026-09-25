# Third-Party Notices

ABot-Recon contains or depends on components with their own licenses. A notice
or license stated in an individual source file takes precedence for that file.

## Pi3

Core image encoding and geometric reconstruction code is derived from Pi3.

- Project: https://github.com/yyfz/Pi3
- Upstream license reference checked on 2026-09-25:
  [`9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/LICENSE`](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/LICENSE).
  This is a verification reference, not a claim that all bundled Pi3 code
  originates from that revision.
- That upstream license is BSD 3-Clause, with copyright `(c) 2025, the authors`;
  its full text is retained in
  [licenses/Pi3-UPSTREAM-BSD-3-Clause.txt](licenses/Pi3-UPSTREAM-BSD-3-Clause.txt).
- The development snapshot used by this project also carries the distinct
  notice headed "Pi3 for non-commercial purposes", retained without changes in
  [licenses/DEVELOPMENT-Pi3.txt](licenses/DEVELOPMENT-Pi3.txt). The current upstream
  BSD text does not establish that development additions or separately attributed
  third-party code may be relicensed. The exact scope of the development notice
  remains a provenance/permission question to resolve before publication.
- Upstream Pi3/Pi3X weights are separately described as CC BY-NC 4.0 with
  non-commercial research/education restrictions in the
  [pinned upstream README](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/README.md#-license).
  See [MODEL_LICENSE.md](MODEL_LICENSE.md) for this project's weight notice.

The bundled Pi3 tree includes separately attributed marepo, LoFTR, DINOv2 and
CroCo/DUSt3R-derived portions described below. Its upstream BSD label is not a
blanket license for those portions.

## marepo camera-head adaptation — permission review required

`abot_recon/modeling/pi3/models/layers/camera_head.py` identifies `ResConvBlock`
as adapted from marepo's
[`transformer/transformer.py` at `9a45e2bb07e5bb8cb997620088d352b439b13e0e`](https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172).
The local adaptation replaces the three 1x1 convolutions with linear layers.
The upstream source file carries `Copyright © Niantic, Inc. 2024.`

- Project: https://github.com/nianticlabs/marepo
- License: the custom non-commercial marepo license in the
  [pinned upstream LICENSE](https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/LICENSE),
  retained in full in [licenses/MAREPO.txt](licenses/MAREPO.txt).
  Its file-scope list includes `transformer/transformer.py`; the BSD license for
  other files later in that document does not cover this camera-head source.
- The upstream notice is `Copyright © Niantic, Inc. 2024. Patent Pending.`
  The license also specifies the notice `Copyright © Niantic, Inc. 2018. All
  rights reserved.` in clause 3.1; the upstream text is retained as written.

This is an unresolved release permission issue, not one cured by attribution:
the license limits use to its defined non-commercial purposes, clause 3.1 limits
source redistribution to the same upstream GitHub repository, and clause 3.3
sets terms for works based on the software. No separate redistribution
permission for this repository has been established here. Publication of the
affected adaptation requires confirmation of applicable permission/provenance
or another reviewed resolution; this notice does not grant that permission.

## LoFTR geometric warping

The `warp_kpts` implementation in
`abot_recon/modeling/pi3/utils/geometry.py` identifies an adaptation of
[`src/loftr/utils/geometry.py` at `94e98b695be18acb43d5d3250f52226a8e36f839`](https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py).
The adaptation adds normalized-coordinate sampling and optional interpolation,
mask and relative-depth-error behavior.

- Project: https://github.com/zju3dv/LoFTR
- License: Apache 2.0, verified against the
  [LICENSE at that source revision](https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/LICENSE);
  full text: [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
- Attribution: `Copyright SenseTime. All Rights Reserved.` The
  [upstream README at that revision](https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/README.md#copyright)
  attributes the work to ZJU-SenseTime Joint Lab of 3D Vision and states that its
  intellectual property belongs to SenseTime Group Ltd.

## CUT3R-format dataset adapters

The training dataset adapters under `abot_recon/training/datasets/` adapt CUT3R
data-reading and sampling code, with the project's development changes ported
into the shared release training interface. Added adapters cover TartanAir,
PointOdyssey, Spring, MVS-Synth, DynamicReplica, UASOL, ARKit HR, WildRGBD and
UnrealStereo4K Seq, ScanNet, Waymo, VKITTI2, HyperSim Seq, BlendedMVS Seq,
ARKitScenes and ScanNet++ Seq. Changes include shared indexing/I/O, bounded bad-frame
retries, sequence sampling and release augmentation integration.

Upstream counterparts are `src/dust3r/datasets/{tartanair,pointodyssey,spring,
mvs_synth,dynamic_replica,uasol,arkitscenes_highres,wildrgbd,unreal4k,scannet,waymo,
vkitti2,hypersim,blendedmvs,arkitscenes,scannetpp}.py`,
WildRGBD's parent `co3d.py`, and the sequence sampler in
`base/base_multiview_dataset.py`. The development pose-graph and foldback
sampling changes are not claimed to be part of upstream CUT3R.

`preprocess/preprocess_scannet_seq.py` is the project's sequence-oriented
ScanNet++ rewrite, using CUT3R/DUSt3R-derived undistortion and pixel-center
conventions. The two inlined intrinsics-conversion helpers retain the NAVER
copyright and CC BY-NC-SA 4.0 notice; derived portions are not relicensed by the
repository's top-level Apache notice. Device sequence selection, completion
manifests and restart-safe export are development additions, not upstream CUT3R.

- CUT3R source: https://github.com/CUT3R/CUT3R
- Format/source reference: `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`
- CUT3R code license: CC BY-NC-SA 4.0; the upstream notice is retained in
  [licenses/CUT3R.txt](licenses/CUT3R.txt).
- The development snapshots used for this port also contain a Pi3
  non-commercial notice, retained verbatim in
  [licenses/DEVELOPMENT-Pi3.txt](licenses/DEVELOPMENT-Pi3.txt).

The repository's Apache-2.0 license does not replace these upstream terms.
Dataset downloads and model weights have their own terms, independent of
these loader files.

## cuRoPE2D / CroCo / DUSt3R

CUDA cuRoPE2D sources under `abot_recon/modeling/pi3/models/curope/` and
`abot_recon/modeling/pi3/models/layers/pos_embed.py` retain their Naver notices
and are licensed under CC BY-NC-SA 4.0.

- License: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
- Full license text: [licenses/CC-BY-NC-SA-4.0.txt](licenses/CC-BY-NC-SA-4.0.txt).
  This text also accompanies the CUT3R notice above.
- Retained source notice: `Copyright (C) 2022-present Naver Corporation.
  All rights reserved.`

## DINOv2 and SALAD

The Pi3 backbone vendors DINOv2-derived code under
`abot_recon/modeling/pi3/models/dinov2/`; the adjacent
`models/layers/attention.py` and `models/layers/block.py` also retain Meta's
DINOv2-style Apache-2.0 notices and source references.

- Retained source copyright: `(c) Meta Platforms, Inc. and affiliates.`
- DINOv2 project: https://github.com/facebookresearch/dinov2
- DINOv2 code license: Apache 2.0; full text:
  [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt), verified against
  [`7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE`](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
  on 2026-09-25. This is a license-verification reference; the exact upstream
  revision of the vendored development copy is not recorded here.

Separately, the optional loop-closure backend contains an independently
organized SALAD-compatible descriptor network, loads DINOv2 code through
Torch Hub, and loads external DINOv2 and SALAD checkpoints. The SALAD repository
itself is not vendored by that backend.

- SALAD project: https://github.com/serizba/salad
- External checkpoints remain subject to the terms published by their
  respective authors and distributors.

## Sparse GPU Pose-Graph Optimization

The matrix-free block-sparse PCG implementation under `abot_recon/sparse_loop/`
is project code developed for ABot-Recon and does not vendor the HorizonStream
loop implementation. The existing source headers in `gpu_pgo.py`, `retrieval.py`
and `sparse_keyframes.py` explicitly specify BSD-3-Clause and
`Copyright (c) 2026 ABot-Recon Authors`. Those headers are preserved; the matching
full license text is in
[licenses/SPARSE-LOOP-BSD-3-Clause.txt](licenses/SPARSE-LOOP-BSD-3-Clause.txt).
This notice does not relabel those files as Apache-2.0 or change their terms.

## FlashInfer

FlashInfer is an optional external dependency used by the paged-KV backend. It
is not vendored in this repository.

- Project: https://github.com/flashinfer-ai/flashinfer
- License: Apache 2.0

Evaluation-only repositories on the “eval” branch are external inputs. Their
own source and checkpoint licenses apply independently.

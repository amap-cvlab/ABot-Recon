import os
import shutil
import time
from contextlib import contextmanager
from types import MethodType
import numpy as np
import torch
from omegaconf import DictConfig
from typing import List, Tuple

import imageio.v3 as iio
from copy import deepcopy

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.fastmodel import TTT3R
from relpose.forward_timing import time_forward
from mv_recon.runtime_manifest import record_model_runtime


def assert_views_never_reset(views) -> None:
    """Long-trajectory CUT3R/TTT3R must keep one continuous recurrent state."""
    for index, view in enumerate(views):
        reset = view.get("reset")
        if reset is None:
            continue
        if bool(torch.as_tensor(reset).any().item()):
            raise ValueError(
                f"CUT3R/TTT3R long-pose input unexpectedly resets at frame {index}"
            )


@contextmanager
def pose_only_downstream_head(net):
    """Temporarily run the official pose decoder without dense point heads.

    The recurrent encoder/state update remains the unmodified official code.
    Only ``head.forward`` is narrowed to the exact camera-pose branch already
    used by the full head, which avoids retaining per-frame dense point maps.
    """
    from dust3r.heads.postprocess import postprocess_pose

    # ``net.head`` is often the function returned by
    # ``transpose_to_landscape``. The actual nn.Module (and pose decoder) is
    # kept in ``downstream_head``; patching it preserves the official wrapper.
    head = getattr(net, "downstream_head", None)
    if head is None:
        head = net.head
    if not getattr(head, "has_pose", False) or not hasattr(head, "pose_head"):
        raise RuntimeError("CUT3R/TTT3R checkpoint does not expose an official pose head")
    had_instance_forward = "forward" in head.__dict__
    previous_forward = head.__dict__.get("forward")

    def forward_pose_only(self, decout, _img_shape, **_kwargs):
        pose_token = decout[-1][:, 0].clone()
        with torch.cuda.amp.autocast(enabled=False):
            pose = self.pose_head(pose_token)
            pose = postprocess_pose(pose, self.pose_mode)
        return {"camera_pose": pose}

    head.forward = MethodType(forward_pose_only, head)
    try:
        yield
    finally:
        if had_instance_forward:
            head.forward = previous_forward
        else:
            delattr(head, "forward")


@contextmanager
def selective_dense_downstream_head(net, dense_indices):
    """Run the official dense head only on requested metric frames.

    The recurrent rollout and pose decoder run on every frame. Dense XYZ is a
    readout and is not fed back into CUT3R/TTT3R state, so suppressing that
    readout on non-metric frames is numerically neutral for retained outputs.
    """
    from dust3r.heads.postprocess import postprocess_pose

    requested = set(int(index) for index in dense_indices)
    head = getattr(net, "downstream_head", None)
    if head is None:
        head = net.head
    if not getattr(head, "has_pose", False) or not hasattr(head, "pose_head"):
        raise RuntimeError("CUT3R/TTT3R checkpoint does not expose an official pose head")
    original_forward = head.forward
    frame_index = 0

    def forward_selected(self, decout, img_shape, **kwargs):
        nonlocal frame_index
        current = frame_index
        frame_index += 1
        if current in requested:
            return original_forward(decout, img_shape, **kwargs)
        pose_token = decout[-1][:, 0].clone()
        with torch.cuda.amp.autocast(enabled=False):
            pose = self.pose_head(pose_token)
            pose = postprocess_pose(pose, self.pose_mode)
        return {"camera_pose": pose}

    head.forward = MethodType(forward_selected, head)
    try:
        yield
    finally:
        head.forward = original_forward


def _extract_pose_only_c2w(outputs) -> torch.Tensor:
    from dust3r.utils.camera import pose_encoding_to_camera

    predictions = outputs["pred"]
    views = outputs["views"]
    reset_flags = [
        bool(torch.as_tensor(view.get("reset", False)).any().item())
        for view in views
    ]
    return _poses_from_predictions(predictions, reset_flags, pose_encoding_to_camera)


def _move_view_to_device(view, device):
    ignore_keys = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
    for name, value in view.items():
        if name in ignore_keys:
            continue
        if isinstance(value, (tuple, list)):
            view[name] = [item.to(device, non_blocking=True) for item in value]
        elif torch.is_tensor(value):
            view[name] = value.to(device, non_blocking=True)
    return view


def _validate_reset_interval(reset_interval):
    if reset_interval is None:
        return None
    reset_interval = int(reset_interval)
    if reset_interval <= 0:
        return None
    return reset_interval


def _iter_prepared_views(filelist, model, device, reset_interval=None):
    """Yield official views with the paper's overlap-frame reset protocol."""
    reset_interval = _validate_reset_interval(reset_interval)
    stream_index = 0
    for index, path in enumerate(filelist):
        view = prepare_input([path], size=model.input_size, device=device)[0]
        is_reset = reset_interval is not None and (index + 1) % reset_interval == 0
        view["idx"] = stream_index
        view["instance"] = str(stream_index)
        view["reset"] = torch.tensor([is_reset], dtype=torch.bool)
        yield _move_view_to_device(view, device)
        stream_index += 1
        if is_reset:
            overlap_view = deepcopy(view)
            overlap_view["idx"] = stream_index
            overlap_view["instance"] = str(stream_index)
            overlap_view["reset"] = torch.tensor(
                [False], dtype=torch.bool, device=overlap_view["img"].device
            )
            yield overlap_view
            stream_index += 1


def _poses_from_predictions(
    predictions, reset_flags=None, pose_decoder=None
) -> torch.Tensor:
    if pose_decoder is None:
        from dust3r.utils.camera import pose_encoding_to_camera

        pose_decoder = pose_encoding_to_camera
    if reset_flags is None:
        reset_flags = [False] * len(predictions)
    if len(reset_flags) != len(predictions):
        raise ValueError(
            "CUT3R/TTT3R reset metadata/prediction count mismatch: "
            f"reset_flags={len(reset_flags)}, predictions={len(predictions)}"
        )

    reset_mask = torch.as_tensor(reset_flags, dtype=torch.bool)
    shifted_reset_mask = torch.cat([reset_mask.new_zeros(1), reset_mask[:-1]])
    kept_predictions = [
        prediction
        for prediction, drop in zip(predictions, shifted_reset_mask)
        if not bool(drop)
    ]
    kept_reset_mask = reset_mask[~shifted_reset_mask]
    poses = [
        pose_decoder(pred["camera_pose"].clone()).detach().cpu().float()
        for pred in kept_predictions
    ]
    if not poses:
        raise RuntimeError("CUT3R/TTT3R pose-only inference returned no poses")
    pose_tensor = torch.cat(poses, dim=0)
    if kept_reset_mask.any():
        identity = torch.eye(4, dtype=pose_tensor.dtype, device=pose_tensor.device)
        reset_poses = torch.where(
            kept_reset_mask[:, None, None], pose_tensor, identity
        )
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity[None], cumulative_bases[:-1]], dim=0)
        pose_tensor = shifted_bases @ pose_tensor
    return pose_tensor


def _forward_recurrent_predictions(net, views, device):
    """Dispatch to the recurrent path that implements the configured update rule."""
    update_type = getattr(net.config, "model_update_type", None)
    if update_type == "ttt3r":
        # TTT3R's forward_recurrent() is inherited from CUT3R and always applies
        # a full state update. Only the official lighter/forward paths apply the
        # attention-derived TTT3R state-update gate.
        predictions, _ = net.forward_recurrent_lighter(
            views, device, ret_state=False
        )
        return predictions
    if update_type == "cut3r":
        predictions, _ = net.forward_recurrent(views, device, ret_state=False)
        return predictions
    raise ValueError(
        "CUT3R/TTT3R recurrent inference requires model_update_type to be "
        f"'cut3r' or 'ttt3r', got {update_type!r}"
    )


def _infer_pose_only_c2w(filelist, model, hydra_cfg):
    eager = bool(hydra_cfg.get("measure_forward_fps", False))
    reset_interval = _validate_reset_interval(
        hydra_cfg.get("cut3r_ttt3r_pose_reset_interval", 100)
    )
    update_type = str(
        getattr(getattr(model.model, "config", None), "model_update_type", "unknown")
    )
    with torch.no_grad(), pose_only_downstream_head(model.model):
        if eager:
            # Decode, resize and transfer every view before starting the timer.
            # The timed path remains the same recurrent rollout used by normal
            # long-pose inference, including TTT3R's gated state update.
            views = [
                _move_view_to_device(view, hydra_cfg.device)
                for view in prepare_input(
                    filelist,
                    size=model.input_size,
                    device=hydra_cfg.device,
                    reset_interval=reset_interval,
                )
            ]
            reset_flags = [
                bool(torch.as_tensor(view["reset"]).any().item()) for view in views
            ]
            record_model_runtime(
                model,
                input_hw=views[0]["img"].shape[-2:],
                input_storage_dtype="float32",
                forward_compute_dtype="fp32",
                preprocess="official_dust3r_long_edge_512_crop_patch16",
                online_state=f"{update_type}+reset-{reset_interval or 0}",
                forward_frames=len(filelist),
            )
            with time_forward(model, hydra_cfg, num_frames=len(filelist)):
                predictions = _forward_recurrent_predictions(
                    model.model, views, hydra_cfg.device
                )
            poses = _poses_from_predictions(predictions, reset_flags)
            del predictions, views
        else:
            reset_flags = []
            runtime_recorded = False

            def tracked_views():
                nonlocal runtime_recorded
                for view in _iter_prepared_views(
                    filelist,
                    model,
                    hydra_cfg.device,
                    reset_interval=reset_interval,
                ):
                    reset_flags.append(
                        bool(torch.as_tensor(view["reset"]).any().item())
                    )
                    if not runtime_recorded:
                        record_model_runtime(
                            model,
                            input_hw=view["img"].shape[-2:],
                            input_storage_dtype="float32",
                            forward_compute_dtype="fp32",
                            preprocess="official_dust3r_long_edge_512_crop_patch16",
                            online_state=f"{update_type}+reset-{reset_interval or 0}",
                            forward_frames=len(filelist),
                        )
                        runtime_recorded = True
                    yield view

            views = tracked_views()
            with time_forward(model, hydra_cfg, num_frames=len(filelist)):
                predictions = _forward_recurrent_predictions(
                    model.model, views, hydra_cfg.device
                )
            poses = _poses_from_predictions(predictions, reset_flags)
            del predictions, views
    if len(poses) != len(filelist):
        raise RuntimeError(
            "CUT3R/TTT3R reset stitching changed the trajectory length: "
            f"input={len(filelist)}, output={len(poses)}, "
            f"reset_interval={reset_interval}"
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return poses


def closed_form_inverse_se3(c2w: torch.Tensor) -> torch.Tensor:
    c2w = torch.as_tensor(c2w)
    rotation = c2w[..., :3, :3]
    translation = c2w[..., :3, 3:4]
    inverse = torch.cat([rotation.transpose(-1, -2), -rotation.transpose(-1, -2) @ translation], dim=-1)
    return inverse


def matrix_cumprod(matrices: torch.Tensor) -> torch.Tensor:
    """Left-to-right homogeneous transform product, independent of fork helpers."""
    running = torch.eye(4, dtype=matrices.dtype, device=matrices.device)
    products = []
    for matrix in matrices:
        running = running @ matrix
        products.append(running)
    return torch.stack(products)


def prepare_input(
    img_paths, size, device, raymaps=None, raymap_mask=None, revisit=1, update=True, reset_interval=None
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Exact official CUT3R/TTT3R preprocessing: long edge 512, bicubic,
    # crop to multiples of 16. Do not route through the VGGT 518 loader.
    from dust3r.utils.image import load_images

    images = load_images(img_paths, size=int(size), verbose=False)
    reset_interval = _validate_reset_interval(reset_interval)
    img_mask = [True] * len(img_paths)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(
                    reset_interval is not None and (i + 1) % reset_interval == 0
                ).unsqueeze(0),
            }
            views.append(view)
            if bool(view["reset"].item()):
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor(
                    reset_interval is not None and (i + 1) % reset_interval == 0
                ).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if bool(view["reset"].item()):
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(outputs, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.geometry import geotrf

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0).detach().cpu().bool()
    shifted_reset_mask = torch.cat([reset_mask.new_zeros(1), reset_mask[:-1]], dim=0)

    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # Convert sequence_scale list
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # if os.path.exists(os.path.join(outdir, "depth")):
    #     shutil.rmtree(os.path.join(outdir, "depth"))
    # if os.path.exists(os.path.join(outdir, "conf")):
    #     shutil.rmtree(os.path.join(outdir, "conf"))
    # if os.path.exists(os.path.join(outdir, "color")):
    #     shutil.rmtree(os.path.join(outdir, "color"))
    # if os.path.exists(os.path.join(outdir, "camera")):
    #     shutil.rmtree(os.path.join(outdir, "camera"))
    # os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    # for f_id in range(len(pts3ds_self)):
    #     depth = depths_tosave[f_id].cpu().numpy()
    #     conf = conf_self_tosave[f_id].cpu().numpy()
    #     color = colors_tosave[f_id].cpu().numpy()
    #     c2w = cam2world_tosave[f_id].cpu().numpy()
    #     intrins = intrinsics_tosave[f_id].cpu().numpy()
    #     np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
    #     np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
    #     iio.imwrite(
    #         os.path.join(outdir, "color", f"{f_id:06d}.png"),
    #         (color * 255).astype(np.uint8),
    #     )
    #     np.savez(
    #         os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
    #         pose=c2w,
    #         intrinsics=intrins,
    #     )

    # # convert_scene_output_to_glb(outdir, (colors_tosave * 255).to(torch.uint8), pts3ds_other_tosave, conf_other_tosave > 1, focal, cam2world_tosave, as_pointcloud=True)
    return pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave


def prepare_pointcloud_output_chunked(
    outputs,
    target_hw,
    chunk_size=16,
    nearest=True,
    output_indices=None,
):
    """Convert official CUT3R outputs to the GT grid with bounded GPU memory.

    The generic ``prepare_output`` materializes several full-sequence tensors
    (self/world points, confidence, colors, and depth) and the old caller then
    resized the complete sequence on CUDA. Point-cloud evaluation only needs
    world points, local depth validity, and camera poses, so process those in
    small CPU chunks and release each prediction as soon as it is consumed.
    """
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import geotrf
    from mv_recon.pc_infer_utils import resize_map_to_hw

    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    predictions = outputs["pred"]
    views = outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in views], 0).detach().cpu().bool()
    shifted_reset_mask = torch.cat([reset_mask.new_zeros(1), reset_mask[:-1]])
    keep_indices = [i for i, drop in enumerate(shifted_reset_mask) if not bool(drop)]
    kept_reset_mask = reset_mask[~shifted_reset_mask]

    poses = []
    for index in keep_indices:
        pose = pose_encoding_to_camera(predictions[index]["camera_pose"].clone())
        poses.append(pose.detach().cpu().float())
    if not poses:
        raise ValueError("CUT3R/TTT3R returned no point-cloud predictions")

    if kept_reset_mask.any():
        pose_tensor = torch.cat(poses, 0)
        identity = torch.eye(4, dtype=pose_tensor.dtype)
        reset_poses = torch.where(
            kept_reset_mask[:, None, None], pose_tensor, identity
        )
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity[None], cumulative_bases[:-1]], dim=0)
        pose_tensor = shifted_bases @ pose_tensor
        poses = list(pose_tensor[:, None].unbind(0))

    full_cam2world = torch.cat(poses, 0).numpy()
    if output_indices is None:
        selected_positions = list(range(len(keep_indices)))
    else:
        selected_positions = [int(value) for value in output_indices]
        if (
            not selected_positions
            or min(selected_positions) < 0
            or max(selected_positions) >= len(keep_indices)
            or any(
                right <= left
                for left, right in zip(selected_positions, selected_positions[1:])
            )
        ):
            raise ValueError(
                f"Invalid dense output indices for {len(keep_indices)} frames: "
                f"{selected_positions}"
            )

    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    num_frames = len(selected_positions)
    points_out = np.empty((num_frames, target_h, target_w, 3), dtype=np.float32)
    mask_out = np.empty((num_frames, target_h, target_w), dtype=bool)

    for start in range(0, num_frames, int(chunk_size)):
        stop = min(start + int(chunk_size), num_frames)
        world_chunk = []
        depth_chunk = []
        for out_index in range(start, stop):
            selected_position = selected_positions[out_index]
            pred_index = keep_indices[selected_position]
            local_points = (
                predictions[pred_index]["pts3d_in_self_view"]
                .detach()
                .cpu()
                .float()[0]
            )
            world_points = geotrf(
                poses[selected_position], local_points.unsqueeze(0)
            )[0]
            world_chunk.append(world_points)
            depth_chunk.append(local_points[..., 2])

            # These objects own the large per-frame CUDA tensors. Replacing
            # list entries lets the caching allocator reuse memory immediately.
            predictions[pred_index] = None
            views[pred_index] = None

        world = torch.stack(world_chunk).permute(0, 3, 1, 2)
        depth = torch.stack(depth_chunk)
        world = resize_map_to_hw(world, (target_h, target_w), nearest=nearest)
        depth = resize_map_to_hw(depth, (target_h, target_w), nearest=nearest)
        points_out[start:stop] = world.permute(0, 2, 3, 1).numpy()
        mask_out[start:stop] = depth.numpy() > 1e-4
        del world_chunk, depth_chunk, world, depth

    for index in range(len(predictions)):
        predictions[index] = None
        views[index] = None
    return points_out, full_cam2world, mask_out


def infer_videodepth(filelist: List[str], model: TTT3R, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    images = prepare_input(
        filelist, size=model.input_size, device=hydra_cfg.device
    )
    
    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            outputs, state_args = model(images)
    end = time.time()

    pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave = prepare_output(
        outputs, 1, True
    )
    depth_map = depths_tosave.cpu()  # depth_map (N, H, W)
    depth_conf = conf_other_tosave.cpu()        # depth_conf (N, H, W)
    return  end - start, depth_map, depth_conf



def infer_monodepth(file: str, model: TTT3R, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    images = prepare_input(
        [file], size=model.input_size, device=hydra_cfg.device
    )
    

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            outputs, state_args = model(images)


    pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave = prepare_output(
        outputs, 1, True
    )
    depth_map = depths_tosave.cpu()  # depth_map (N, H, W)
    depth_conf = conf_other_tosave.cpu()        # depth_conf (N, H, W)
    return  depth_map[0].detach()  



def infer_mv_pointclouds(filelist: List[str], model: TTT3R, hydra_cfg: DictConfig, data_size: Tuple[int, int]):
    # images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    configured = getattr(hydra_cfg, "mv_recon_output_indices", None)
    dense_indices = (
        list(range(len(filelist)))
        if configured is None
        else [int(index) for index in configured]
    )
    update_type = str(getattr(model.model.config, "model_update_type", "unknown"))
    runtime_recorded = False

    def tracked_views():
        nonlocal runtime_recorded
        for view in _iter_prepared_views(filelist, model, hydra_cfg.device):
            if not runtime_recorded:
                record_model_runtime(
                    model,
                    input_hw=view["img"].shape[-2:],
                    input_storage_dtype="float32",
                    forward_compute_dtype="fp32",
                    preprocess="official_dust3r_long_edge_512_crop_patch16",
                    online_state=f"{update_type}+no-reset",
                    forward_frames=len(filelist),
                )
                runtime_recorded = True
            yield view

    views = tracked_views()
    with torch.no_grad():
        # CUT3R's released mv_recon evaluator explicitly disables autocast.
        # TTT3R uses the same frozen checkpoint and official FP32 recurrent path.
        with torch.amp.autocast(
            device_type=str(hydra_cfg.device).split(":")[0], enabled=False
        ):
            with selective_dense_downstream_head(model.model, dense_indices):
                predictions = _forward_recurrent_predictions(
                    model.model, views, hydra_cfg.device
                )
    prepared_views = [
        {"reset": torch.tensor([False], dtype=torch.bool)}
        for _ in range(len(predictions))
    ]
    outputs = {"pred": predictions, "views": prepared_views}
    del views
    torch.cuda.empty_cache()
    resize_chunk_size = int(getattr(hydra_cfg, "cut3r_resize_chunk_size", 16))
    from mv_recon.pc_infer_utils import nearest_depth_to_gt_enabled

    result = prepare_pointcloud_output_chunked(
        outputs,
        data_size,
        chunk_size=resize_chunk_size,
        nearest=nearest_depth_to_gt_enabled(hydra_cfg),
        output_indices=dense_indices,
    )
    del outputs
    torch.cuda.empty_cache()
    return result


def infer_cameras_c2w(filelist: List[str], model: TTT3R, hydra_cfg: DictConfig):
    return _infer_pose_only_c2w(filelist, model, hydra_cfg)[:, :3, :], None


def infer_cameras_w2c(filelist: List[str], model: TTT3R, hydra_cfg: DictConfig):
    c2w = _infer_pose_only_c2w(filelist, model, hydra_cfg)
    return closed_form_inverse_se3(c2w[:, :3, :]), None

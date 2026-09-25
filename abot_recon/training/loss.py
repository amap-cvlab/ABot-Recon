from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..modeling.pi3.utils.geometry import depth_edge


# Public labels for the active CUT3R entries in the development normal-loss list.
_NORMAL_DATASETS = frozenset({"tartanground", "tartanair", "pointodyssey", "scannet", "vkitti2"})


def se3_inverse(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3:]
    output = torch.zeros_like(transform)
    output[..., :3, :3] = rotation.transpose(-1, -2)
    output[..., :3, 3:] = -rotation.transpose(-1, -2) @ translation
    output[..., 3, 3] = 1
    return output


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "...ij,...hwj->...hwi", transform[..., :3, :3], points
    ) + transform[..., None, None, :3, 3]


def rotation_angle(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    relative = left.transpose(-1, -2).float() @ right.float()
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(
        -1 + 1e-6, 1 - 1e-6
    )
    return torch.acos(cosine)


def _sample_valid(values: torch.Tensor, mask: torch.Tensor, count: int) -> torch.Tensor:
    """Original ROE behavior: nearest resample valid points to a fixed count."""
    output = []
    for sample, sample_mask in zip(values, mask, strict=True):
        valid = sample[sample_mask]
        if valid.numel() == 0:
            valid = torch.ones((count, sample.shape[-1]), device=sample.device)
        else:
            valid = F.interpolate(
                valid.transpose(0, 1)[None], size=count, mode="nearest"
            )[0].transpose(0, 1)
        output.append(valid)
    return torch.stack(output)


def align_points_scale(
    predicted: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Exact untruncated weighted-L1 scale solver used by the original loss."""
    source = predicted.flatten(-2)
    destination = target.flatten(-2)
    weights = weight[..., None].expand_as(predicted).flatten(-2)
    sign = source.sign()
    source = source * sign
    destination = destination * sign
    ratios = destination / source.clamp_min(1e-7)
    order = ratios.argsort(dim=-1)
    sorted_ratios = ratios.gather(-1, order)
    sorted_weight = (source * weights).gather(-1, order)
    derivative = 2 * sorted_weight.cumsum(-1) - sorted_weight.sum(-1, keepdim=True)
    index = torch.searchsorted(
        derivative, torch.zeros_like(derivative[..., :1]), side="left"
    ).squeeze(-1).clamp_max(ratios.shape[-1] - 1)
    return sorted_ratios.gather(-1, index[:, None]).squeeze(-1).abs()


def _camera_normalizer(poses: torch.Tensor) -> torch.Tensor:
    """Mean translation norm in the first camera's frame; keep its gradient."""
    relative = torch.einsum("bij,bnjk->bnik", se3_inverse(poses[:, 0]), poses)
    return relative[..., :3, 3].norm(dim=-1).mean(dim=-1).clamp_min(1e-8)


@torch.no_grad()
def align_camera_scale(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Development camera-only alignment: robust adjacent-motion length ratio.

    Reject log-ratio outliers using median/MAD. Fewer than two usable pairs
    (including stationary trajectories) leave the scale at one.
    """
    scale = predicted.new_ones(predicted.shape[0])
    if predicted.shape[1] < 2:
        return scale
    pred_relative = se3_inverse(predicted[:, :-1]) @ predicted[:, 1:]
    target_relative = se3_inverse(target[:, :-1]) @ target[:, 1:]
    pred_length = pred_relative[..., :3, 3].norm(dim=-1)
    target_length = target_relative[..., :3, 3].norm(dim=-1)
    valid = (
        torch.isfinite(pred_length) & torch.isfinite(target_length)
        & (pred_length > 1e-8) & (target_length > 1e-8)
    )
    log_ratio = target_length.clamp_min(1e-8).log() - pred_length.clamp_min(1e-8).log()
    for index in range(predicted.shape[0]):
        values = log_ratio[index][valid[index]]
        if values.numel() < 2:
            continue
        median = values.median()
        mad = (values - median).abs().median()
        threshold = (3.0 * 1.4826 * mad).clamp_min(math.log(3.0))
        kept = values[(values - median).abs() <= threshold]
        if kept.numel() >= 2:
            scale[index] = kept.mean().exp().clamp(1e-8, 1e8)
    return scale


def prepare_ground_truth(views: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    points = torch.stack([view["pts3d"] for view in views], dim=1)
    masks = torch.stack([view["valid_mask"] for view in views], dim=1).bool()
    poses = torch.stack([view["camera_pose"] for view in views], dim=1)
    world_to_first = se3_inverse(poses[:, 0])
    points = transform_points(world_to_first[:, None], points)
    poses = world_to_first[:, None] @ poses
    batch, frames = points.shape[:2]
    # camera_only is a sequence-level flag. collate_views supplies a list of bools.
    camera_only = torch.as_tensor(
        views[0].get("camera_only", False), device=poses.device, dtype=torch.bool
    )
    camera_only = camera_only.expand(batch) if camera_only.ndim == 0 else camera_only.reshape(-1)
    if camera_only.numel() != batch:
        raise ValueError("camera_only must have one flag per sequence in the batch")
    point_mask = masks & ~camera_only[:, None, None, None]
    valid_points = points.clone()
    valid_points[~point_mask] = 0
    normalizer = valid_points.reshape(batch, frames, -1, 3).norm(dim=-1).sum((1, 2))
    valid_count = point_mask.sum((1, 2, 3))
    normalizer = normalizer / valid_count.clamp_min(1)
    normalizer = torch.where(
        valid_count > 0, normalizer.clamp_min(1e-8), torch.ones_like(normalizer)
    )
    if camera_only.any():
        normalizer[camera_only] = poses[camera_only, ..., :3, 3].norm(dim=-1).mean(-1).clamp_min(1e-8)
    points = points / normalizer[:, None, None, None, None]
    poses = poses.clone()
    poses[..., :3, 3] /= normalizer[:, None, None]
    local = transform_points(se3_inverse(poses), points)
    names = views[0]["dataset"]
    if isinstance(names, str):
        names = [names] * batch
    return {
        "local_points": local,
        "valid_masks": masks,
        "camera_poses": poses,
        "dataset_names": list(names),
        "camera_only": camera_only,
    }


class PointLoss(nn.Module):
    def __init__(
        self,
        align_resolution: int = 4096,
        *,
        train_confidence: bool = False,
        confidence_weight: float = 0.05,
        confidence_error_threshold: float = 0.02,
        confidence_invalid_as_zero: bool = True,
    ) -> None:
        super().__init__()
        self.align_resolution = align_resolution
        self.train_confidence = bool(train_confidence)
        self.confidence_weight = float(confidence_weight)
        self.confidence_error_threshold = float(confidence_error_threshold)
        self.confidence_invalid_as_zero = bool(confidence_invalid_as_zero)

    @staticmethod
    def normal_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        valid = mask & ~depth_edge(target[..., 2], rtol=0.03)
        pred = (
            predicted[..., :-1, :-1, :], predicted[..., :-1, 1:, :],
            predicted[..., 1:, :-1, :], predicted[..., 1:, 1:, :],
        )
        gt = (
            target[..., :-1, :-1, :], target[..., :-1, 1:, :],
            target[..., 1:, :-1, :], target[..., 1:, 1:, :],
        )
        keep = (
            valid[..., :-1, :-1], valid[..., :-1, 1:],
            valid[..., 1:, :-1], valid[..., 1:, 1:],
        )

        minimum, maximum, beta = map(math.radians, (1.0, 90.0, 3.0))
        losses = []
        for first, second, origin in ((1, 2, 3), (0, 3, 1), (2, 1, 0), (3, 0, 2)):
            pred_normal = torch.cross(
                pred[first] - pred[origin], pred[second] - pred[origin], dim=-1
            )
            gt_normal = torch.cross(
                gt[first] - gt[origin], gt[second] - gt[origin], dim=-1
            )
            angle = torch.atan2(
                torch.cross(pred_normal, gt_normal, dim=-1).norm(dim=-1) + 1e-12,
                (pred_normal * gt_normal).sum(dim=-1),
            ).clamp(minimum, maximum)
            smooth = torch.where(
                angle < beta, 0.5 * angle.square() / beta, angle - 0.5 * beta
            )
            losses.append((keep[first] & keep[second] & keep[origin]) * smooth)

        return sum(losses).mean() / (4 * max(predicted.shape[-3:-1]))

    def forward(self, prediction: dict, ground_truth: dict):
        predicted = prediction["local_points"]
        target = ground_truth["local_points"]
        mask = ground_truth["valid_masks"]
        batch, frames = predicted.shape[:2]
        camera_only = ground_truth.get("camera_only", mask.new_zeros(batch))
        mask = mask & ~camera_only[:, None, None, None]
        # Unsupervised depth may be arbitrary, including non-finite placeholders.
        # Remove it before division so it cannot poison the camera gradient.
        predicted = predicted.masked_fill(camera_only[:, None, None, None, None], 0)
        pred_norm = predicted.clone()
        pred_norm[~mask] = 0
        norm = pred_norm.reshape(batch, frames, -1, 3).norm(dim=-1).sum((1, 2))
        valid_count = mask.sum((1, 2, 3))
        has_points = valid_count > 0
        norm = (norm / valid_count.clamp_min(1)).clamp_min(1e-8)
        norm = torch.where(has_points, norm, torch.ones_like(norm))
        if camera_only.any():
            norm[camera_only] = _camera_normalizer(prediction["camera_poses"][camera_only])
        predicted = predicted / norm[:, None, None, None, None]
        prediction["local_points"] = predicted
        prediction["camera_poses"] = prediction["camera_poses"].clone()
        prediction["camera_poses"][..., :3, 3] /= norm[:, None, None]

        depth_weight = target[..., 2].clamp_min(1e-6)
        mean_depth = (depth_weight * mask).sum((-2, -1), keepdim=True) / mask.sum(
            (-2, -1), keepdim=True
        ).clamp_min(1)
        depth_weight = 1 / depth_weight.clamp_min(0.1 * mean_depth)
        with torch.no_grad():
            scale = predicted.new_ones(batch)
            if has_points.any():
                valid_mask = mask[has_points]
                sampled_pred = _sample_valid(predicted[has_points], valid_mask, self.align_resolution)
                sampled_target = _sample_valid(target[has_points], valid_mask, self.align_resolution)
                sampled_weight = _sample_valid(depth_weight[has_points, ..., None], valid_mask, self.align_resolution)[..., 0]
                scale[has_points] = align_points_scale(sampled_pred, sampled_target, sampled_weight)
            if camera_only.any():
                scale[camera_only] = align_camera_scale(
                    prediction["camera_poses"][camera_only], ground_truth["camera_poses"][camera_only]
                ).to(scale)
        aligned = predicted * scale[:, None, None, None, None]
        # Index before residual arithmetic: an invalid sample may contain NaNs.
        residual = (aligned[mask].float() - target[mask].float()).abs() * depth_weight[mask][..., None]
        point = residual.mean() if residual.numel() else predicted.reshape(-1)[:0].sum()
        normal_ids = [
            i for i, name in enumerate(ground_truth["dataset_names"])
            if name in _NORMAL_DATASETS and has_points[i]
        ]
        normal = (
            self.normal_loss(aligned[normal_ids], target[normal_ids], mask[normal_ids])
            if normal_ids
            else point * 0
        )
        confidence = point * 0
        confidence_valid = point * 0
        confidence_invalid = point * 0
        if self.train_confidence:
            logits = prediction.get("conf")
            if logits is None:
                raise RuntimeError(
                    "enable_confidence=true requires the model to return confidence logits"
                )
            if logits.shape[-1:] == (1,):
                logits = logits[..., 0]
            if logits.shape != mask.shape:
                raise ValueError(
                    f"confidence shape {tuple(logits.shape)} does not match mask {tuple(mask.shape)}"
                )
            # Match the source objective: confidence targets are derived from the
            # same inverse-depth-weighted point residual used by the point loss.
            pixel_error = residual.detach().mean(-1)
            confidence_target = (pixel_error < self.confidence_error_threshold).to(logits)
            if mask.any():
                confidence_valid = F.binary_cross_entropy_with_logits(
                    logits[mask].float(), confidence_target.float()
                )
            else:
                confidence_valid = logits.reshape(-1)[:0].sum()
            confidence = confidence_valid
            if self.confidence_invalid_as_zero:
                invalid = ~mask & ~camera_only[:, None, None, None]
                if invalid.any():
                    confidence_invalid = F.binary_cross_entropy_with_logits(
                        logits[invalid].float(), torch.zeros_like(logits[invalid]).float()
                    )
                else:
                    confidence_invalid = logits.reshape(-1)[:0].sum()
                confidence = confidence + confidence_invalid
        total = point + normal + self.confidence_weight * confidence
        details = {
            "local_pts_loss": point,
            "normal_loss": normal,
        }
        if self.train_confidence:
            details.update(
                confidence_loss=confidence,
                confidence_valid_loss=confidence_valid,
                confidence_invalid_loss=confidence_invalid,
            )
        return total, details, scale


@dataclass
class CameraLossConfig:
    alpha_translation: float = 100.0
    alpha_rotation: float = 1.0
    max_pair_distance: int = 11
    pair_mode: str = "causal_upper"
    rotation_gap_weight: str = "pow0.75"
    translation_gap_weight: str = "none"
    alpha_corr_magnitude: float = 1e-3
    alpha_corr_smooth: float = 1e-3


class CameraLoss(nn.Module):
    def __init__(self, config: CameraLossConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _weight(gaps: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "none":
            return torch.ones_like(gaps, dtype=torch.float32)
        power = {"sqrt": 0.5, "pow0.75": 0.75, "linear": 1.0}[mode]
        weight = gaps.float().pow(power)
        return weight / weight.mean().clamp_min(1e-8)

    def _pairs(self, poses: torch.Tensor):
        frames = poses.shape[1]
        if self.config.pair_mode == "causal_upper":
            i, j = torch.triu_indices(frames, frames, 1, device=poses.device)
        elif self.config.pair_mode == "dense":
            grid_i, grid_j = torch.meshgrid(
                torch.arange(frames, device=poses.device),
                torch.arange(frames, device=poses.device),
                indexing="ij",
            )
            keep = grid_i != grid_j
            i, j = grid_i[keep], grid_j[keep]
        else:
            raise ValueError(f"Unsupported camera pair mode: {self.config.pair_mode}")
        if self.config.max_pair_distance > 0:
            gap = (j - i).abs() if self.config.pair_mode == "dense" else j - i
            keep = gap <= self.config.max_pair_distance
            i, j = i[keep], j[keep]
        inverse = se3_inverse(poses[:, i])
        return inverse @ poses[:, j], i, j

    @staticmethod
    def _correction(prediction: dict) -> torch.Tensor | None:
        for key in ("camera_state_metrics", "camera_state"):
            state = prediction.get(key)
            residual = state.get("rotation_residual") if isinstance(state, dict) else None
            if residual is not None and residual.numel() > 0:
                return residual
        return None

    def _corr_auxiliary(self, prediction: dict):
        corrected = prediction["camera_poses"]
        zero = corrected.sum() * 0
        residual = self._correction(prediction)
        if residual is None:
            magnitude = smooth = zero
        else:
            residual = residual.float()
            magnitude = residual.square().sum(-1).mean()
            smooth = (
                (residual[:, 1:] - residual[:, :-1]).square().sum(-1).mean()
                if residual.shape[1] > 1
                else zero
            )
        loss = (
            self.config.alpha_corr_magnitude * magnitude
            + self.config.alpha_corr_smooth * smooth
        )
        details = {
            "rot_corr_mag_loss": magnitude,
            "rot_corr_smooth_loss": smooth,
        }
        if residual is not None:
            degrees = residual.detach().norm(dim=-1).flatten() * 180 / math.pi
            details.update(rot_corr_bias_deg_mean=degrees.mean(), rot_corr_bias_deg_p90=torch.quantile(degrees, 0.9), rot_corr_bias_deg_max=degrees.max())
        return loss, details

    def forward(self, prediction: dict, ground_truth: dict, scale: torch.Tensor):
        predicted = prediction["camera_poses"].clone()
        predicted[..., :3, 3] *= scale[:, None, None]
        target = ground_truth["camera_poses"]
        pred_relative, i, j = self._pairs(predicted)
        gt_relative, _, _ = self._pairs(target)
        gaps = (j - i).abs() if self.config.pair_mode == "dense" else j - i
        if not gaps.numel():
            zero = predicted.reshape(-1)[:0].sum()
            return zero, {
                "trans_loss": zero, "rot_loss": zero,
                "rot_corr_mag_loss": zero, "rot_corr_smooth_loss": zero,
                "camera_pairs": torch.zeros((), dtype=torch.long, device=target.device),
            }
        trans_residual = F.huber_loss(pred_relative[..., :3, 3].float(), gt_relative[..., :3, 3].float(), reduction="none", delta=0.1).mean(-1)
        translation = (trans_residual * self._weight(gaps, self.config.translation_gap_weight)).mean()
        angle = rotation_angle(pred_relative[..., :3, :3], gt_relative[..., :3, :3])
        rotation = (angle * self._weight(gaps, self.config.rotation_gap_weight)).mean()
        auxiliary, details = self._corr_auxiliary(prediction)
        total = self.config.alpha_translation * translation + self.config.alpha_rotation * rotation + auxiliary
        details.update(trans_loss=translation, rot_loss=rotation, camera_pairs=torch.as_tensor(gaps.numel(), device=target.device))
        return total, details


class Pi3Loss(nn.Module):
    def __init__(
        self,
        camera: CameraLossConfig | None = None,
        camera_weight: float = 0.1,
        *,
        train_confidence: bool = False,
        confidence_weight: float = 0.05,
        confidence_error_threshold: float = 0.02,
        confidence_invalid_as_zero: bool = True,
    ) -> None:
        super().__init__()
        self.point_loss = PointLoss(
            train_confidence=train_confidence,
            confidence_weight=confidence_weight,
            confidence_error_threshold=confidence_error_threshold,
            confidence_invalid_as_zero=confidence_invalid_as_zero,
        )
        self.camera_loss = CameraLoss(camera or CameraLossConfig())
        self.camera_weight = camera_weight

    def forward(self, prediction: dict, views: list[dict]):
        ground_truth = prepare_ground_truth(views)
        point, details, scale = self.point_loss(prediction, ground_truth)
        camera, camera_details = self.camera_loss(prediction, ground_truth, scale)
        total = point + self.camera_weight * camera
        details.update(camera_details)
        details.update(point_loss=point, camera_loss=camera, loss=total)
        return total, details


ReconstructionLoss = Pi3Loss

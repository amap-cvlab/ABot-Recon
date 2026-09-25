from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PoseState = Dict[str, torch.Tensor]


class TemporalRotationRefiner(nn.Module):
    """Causal rotation residual from adjacent visual-motion descriptors."""

    def __init__(
        self,
        *,
        desc_dim: int,
        frame_dim: int,
        hidden_dim: int,
        kernel_size: int,
        max_rot_deg: float,
        num_heads: int = 8,
        use_age_embed: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")

        self.kernel_size = int(kernel_size)
        self.hidden_dim = int(hidden_dim)
        self.max_rad = float(max_rot_deg) * math.pi / 180.0
        if self.max_rad < 0:
            raise ValueError("max_rot_deg must be non-negative")

        self.desc_proj = nn.Sequential(
            nn.LayerNorm(int(desc_dim) * 4),
            nn.Linear(int(desc_dim) * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.frame_query_proj = nn.Sequential(
            nn.LayerNorm(int(frame_dim) * 4),
            nn.Linear(int(frame_dim) * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.frame_proj = nn.Linear(int(frame_dim), self.hidden_dim)
        self.frame_role_embed = nn.Parameter(torch.empty(2, self.hidden_dim))
        self.frame_attn = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_heads),
            dropout=0.0,
            batch_first=True,
        )
        self.frame_dropout = nn.Dropout(0.0)
        self.frame_norm = nn.LayerNorm(self.hidden_dim)
        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.age_embed = (
            nn.Embedding(self.kernel_size, self.hidden_dim) if use_age_embed else None
        )
        self.conv = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            self.kernel_size,
            groups=self.hidden_dim,
        )
        self.gate = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            self.kernel_size,
            groups=self.hidden_dim,
        )
        self.out = nn.Linear(self.hidden_dim, 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for sequence in (self.desc_proj, self.frame_query_proj, self.fuse_proj):
            for module in sequence:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.frame_proj.weight)
        nn.init.zeros_(self.frame_proj.bias)
        nn.init.normal_(self.frame_role_embed, std=0.02)
        nn.init.xavier_uniform_(self.frame_attn.in_proj_weight)
        if self.frame_attn.in_proj_bias is not None:
            nn.init.zeros_(self.frame_attn.in_proj_bias)
        nn.init.xavier_uniform_(self.frame_attn.out_proj.weight)
        nn.init.zeros_(self.frame_attn.out_proj.bias)
        if self.age_embed is not None:
            nn.init.normal_(self.age_embed.weight, std=0.02)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv.bias)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        prev_desc: torch.Tensor,
        curr_desc: torch.Tensor,
        prev_frame_tokens: torch.Tensor,
        curr_frame_tokens: torch.Tensor,
        feature_buffer: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prev = prev_desc.float()
        curr = curr_desc.float()
        desc_pair = torch.cat([prev, curr, curr - prev, curr * prev], dim=-1)
        desc_feature = self.desc_proj(desc_pair)

        prev_mean = prev_frame_tokens.float().mean(dim=1)
        curr_mean = curr_frame_tokens.float().mean(dim=1)
        frame_pair = torch.cat(
            [prev_mean, curr_mean, curr_mean - prev_mean, curr_mean * prev_mean],
            dim=-1,
        )
        frame_query = self.frame_query_proj(frame_pair)[:, None, :]

        prev_len = prev_frame_tokens.shape[1]
        curr_len = curr_frame_tokens.shape[1]
        memory = self.frame_proj(
            torch.cat([prev_frame_tokens, curr_frame_tokens], dim=1).float()
        )
        role = self.frame_role_embed.to(device=memory.device, dtype=memory.dtype)
        memory = memory.clone()
        memory[:, :prev_len] = memory[:, :prev_len] + role[0]
        memory[:, prev_len : prev_len + curr_len] = (
            memory[:, prev_len : prev_len + curr_len] + role[1]
        )

        frame_output, _ = self.frame_attn(
            frame_query, memory, memory, need_weights=False
        )
        temporal_context = self.frame_norm(
            frame_query + self.frame_dropout(frame_output)
        ).squeeze(1)
        fused = self.fuse_proj(
            torch.cat([desc_feature, temporal_context], dim=-1)
        ).to(dtype=curr_desc.dtype)

        if feature_buffer is None:
            feature_buffer = fused[:, None]
        else:
            feature_buffer = torch.cat([feature_buffer, fused[:, None]], dim=1)
        feature_buffer = feature_buffer[:, -self.kernel_size :]

        window = feature_buffer
        pad_len = self.kernel_size - window.shape[1]
        if pad_len > 0:
            padding = window.new_zeros(window.shape[0], pad_len, window.shape[2])
            window = torch.cat([padding, window], dim=1)
        if self.age_embed is not None:
            age_ids = torch.arange(
                self.kernel_size - 1, -1, -1, device=window.device
            )
            age_embedding = self.age_embed(age_ids).to(dtype=window.dtype)
            valid_mask = window.new_zeros(self.kernel_size)
            valid_mask[pad_len:] = 1
            window = window + age_embedding.unsqueeze(0) * valid_mask.view(
                1, self.kernel_size, 1
            )

        temporal = window.transpose(1, 2).float()
        hidden = (
            self.conv(temporal) * torch.sigmoid(self.gate(temporal))
        ).squeeze(-1)
        residual = self.max_rad * torch.tanh(self.out(hidden))
        return residual.to(dtype=curr_desc.dtype), feature_buffer


class AdjacentPoseHead(nn.Module):
    """Predict adjacent SE(3) increments and compose them into a trajectory."""

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 512,
        pair_hidden_dim: Optional[int] = None,
        num_pose_tokens: int = 5,
        rotation_format: str = "quat",
        init_std: float = 1.0e-4,
        translation_param: str = "vector",
        rot_correction_mode: str = "temporal_rotation_refinement",
        rot_correction_kernel: int = 10,
        rot_correction_hidden_dim: Optional[int] = None,
        rot_correction_max_deg: float = 2.0,
        rot_correction_use_age_embed: bool = True,
    ) -> None:
        super().__init__()
        if rotation_format != "quat":
            raise ValueError("The released model uses scalar-last quaternion rotation")
        if translation_param != "vector":
            raise ValueError("The released model uses direct translation vectors")
        if rot_correction_mode != "temporal_rotation_refinement":
            raise ValueError("The released model uses temporal rotation refinement")
        if num_pose_tokens <= 0:
            raise ValueError("num_pose_tokens must be positive")

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.pair_hidden_dim = int(pair_hidden_dim or hidden_dim)
        self.num_pose_tokens = int(num_pose_tokens)
        self.init_std = float(init_std)

        self.frame_descriptor = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.pair_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pair_hidden_dim, self.pair_hidden_dim),
            nn.ReLU(),
        )
        self.rot_correction = TemporalRotationRefiner(
            desc_dim=self.hidden_dim,
            frame_dim=self.dim,
            hidden_dim=int(rot_correction_hidden_dim or self.hidden_dim),
            kernel_size=int(rot_correction_kernel),
            max_rot_deg=float(rot_correction_max_deg),
            num_heads=8,
            use_age_embed=bool(rot_correction_use_age_embed),
        )
        self.delta_t_head = nn.Linear(self.pair_hidden_dim, 3)
        self.delta_q_head = nn.Linear(self.pair_hidden_dim, 4)
        self._init_weights()

    def _init_weights(self) -> None:
        for sequence in (self.frame_descriptor, self.pair_mlp):
            for module in sequence:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        self.rot_correction.reset_parameters()
        nn.init.normal_(self.delta_t_head.weight, std=self.init_std)
        nn.init.zeros_(self.delta_t_head.bias)
        nn.init.normal_(self.delta_q_head.weight, std=self.init_std)
        with torch.no_grad():
            self.delta_q_head.bias.zero_()
            self.delta_q_head.bias[-1] = 1.0

    def forward(
        self,
        feat: torch.Tensor,
        *,
        camera_state: Optional[PoseState] = None,
        return_state: bool = False,
    ):
        if feat.ndim != 4:
            raise ValueError(
                "AdjacentPoseHead expects [batch, frames, tokens, channels], "
                f"got {tuple(feat.shape)}"
            )
        if feat.shape[2] <= self.num_pose_tokens:
            raise ValueError("Pose refinement requires image tokens after the pose tokens")

        descriptors = self._describe_frames(feat)
        frame_tokens = feat[:, :, self.num_pose_tokens :]
        poses, state = self._compose(descriptors, frame_tokens, camera_state)
        return (poses, state) if return_state else poses

    def _describe_frames(self, feat: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, channels = feat.shape
        if tokens < self.num_pose_tokens:
            raise ValueError(
                f"Expected at least {self.num_pose_tokens} pose tokens, got {tokens}"
            )
        pose_tokens = feat[:, :, : self.num_pose_tokens]
        descriptors = self.frame_descriptor(
            pose_tokens.reshape(batch * frames, self.num_pose_tokens, channels)
        )
        return descriptors.reshape(
            batch, frames, self.num_pose_tokens, self.hidden_dim
        ).mean(dim=2)

    def _compose(
        self,
        descriptors: torch.Tensor,
        frame_tokens: torch.Tensor,
        state: Optional[PoseState],
    ) -> Tuple[torch.Tensor, PoseState]:
        batch, num_frames, _ = descriptors.shape
        device, dtype = descriptors.device, descriptors.dtype
        identity = torch.eye(4, device=device, dtype=dtype).expand(batch, 4, 4).clone()

        if state is None:
            previous_desc = descriptors[:, 0]
            previous_frame_tokens = frame_tokens[:, 0]
            previous_pose = identity
            previous_index = torch.zeros(batch, device=device, dtype=torch.long)
            feature_buffer = None
            poses = [identity]
            raw_relatives = [identity]
            source_indices = [previous_index]
            start = 1
        else:
            previous_desc = state["previous_descriptor"].to(device=device, dtype=dtype)
            previous_frame_tokens = state["previous_frame_tokens"].to(
                device=device, dtype=dtype
            )
            previous_pose = state["previous_pose"].to(device=device, dtype=dtype)
            previous_index = state["previous_index"].to(device=device, dtype=torch.long)
            feature_buffer = state.get("rotation_feature_buffer")
            if feature_buffer is not None:
                feature_buffer = feature_buffer.to(device=device, dtype=dtype)
            poses = []
            raw_relatives = []
            source_indices = []
            start = 0

        rotation_residuals = []
        for frame_idx in range(start, num_frames):
            current_desc = descriptors[:, frame_idx]
            current_frame_tokens = frame_tokens[:, frame_idx]
            raw_delta = self._predict_delta(previous_desc, current_desc)
            rotation_residual, feature_buffer = self.rot_correction(
                previous_desc,
                current_desc,
                previous_frame_tokens,
                current_frame_tokens,
                feature_buffer,
            )
            corrected_delta = raw_delta.clone()
            correction_matrix = self._rotvec_to_mat(rotation_residual.float()).to(
                dtype=dtype
            )
            corrected_delta[:, :3, :3] = torch.matmul(
                raw_delta[:, :3, :3], correction_matrix
            )
            current_pose = torch.matmul(previous_pose, corrected_delta)

            poses.append(current_pose)
            raw_relatives.append(raw_delta)
            source_indices.append(previous_index)
            rotation_residuals.append(rotation_residual)

            previous_desc = current_desc
            previous_frame_tokens = current_frame_tokens
            previous_pose = current_pose
            previous_index = previous_index + 1

        pose_sequence = torch.stack(poses, dim=1) if poses else identity[:, None]
        new_state: PoseState = {
            "previous_descriptor": previous_desc.detach(),
            "previous_frame_tokens": previous_frame_tokens.detach(),
            "previous_pose": previous_pose.detach(),
            "previous_index": previous_index.detach(),
            "raw_adjacent_rel_poses": torch.stack(raw_relatives, dim=1),
            "source_frame_indices": torch.stack(source_indices, dim=1).detach(),
        }
        if feature_buffer is not None:
            new_state["rotation_feature_buffer"] = feature_buffer.detach()
        if rotation_residuals:
            new_state["rotation_residual"] = torch.stack(rotation_residuals, dim=1)
        return pose_sequence, new_state

    def _predict_delta(
        self, previous_desc: torch.Tensor, current_desc: torch.Tensor
    ) -> torch.Tensor:
        pair = torch.cat(
            [
                previous_desc,
                current_desc,
                current_desc - previous_desc,
                current_desc * previous_desc,
            ],
            dim=-1,
        )
        hidden = self.pair_mlp(pair)
        translation = self.delta_t_head(hidden.float()).to(dtype=current_desc.dtype)
        quaternion = self.delta_q_head(hidden.float())
        rotation = self._quat_to_mat(quaternion.float()).to(dtype=current_desc.dtype)

        delta = torch.zeros(
            (current_desc.shape[0], 4, 4),
            device=current_desc.device,
            dtype=current_desc.dtype,
        )
        delta[:, :3, :3] = rotation
        delta[:, :3, 3] = translation
        delta[:, 3, 3] = 1.0
        return delta

    @staticmethod
    def _rotvec_to_mat(rotvec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        theta2 = rotvec.square().sum(dim=-1, keepdim=True)
        theta2_safe = theta2.clamp_min(eps * eps)
        theta = theta2_safe.sqrt()
        theta4 = theta2.square()
        small = theta2 < (eps * eps)
        a = torch.where(
            small,
            1.0 - theta2 / 6.0 + theta4 / 120.0,
            torch.sin(theta) / theta,
        )
        b = torch.where(
            small,
            0.5 - theta2 / 24.0 + theta4 / 720.0,
            (1.0 - torch.cos(theta)) / theta2_safe,
        )
        x, y, z = rotvec.unbind(-1)
        zeros = torch.zeros_like(x)
        skew = torch.stack(
            [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
        ).reshape(rotvec.shape[:-1] + (3, 3))
        identity = torch.eye(
            3, device=rotvec.device, dtype=rotvec.dtype
        ).expand_as(skew)
        return identity + a[..., None] * skew + b[..., None] * torch.matmul(skew, skew)

    @staticmethod
    def _quat_to_mat(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        quaternion = F.normalize(quaternion, p=2, dim=-1, eps=eps)
        i, j, k, real = torch.unbind(quaternion, -1)
        two_s = 2.0 / (quaternion * quaternion).sum(-1).clamp_min(eps)
        matrix = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * real),
                two_s * (i * k + j * real),
                two_s * (i * j + k * real),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * real),
                two_s * (i * k - j * real),
                two_s * (j * k + i * real),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return matrix.reshape(quaternion.shape[:-1] + (3, 3))

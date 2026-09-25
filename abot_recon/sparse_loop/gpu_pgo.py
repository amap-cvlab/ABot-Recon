# Copyright (c) 2026 ABot-Recon Authors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation


def _matrix_to_se3_data(mat: np.ndarray) -> np.ndarray:
    quat = Rotation.from_matrix(mat[:3, :3]).as_quat()
    return np.concatenate([mat[:3, 3], quat], axis=0).astype(np.float32, copy=False)


def _project_to_rotation(mat: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(mat)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = u @ vt
    return rot.astype(np.float64, copy=False)


def _matrix_to_sim3_data(mat: np.ndarray) -> np.ndarray:
    linear = np.asarray(mat[:3, :3], dtype=np.float64)
    trans = np.asarray(mat[:3, 3], dtype=np.float64)
    scale = float(np.mean(np.linalg.norm(linear, axis=0)))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    rot = _project_to_rotation(linear / scale)
    quat = Rotation.from_matrix(rot).as_quat()
    return np.concatenate([trans, quat, [scale]], axis=0).astype(np.float32, copy=False)


def _matrices_to_se3_data(matrices: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    quaternions = Rotation.from_matrix(values[:, :3, :3]).as_quat()
    return np.concatenate([values[:, :3, 3], quaternions], axis=1).astype(np.float32, copy=False)


def _se3_batch_to_matrix(value: pp.LieTensor) -> np.ndarray:
    return value.matrix().detach().cpu().numpy().astype(np.float64, copy=False)


def _sim3_batch_to_matrix(value: pp.LieTensor) -> np.ndarray:
    data = value.data.detach().cpu().numpy().astype(np.float64, copy=False)
    output = np.repeat(np.eye(4, dtype=np.float64)[None], len(data), axis=0)
    rotations = Rotation.from_quat(data[:, 3:7]).as_matrix()
    output[:, :3, :3] = data[:, 7, None, None] * rotations
    output[:, :3, 3] = data[:, :3]
    return output


def _batch_edge_jacobians(
    func: Callable[[pp.LieTensor, torch.Tensor, torch.Tensor], torch.Tensor],
    constants: pp.LieTensor,
    gi: torch.Tensor,
    gj: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the two non-zero Jacobian blocks for every graph edge."""

    def summed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return func(constants, x, y).sum(dim=0)

    jac_i, jac_j = torch.autograd.functional.jacobian(
        summed,
        (gi, gj),
        vectorize=True,
        create_graph=False,
    )
    return (
        jac_i.permute(1, 0, 2).contiguous(),
        jac_j.permute(1, 0, 2).contiguous(),
    )


def _build_rhs_and_preconditioner(
    jac_i: torch.Tensor,
    jac_j: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    residual: torch.Tensor,
    num_nodes: int,
    free_mask: torch.Tensor,
    damping: float,
    solve_dtype: torch.dtype,
    coarse_group_size: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Build GPU block-Jacobi and optional coarse-grid preconditioners."""
    jac_i = jac_i.to(dtype=solve_dtype)
    jac_j = jac_j.to(dtype=solve_dtype)
    residual = residual.to(dtype=solve_dtype)
    dim = int(residual.shape[1])
    device = residual.device

    rhs = torch.zeros((num_nodes, dim), dtype=solve_dtype, device=device)
    rhs.index_add_(0, edge_src, -torch.einsum("erc,er->ec", jac_i, residual))
    rhs.index_add_(0, edge_dst, -torch.einsum("erc,er->ec", jac_j, residual))

    block_diagonal = torch.zeros((num_nodes, dim, dim), dtype=solve_dtype, device=device)
    block_diagonal.index_add_(0, edge_src, torch.einsum("erc,erd->ecd", jac_i, jac_i))
    block_diagonal.index_add_(0, edge_dst, torch.einsum("erc,erd->ecd", jac_j, jac_j))
    diagonal = torch.diagonal(block_diagonal, dim1=-2, dim2=-1).clone()
    diagonal = diagonal.clamp_min(torch.finfo(solve_dtype).eps)

    system_blocks = block_diagonal + float(damping) * torch.diag_embed(diagonal)
    component_mask = free_mask.bool()
    pair_mask = component_mask.unsqueeze(-1) & component_mask.unsqueeze(-2)
    system_blocks = system_blocks * pair_mask
    system_blocks = system_blocks + torch.diag_embed((~component_mask).to(solve_dtype))
    identity = torch.eye(dim, dtype=solve_dtype, device=device)
    system_blocks[0] = identity
    system_blocks = system_blocks + torch.finfo(solve_dtype).eps * identity.unsqueeze(0)
    inverse_blocks = torch.linalg.inv(system_blocks)
    rhs = rhs * free_mask

    node_groups: Optional[torch.Tensor] = None
    coarse_factor: Optional[torch.Tensor] = None
    max_coarse_groups = max(2, 2048 // dim)
    group_size = max(
        1,
        int(coarse_group_size),
        math.ceil(num_nodes / max_coarse_groups),
    )
    num_groups = (num_nodes + group_size - 1) // group_size
    # Dense coarse solve is deliberately capped. It represents long-wavelength
    # corrections while the matrix-free fine operator remains O(E).
    if num_groups > 1:
        node_groups = torch.div(
            torch.arange(num_nodes, device=device), group_size, rounding_mode="floor"
        )
        src_groups = node_groups[edge_src]
        dst_groups = node_groups[edge_dst]
        coarse_dim = num_groups * dim
        coarse_flat = torch.zeros(coarse_dim * coarse_dim, dtype=solve_dtype, device=device)
        row_component = torch.arange(dim, device=device).view(1, dim, 1)
        col_component = torch.arange(dim, device=device).view(1, 1, dim)

        jac_i_free = jac_i * free_mask[edge_src].unsqueeze(1)
        jac_j_free = jac_j * free_mask[edge_dst].unsqueeze(1)

        def accumulate_blocks(
            row_groups: torch.Tensor,
            col_groups: torch.Tensor,
            blocks: torch.Tensor,
        ) -> None:
            flat_indices = (
                (row_groups[:, None, None] * dim + row_component) * coarse_dim
                + col_groups[:, None, None] * dim
                + col_component
            )
            coarse_flat.index_add_(0, flat_indices.reshape(-1), blocks.reshape(-1))

        accumulate_blocks(
            src_groups, src_groups, torch.einsum("era,erb->eab", jac_i_free, jac_i_free)
        )
        accumulate_blocks(
            dst_groups, dst_groups, torch.einsum("era,erb->eab", jac_j_free, jac_j_free)
        )
        cross = torch.einsum("era,erb->eab", jac_i_free, jac_j_free)
        accumulate_blocks(src_groups, dst_groups, cross)
        accumulate_blocks(dst_groups, src_groups, cross.transpose(-1, -2))

        coarse_matrix = coarse_flat.view(coarse_dim, coarse_dim)
        group_diagonal = torch.zeros((num_groups, dim), dtype=solve_dtype, device=device)
        group_diagonal.index_add_(0, node_groups, diagonal * free_mask)
        coarse_indices = torch.arange(coarse_dim, device=device)
        coarse_matrix[coarse_indices, coarse_indices] += float(damping) * group_diagonal.reshape(-1)
        coarse_matrix = 0.5 * (coarse_matrix + coarse_matrix.T)
        mean_diagonal = torch.diagonal(coarse_matrix).abs().mean().clamp_min(1.0)
        coarse_matrix[coarse_indices, coarse_indices] += (
            64.0 * torch.finfo(solve_dtype).eps * mean_diagonal
        )
        coarse_factor = torch.linalg.cholesky(coarse_matrix)

    return rhs, diagonal, inverse_blocks, node_groups, coarse_factor


def _normal_matvec(
    value: torch.Tensor,
    jac_i: torch.Tensor,
    jac_j: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    diagonal: torch.Tensor,
    damping: float,
    free_mask: torch.Tensor,
) -> torch.Tensor:
    edge_value = torch.einsum("erc,ec->er", jac_i, value[edge_src]) + torch.einsum(
        "erc,ec->er", jac_j, value[edge_dst]
    )
    out = torch.zeros_like(value)
    out.index_add_(0, edge_src, torch.einsum("erc,er->ec", jac_i, edge_value))
    out.index_add_(0, edge_dst, torch.einsum("erc,er->ec", jac_j, edge_value))
    out = out + float(damping) * diagonal * value
    return out * free_mask


def _solve_normal_pcg(
    jac_i: torch.Tensor,
    jac_j: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    residual: torch.Tensor,
    num_nodes: int,
    damping: float,
    free_mask: torch.Tensor,
    *,
    max_iterations: int,
    tolerance: float,
    solve_dtype: torch.dtype,
    check_interval: int = 8,
    coarse_group_size: int = 64,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solve sparse normal equations with matrix-free GPU block-PCG."""

    jac_i = jac_i.to(dtype=solve_dtype)
    jac_j = jac_j.to(dtype=solve_dtype)
    (
        rhs,
        diagonal,
        inverse_blocks,
        node_groups,
        coarse_factor,
    ) = _build_rhs_and_preconditioner(
        jac_i,
        jac_j,
        edge_src,
        edge_dst,
        residual,
        num_nodes,
        free_mask,
        damping,
        solve_dtype,
        coarse_group_size,
    )

    def apply_preconditioner(value: torch.Tensor) -> torch.Tensor:
        output = torch.einsum("nij,nj->ni", inverse_blocks, value)
        if node_groups is not None and coarse_factor is not None:
            num_groups = int(coarse_factor.shape[0]) // int(value.shape[1])
            coarse_rhs = torch.zeros(
                (num_groups, value.shape[1]), dtype=value.dtype, device=value.device
            )
            coarse_rhs.index_add_(0, node_groups, value)
            coarse_solution = torch.cholesky_solve(
                coarse_rhs.reshape(-1, 1), coarse_factor
            ).reshape(num_groups, value.shape[1])
            output = output + coarse_solution[node_groups]
        return output * free_mask

    value = torch.zeros_like(rhs)
    resid = rhs.clone()
    precond_resid = apply_preconditioner(resid)
    direction = precond_resid.clone()
    rz = torch.sum(resid * precond_resid)
    rhs_norm = torch.linalg.vector_norm(rhs)
    rhs_norm_value = float(rhs_norm.item())
    target = max(float(tolerance) * rhs_norm_value, torch.finfo(solve_dtype).eps)
    initial_norm = float(torch.linalg.vector_norm(resid).item())
    converged = initial_norm <= target
    iterations = 0
    termination_reason = "initial_convergence" if converged else "max_iterations"
    check_interval = max(1, int(check_interval))
    scalar_eps = torch.as_tensor(
        torch.finfo(solve_dtype).eps, dtype=solve_dtype, device=residual.device
    )
    step_is_valid = torch.ones((), dtype=torch.bool, device=residual.device)

    for itr in range(max(1, int(max_iterations))):
        if converged:
            break
        mat_direction = _normal_matvec(
            direction,
            jac_i,
            jac_j,
            edge_src,
            edge_dst,
            diagonal,
            damping,
            free_mask,
        )
        denom = torch.sum(direction * mat_direction)
        step_is_valid = torch.isfinite(denom) & torch.isfinite(rz) & (denom > scalar_eps)
        safe_denom = torch.where(step_is_valid, denom, torch.ones_like(denom))
        alpha = torch.where(step_is_valid, rz / safe_denom, torch.zeros_like(rz))
        value = value + alpha * direction
        resid = resid - alpha * mat_direction
        iterations = itr + 1

        next_precond_resid = apply_preconditioner(resid)
        next_rz = torch.sum(resid * next_precond_resid)
        rz_is_valid = torch.isfinite(next_rz) & (torch.abs(rz) > scalar_eps)
        recurrence_is_valid = step_is_valid & rz_is_valid
        safe_rz = torch.where(rz_is_valid, rz, torch.ones_like(rz))
        beta = torch.where(recurrence_is_valid, next_rz / safe_rz, torch.zeros_like(next_rz))
        next_direction = next_precond_resid + beta * direction
        direction = torch.where(recurrence_is_valid, next_direction, torch.zeros_like(direction))
        rz = next_rz

        should_check = iterations % check_interval == 0 or iterations == int(max_iterations)
        if should_check:
            residual_norm = torch.linalg.vector_norm(resid)
            residual_norm_value = float(residual_norm.item())
            if not bool(recurrence_is_valid.item()):
                relative_residual = residual_norm_value / max(rhs_norm_value, 1e-30)
                converged = relative_residual <= max(float(tolerance), 1e-8)
                termination_reason = "roundoff_floor" if converged else "invalid_recurrence"
                break
            if residual_norm_value <= target:
                converged = True
                termination_reason = "tolerance"
                break

    final_norm = float(torch.linalg.vector_norm(resid).item())
    stats = {
        "iterations": int(iterations),
        "converged": bool(converged),
        "rhs_norm": rhs_norm_value,
        "residual_norm": final_norm,
        "relative_residual": float(final_norm / max(rhs_norm_value, 1e-30)),
        "termination_reason": termination_reason,
        "check_interval": int(check_interval),
        "preconditioner": (
            "two_level_block_jacobi" if coarse_factor is not None else "block_jacobi"
        ),
        "coarse_groups": (
            int(coarse_factor.shape[0]) // int(residual.shape[1])
            if coarse_factor is not None
            else 0
        ),
    }
    return value.to(dtype=residual.dtype), stats


def optimize_pose_graph_gpu_sparse(
    keyframe_c2w_init: np.ndarray,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    edge_measurements: Sequence[np.ndarray],
    edge_weights: Sequence[float],
    *,
    model: str = "se3",
    update_mode: str = "all",
    trans_weight: float = 1.0,
    rot_weight: float = 1.0,
    scale_weight: float = 1.0,
    max_iterations: int = 30,
    lambda_init: float = 1e-4,
    device: str = "cuda",
    pcg_max_iterations: int = 256,
    pcg_tolerance: float = 1e-5,
    pcg_check_interval: int = 8,
    coarse_group_size: int = 64,
    solve_dtype: str = "float64",
    outer_relative_tolerance: float = 0.0,
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
    runtime_stats: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Sparse block-Jacobian PGO with matrix-free GPU normal equations."""

    def log(message: str) -> None:
        if verbose:
            if log_fn is not None:
                log_fn(message)
            else:
                print(message, flush=True)

    if len(keyframe_c2w_init) <= 1 or len(edge_measurements) == 0:
        return keyframe_c2w_init.copy()
    if model not in {"se3", "sim3"}:
        raise ValueError(f"Unsupported pose graph model: {model}")

    total_t0 = time.perf_counter()
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"GPU sparse PGO requires an available CUDA device, got {device!r}")

    dim = 7 if model == "sim3" else 6
    valid_modes = (
        {"all", "translation_only", "translation_scale"}
        if model == "sim3"
        else {"all", "translation_only"}
    )
    if update_mode not in valid_modes:
        raise ValueError(
            f"Unsupported update_mode={update_mode!r} for model={model!r}; "
            f"expected one of {sorted(valid_modes)}"
        )

    active_components = np.zeros(dim, dtype=bool)
    if update_mode == "all":
        active_components[:] = True
    elif update_mode == "translation_only":
        active_components[:3] = True
    else:
        active_components[:3] = True
        active_components[6] = True

    if model == "sim3":
        pose_data = np.stack([_matrix_to_sim3_data(pose) for pose in keyframe_c2w_init])
        meas_data = np.stack([_matrix_to_sim3_data(meas) for meas in edge_measurements])
        poses = pp.Sim3(torch.from_numpy(pose_data).to(torch_device))
        constants = pp.Sim3(torch.from_numpy(meas_data).to(torch_device))
    else:
        pose_data = _matrices_to_se3_data(keyframe_c2w_init)
        meas_data = _matrices_to_se3_data(edge_measurements)
        poses = pp.SE3(torch.from_numpy(pose_data).to(torch_device))
        constants = pp.SE3(torch.from_numpy(meas_data).to(torch_device))

    edge_src_tensor = torch.as_tensor(edge_src, dtype=torch.long, device=torch_device)
    edge_dst_tensor = torch.as_tensor(edge_dst, dtype=torch.long, device=torch_device)
    weights = torch.as_tensor(edge_weights, dtype=torch.float32, device=torch_device).view(-1, 1)
    free_mask = (
        torch.as_tensor(active_components, dtype=torch.float64, device=torch_device)
        .view(1, dim)
        .expand(len(keyframe_c2w_init), dim)
        .clone()
    )
    free_mask[0] = 0.0
    linear_dtype = torch.float64 if str(solve_dtype).lower() == "float64" else torch.float32

    inverse_log = poses.Inv().Log().detach()
    damping = float(lambda_init)
    cost_history: list[float] = []
    total_jacobian_sec = 0.0
    total_solve_sec = 0.0
    total_pcg_iterations = 0
    accepted_steps = 0
    rejected_steps = 0
    last_pcg_stats: Dict[str, Any] = {}

    def residual_fn(
        measurement: pp.LieTensor,
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> torch.Tensor:
        residual = (measurement @ pp.Exp(source) @ pp.Exp(destination).Inv()).Log().tensor()
        residual = residual.clone()
        residual[:, :3] *= float(trans_weight)
        residual[:, 3:6] *= float(rot_weight)
        if model == "sim3":
            residual[:, 6:7] *= float(scale_weight)
        return residual

    def evaluate(current: torch.Tensor) -> torch.Tensor:
        return (
            residual_fn(
                constants,
                current[edge_src_tensor],
                current[edge_dst_tensor],
            )
            * weights
        )

    log(
        f"GPU sparse PGO ({model}, {update_mode}): "
        f"{len(keyframe_c2w_init)} nodes, {len(edge_measurements)} edges, "
        f"outer={max_iterations}, pcg={pcg_max_iterations}, "
        f"tol={pcg_tolerance:g}, coarse_group={coarse_group_size}, "
        f"dtype={linear_dtype}, device={torch_device}"
    )

    for outer_iteration in range(max(1, int(max_iterations))):
        residual_unweighted = residual_fn(
            constants,
            inverse_log[edge_src_tensor],
            inverse_log[edge_dst_tensor],
        )

        torch.cuda.synchronize(torch_device)
        jacobian_t0 = time.perf_counter()
        jac_i, jac_j = _batch_edge_jacobians(
            residual_fn,
            constants,
            inverse_log[edge_src_tensor],
            inverse_log[edge_dst_tensor],
        )
        jac_i = jac_i * weights.unsqueeze(-1)
        jac_j = jac_j * weights.unsqueeze(-1)
        residual = residual_unweighted * weights
        torch.cuda.synchronize(torch_device)
        total_jacobian_sec += time.perf_counter() - jacobian_t0

        current_cost = float(residual.square().mean().item())
        cost_history.append(current_cost)

        torch.cuda.synchronize(torch_device)
        solve_t0 = time.perf_counter()
        delta, pcg_stats = _solve_normal_pcg(
            jac_i,
            jac_j,
            edge_src_tensor,
            edge_dst_tensor,
            residual,
            len(keyframe_c2w_init),
            damping,
            free_mask.to(dtype=linear_dtype),
            max_iterations=pcg_max_iterations,
            tolerance=pcg_tolerance,
            solve_dtype=linear_dtype,
            check_interval=pcg_check_interval,
            coarse_group_size=coarse_group_size,
        )
        torch.cuda.synchronize(torch_device)
        total_solve_sec += time.perf_counter() - solve_t0
        total_pcg_iterations += int(pcg_stats["iterations"])
        last_pcg_stats = pcg_stats

        candidate = inverse_log + delta
        new_cost = float(evaluate(candidate).square().mean().item())
        accepted = new_cost <= current_cost + 1e-12
        if accepted:
            inverse_log = candidate
            damping = max(damping / 2.0, 1e-12)
            accepted_steps += 1
        else:
            damping = min(damping * 2.0, 1e8)
            rejected_steps += 1

        log(
            f"GPU sparse PGO iter {outer_iteration + 1}/{max_iterations}: "
            f"cost {current_cost:.8e} -> {new_cost:.8e} "
            f"({'accepted' if accepted else 'rejected'}), "
            f"lambda={damping:.3e}, pcg={pcg_stats['iterations']}, "
            f"pcg_rel={pcg_stats['relative_residual']:.3e}"
        )

        relative_stop = (
            float(outer_relative_tolerance) > 0.0
            and accepted
            and outer_iteration >= 4
            and abs(current_cost - new_cost)
            <= float(outer_relative_tolerance) * max(current_cost, 1e-12)
        )
        cpu_compatible_stop = False
        if current_cost < 1e-6 and outer_iteration >= 4 and len(cost_history) >= 5:
            improvement = cost_history[-5] / max(cost_history[-1], 1e-12)
            cpu_compatible_stop = improvement < 1.2
        if relative_stop or cpu_compatible_stop:
            log(f"GPU sparse PGO converged at outer iteration {outer_iteration + 1}")
            break

    final_cost = float(evaluate(inverse_log).square().mean().item())
    optimized = pp.Exp(inverse_log).Inv()
    if model == "sim3":
        output = _sim3_batch_to_matrix(optimized)
    else:
        output = _se3_batch_to_matrix(optimized)

    if runtime_stats is not None:
        runtime_stats.update(
            {
                "gpu_pgo_backend": "matrix_free_block_sparse_pcg",
                "gpu_pgo_device": str(torch_device),
                "gpu_pgo_nodes": int(len(keyframe_c2w_init)),
                "gpu_pgo_edges": int(len(edge_measurements)),
                "gpu_pgo_outer_iterations": int(len(cost_history)),
                "gpu_pgo_accepted_steps": int(accepted_steps),
                "gpu_pgo_rejected_steps": int(rejected_steps),
                "gpu_pgo_total_pcg_iterations": int(total_pcg_iterations),
                "gpu_pgo_jacobian_sec": float(total_jacobian_sec),
                "gpu_pgo_linear_solve_sec": float(total_solve_sec),
                "gpu_pgo_initial_cost": float(cost_history[0]) if cost_history else 0.0,
                "gpu_pgo_final_cost": float(final_cost),
                "gpu_pgo_total_sec": float(time.perf_counter() - total_t0),
                "gpu_pgo_pcg_preconditioner": str(last_pcg_stats.get("preconditioner", "unknown")),
                "gpu_pgo_coarse_groups": int(last_pcg_stats.get("coarse_groups", 0)),
                "gpu_pgo_last_pcg_converged": bool(last_pcg_stats.get("converged", False)),
                "gpu_pgo_last_pcg_termination": str(
                    last_pcg_stats.get("termination_reason", "unknown")
                ),
                "gpu_pgo_last_pcg_relative_residual": float(
                    last_pcg_stats.get("relative_residual", 0.0)
                ),
            }
        )
    return output


def optimize_keyframe_pose_graph_gpu_sparse(
    keyframe_c2w_init: np.ndarray,
    odom_edges: Sequence[Any],
    loop_edges: Sequence[Any],
    cfg: Any,
    *,
    device: str = "cuda",
    pcg_max_iterations: int = 256,
    pcg_tolerance: float = 1e-5,
    pcg_check_interval: int = 8,
    coarse_group_size: int = 64,
    solve_dtype: str = "float64",
    outer_relative_tolerance: float = 0.0,
    runtime_stats: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    all_edges = list(odom_edges) + list(loop_edges)
    edge_weights = []
    for edge in all_edges:
        if edge.method == "odometry":
            edge_weights.append(1.0)
        else:
            edge_weights.append(
                float(cfg.pose_graph_loop_weight)
                * math.sqrt(max(int(edge.inliers), 1) / max(int(cfg.rigid_min_inliers), 1))
            )

    return optimize_pose_graph_gpu_sparse(
        keyframe_c2w_init=keyframe_c2w_init,
        edge_src=[int(edge.src_pos) for edge in all_edges],
        edge_dst=[int(edge.dst_pos) for edge in all_edges],
        edge_measurements=[edge.transform_ji for edge in all_edges],
        edge_weights=edge_weights,
        model=str(cfg.pose_graph_model),
        update_mode=str(cfg.pose_graph_update_mode),
        trans_weight=float(cfg.pose_graph_trans_weight),
        rot_weight=float(cfg.pose_graph_rot_weight),
        scale_weight=float(cfg.pose_graph_scale_weight),
        max_iterations=int(cfg.pose_graph_max_iterations),
        lambda_init=float(cfg.pose_graph_lambda_init),
        device=device,
        pcg_max_iterations=pcg_max_iterations,
        pcg_tolerance=pcg_tolerance,
        pcg_check_interval=pcg_check_interval,
        coarse_group_size=coarse_group_size,
        solve_dtype=solve_dtype,
        outer_relative_tolerance=outer_relative_tolerance,
        verbose=bool(cfg.pose_graph_solver_verbose),
        runtime_stats=runtime_stats,
    )

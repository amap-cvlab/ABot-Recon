"""Shared seq-id sampling helpers for mv_recon."""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig


def sample_ids_for_sequence(
    seq_num_frames: int,
    sample_config: DictConfig,
    seq_name: str = "",
    logger: logging.Logger | None = None,
) -> List[int] | None:
    """Return frame ids for one sequence, or None if the sequence should be skipped."""
    strategy = sample_config.strategy
    if strategy == "all":
        return np.arange(seq_num_frames).tolist()
    if strategy == "random_order":
        num_frames = int(sample_config.num_frames)
        if seq_num_frames < num_frames:
            if logger is not None:
                logger.warning(
                    f"sequence {seq_name} has only {seq_num_frames} frames < {num_frames}, skip..."
                )
            return None
        return np.random.choice(seq_num_frames, num_frames, replace=False).tolist()
    if strategy == "stride":
        return np.arange(0, seq_num_frames, int(sample_config.kf_every)).tolist()
    if strategy == "stride-half":
        return np.arange(0, seq_num_frames // 2, int(sample_config.kf_every)).tolist()
    if strategy == "first":
        assert int(sample_config.kf_every) == 1, "first strategy only supports kf_every=1"
        num_frames = int(sample_config.num_frames)
        assert num_frames <= seq_num_frames, (
            f"sequence {seq_name} has only {seq_num_frames} frames < {num_frames}"
        )
        return np.arange(0, num_frames).tolist()
    if strategy == "uniform":
        num_frames = int(sample_config.num_frames)
        assert num_frames <= seq_num_frames, (
            f"sequence {seq_name} has only {seq_num_frames} frames < {num_frames}"
        )
        return np.linspace(0, seq_num_frames - 1, num_frames, dtype=int).tolist()
    raise ValueError(f"Sampling strategy {strategy} is not implemented yet.")


def build_seq_id_map(dataset: Any, sample_config: DictConfig, logger: logging.Logger | None = None) -> Dict[str, List[int]]:
    """Build {seq_name: frame_ids} from dataset + sampling config."""
    seq_id_map: Dict[str, List[int]] = {}
    for seq_name in dataset.sequence_list:
        seq_num_frames = dataset.get_seq_framenum(sequence_name=seq_name)
        ids = sample_ids_for_sequence(seq_num_frames, sample_config, seq_name, logger)
        if ids is None:
            continue
        seq_id_map[seq_name] = ids
    return seq_id_map


def validate_seq_id_map(
    seq_id_map: Dict[str, List[int]],
    dataset: Any,
    sample_config: DictConfig,
) -> tuple[bool, str]:
    """Reject stale maps whose scenes, indices, or deterministic sampling differ."""
    expected_scenes = set(dataset.sequence_list)
    actual_scenes = set(seq_id_map)
    if actual_scenes != expected_scenes:
        return False, f"scene mismatch: map={sorted(actual_scenes)}, dataset={sorted(expected_scenes)}"

    for seq_name, ids in seq_id_map.items():
        n_frames = int(dataset.get_seq_framenum(sequence_name=seq_name))
        if any(not isinstance(i, int) or i < 0 or i >= n_frames for i in ids):
            return False, f"out-of-range frame id in {seq_name} (num_frames={n_frames})"

    if sample_config.strategy != "random_order":
        expected = build_seq_id_map(dataset, sample_config)
        if seq_id_map != expected:
            return False, "sampling or frame-count mismatch"
    return True, ""


def select_seq_id_map(
    seq_id_map: Dict[str, List[int]],
    requested_names: List[str] | None,
) -> Dict[str, List[int]]:
    """Select complete sequences in the requested order for parallel evaluation."""
    if requested_names is None:
        return seq_id_map
    names = [str(name) for name in requested_names]
    if not names:
        raise ValueError("eval_seq_names must contain at least one sequence")
    if len(names) != len(set(names)):
        raise ValueError(f"eval_seq_names contains duplicates: {names}")
    unknown = [name for name in names if name not in seq_id_map]
    if unknown:
        raise ValueError(
            f"Unknown eval sequence(s): {unknown}; available={list(seq_id_map)}"
        )
    return {name: seq_id_map[name] for name in names}


def ensure_seq_id_map(
    map_path: str,
    dataset: Any,
    sample_config: DictConfig,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> Dict[str, List[int]]:
    """Load seq-id-map from disk, or build+save if missing (or ``force``)."""
    map_path = str(map_path)
    if (not force) and osp.exists(map_path):
        with open(map_path, "r") as f:
            seq_id_map = json.load(f)
        valid, reason = validate_seq_id_map(seq_id_map, dataset, sample_config)
        if valid:
            if logger is not None:
                n_frames = sum(len(v) for v in seq_id_map.values())
                logger.info(
                    f"Loaded seq-id-map {map_path} "
                    f"({len(seq_id_map)} seqs, {n_frames} frames), "
                    f"sampling={sample_config}"
                )
            return seq_id_map
        raise ValueError(
            f"Invalid existing seq-id-map {map_path}: {reason}. "
            "Refusing to rebuild it implicitly; set rebuild_seq_id_map=true "
            "only when regenerating protocol files intentionally."
        )

    if logger is not None:
        logger.info(
            f"{'Rebuilding' if force else 'Creating'} seq-id-map at {map_path} "
            f"with sampling={sample_config}"
        )
    seq_id_map = build_seq_id_map(dataset, sample_config, logger=logger)
    os.makedirs(osp.dirname(map_path) or ".", exist_ok=True)
    with open(map_path, "w") as f:
        json.dump(seq_id_map, f, indent=4)
    if logger is not None:
        n_frames = sum(len(v) for v in seq_id_map.values())
        logger.info(
            f"Wrote seq-id-map {map_path} ({len(seq_id_map)} seqs, {n_frames} frames)"
        )
    return seq_id_map
